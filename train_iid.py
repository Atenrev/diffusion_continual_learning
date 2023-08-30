import os
import argparse
import torch
import numpy as np
import random
import pandas as pd

from torch.optim import Adam
from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler

from src.datasets.fashion_mnist import create_dataloader as create_fashion_mnist_dataloader
from src.datasets.mnist import create_dataloader as create_mnist_dataloader
from src.datasets.cifar100 import create_dataloader as create_cifar100_dataloader 
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.models.vae import MlpVAE, VAE_loss
from src.losses.diffusion_losses import MSELoss, MinSNRLoss, SmoothL1Loss
from src.trainers.diffusion_training import DiffusionTraining
from src.trainers.diffusion_distillation import (
    GaussianDistillation,
    GaussianSymmetryDistillation,
    PartialGenerationDistillation,
    GenerationDistillation,
    NoDistillation
)
from src.trainers.generative_training import GenerativeTraining
from src.evaluators.generative_evaluator import GenerativeModelEvaluator
from src.trackers.wandb_tracker import WandbTracker
from src.trackers.csv_tracker import CSVTracker


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="fashion_mnist")

    parser.add_argument("--model_config_path", type=str,
                        default="configs/model/ddim_medium.json")
    parser.add_argument("--training_type", type=str, default="evaluate",
                        help="Type of training to use (evaluate, diffusion, generative)")
    parser.add_argument("--distillation_type", type=str, default=None,
                        help="Type of distillation to use (gaussian, gaussian_symmetry, generation, partial_generation, no_distillation)")
    parser.add_argument("--teacher_path", type=str, default="results_fuji/smasipca/iid_results/comparison/diffusion/no_distillation/ddim_medium_mse_teacher_20_eta_0.0/69/last_model",
                        help="Path to teacher model (only for distillation)")
    parser.add_argument("--criterion", type=str, default="mse",
                        help="Criterion to use for training (smooth_l1, mse, min_snr)")

    parser.add_argument("--generation_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--teacher_generation_steps", type=int, default=2)
    parser.add_argument("--teacher_eta", type=float, default=0.0)

    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--results_folder", type=str, default="/esat/fuji/smasipca/iid_results")
    parser.add_argument("--save_every", type=int, default=5,
                        help="Evaluate and save model every n epochs (normal) or n iterations (distillation)")
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def run_experiment(args, device, model_config, tracker, results_folder):
    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    if args.dataset == "mnist":
        train_dataloader, test_dataloader = create_mnist_dataloader(
            args.batch_size, preprocess)
    elif args.dataset == "fashion_mnist":
        train_dataloader, test_dataloader = create_fashion_mnist_dataloader(
            args.batch_size, preprocess)
    elif args.dataset == "cifar100":
        train_dataloader, test_dataloader = create_cifar100_dataloader(
            args.batch_size, preprocess)
    else:
        raise NotImplementedError

    evaluator = GenerativeModelEvaluator(
        device=device, save_images=100, save_path=results_folder)

    if args.training_type == "diffusion":
        model = UNet2DModel(
            sample_size=model_config.model.input_size,
            in_channels=model_config.model.in_channels,
            out_channels=model_config.model.out_channels,
            layers_per_block=model_config.model.layers_per_block,
            block_out_channels=model_config.model.block_out_channels,
            norm_num_groups=model_config.model.norm_num_groups,
            down_block_types=model_config.model.down_block_types,
            up_block_types=model_config.model.up_block_types,
        )
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=model_config.scheduler.train_timesteps)
        wrap_in_pipeline(model, noise_scheduler,
                         DDIMPipeline, args.generation_steps, args.eta)
        model = model.to(device)
        print("Number of parameters:", sum(p.numel()
                                             for p in model.parameters() if p.requires_grad))

        optimizer = Adam(model.parameters(), lr=model_config.optimizer.lr)

        # scheduler = OneCycleLR(optimizer, args.lr, total_steps=args.epochs*len(train_dataloader), pct_start=0.25, anneal_strategy='cos')

        if args.criterion == "mse":
            criterion = MSELoss(noise_scheduler)
        elif args.criterion == "min_snr":
            criterion = MinSNRLoss(noise_scheduler)
        elif args.criterion == "smooth_l1":
            criterion = SmoothL1Loss(noise_scheduler)
        else:
            raise NotImplementedError

        if args.distillation_type is None:
            trainer = DiffusionTraining(
                model=model,
                scheduler=noise_scheduler,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=args.batch_size,
                train_epochs=args.num_epochs,
                eval_mb_size=args.eval_batch_size,
                device=device,
                train_timesteps=model_config.scheduler.train_timesteps,
                evaluator=evaluator,
                tracker=tracker,
            )
            trainer.train(train_dataloader, test_dataloader,
                          save_path=results_folder, save_every=args.save_every)

        else:
            assert args.teacher_path is not None
            teacher_pipeline = DDIMPipeline.from_pretrained(args.teacher_path)
            teacher_pipeline.set_progress_bar_config(disable=True)
            teacher = teacher_pipeline.unet.to(device)
            wrap_in_pipeline(teacher, noise_scheduler, DDIMPipeline,
                             args.teacher_generation_steps, args.teacher_eta, def_output_type="torch_raw")

            if args.distillation_type == "gaussian":
                trainer_class = GaussianDistillation
            elif args.distillation_type == "gaussian_symmetry":
                trainer_class = GaussianSymmetryDistillation
            elif args.distillation_type == "generation":
                trainer_class = GenerationDistillation
            elif args.distillation_type == "partial_generation":
                trainer_class = PartialGenerationDistillation
            elif args.distillation_type == "no_distillation":
                trainer_class = NoDistillation
            else:
                raise NotImplementedError

            trainer = trainer_class(
                model=model,
                scheduler=noise_scheduler,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=args.batch_size,
                train_iterations=args.num_epochs,
                eval_mb_size=args.eval_batch_size,
                device=device,
                train_timesteps=model_config.scheduler.train_timesteps,
                evaluator=evaluator,
                tracker=tracker,
            )

            trainer.train(teacher, test_dataloader,
                          save_path=results_folder, save_every=args.save_every)

    elif args.training_type == "generative":
        model = MlpVAE(
            (model_config.model.channels, model_config.model.input_size,
             model_config.model.input_size),
            encoder_dims=model_config.model.encoder_dims,
            decoder_dims=model_config.model.decoder_dims,
            latent_dim=model_config.model.latent_dim,
            n_classes=model_config.model.n_classes,
            device=device
        )
        model = model.to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=model_config.optimizer.lr,
            betas=(0.9, 0.999),
        )
        trainer = GenerativeTraining(
            model=model,
            optimizer=optimizer,
            criterion=VAE_loss,
            train_mb_size=args.batch_size,
            train_epochs=args.num_epochs,
            eval_mb_size=args.eval_batch_size,
            device=device,
            evaluator=evaluator
        )
        trainer.train(train_dataloader, test_dataloader,
                      save_path=results_folder)

    elif args.training_type == "evaluate":
        model_pipeline = DDIMPipeline.from_pretrained(args.teacher_path)
        model_pipeline.set_progress_bar_config(disable=True)
        model = model_pipeline.unet.to(device)
        wrap_in_pipeline(model, model_pipeline.scheduler, DDIMPipeline,
                            args.teacher_generation_steps, args.eta)
        evaluator.evaluate(model, test_dataloader, gensteps=args.generation_steps, compute_auc=False, fid_images=0)

    else:
        raise NotImplementedError
    

def main(args):
    model_name = args.model_config_path.split("/")[-1].split(".")[0]
    run_name = f"{args.dataset}/{args.training_type}/{args.distillation_type}/{model_name}_{args.criterion}"
    if args.distillation_type is not None:
        run_name += f"_teacher_steps_{args.teacher_generation_steps}_eta_{args.teacher_eta}"
    if args.seed is not None:
        run_name += f"_{args.seed}"
    results_folder = os.path.join(args.results_folder, run_name)
    os.makedirs(results_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_config = get_configuration(args.model_config_path)
        
    all_configs = {
        "model_config": model_config,
        "args": vars(args)
    }

    if args.use_wandb and args.training_type != "evaluate":
        tracker = WandbTracker(
            configs=all_configs,
            experiment_name=run_name.split("/")[-1],
            project_name=f"master-thesis-{args.dataset}-{args.training_type}-{args.distillation_type}",
            tags=[args.dataset, args.training_type, str(args.distillation_type), model_name, args.criterion],
        )
    else:
        tracker = None

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        run_experiment(args, device, model_config, tracker, results_folder)
    else:
        assert not args.use_wandb, "Cannot use wandb with multiple seeds"
         
        for seed in [42, 69, 420, 666, 1714]:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            results_seed_folder = os.path.join(results_folder, str(seed))
            os.makedirs(results_seed_folder, exist_ok=True)
            tracker = CSVTracker(all_configs, results_seed_folder)
            run_experiment(args, device, model_config, tracker, results_seed_folder)

        # Check the best model for each seed and print the best one
        best_auc = torch.inf
        best_epoch = None
        best_seed = None
        for seed in [42, 69, 420, 666, 1714]:
            results_seed_folder = os.path.join(results_folder, str(seed))
            csv_file = open(os.path.join(results_seed_folder, "test.csv"), "r")
            df = pd.read_csv(csv_file)

            # Get the row with the best AUC
            row = df.loc[df["metric"] == "auc"].sort_values(by=["value"]).iloc[0]
            auc = row["value"]
            epoch = row["epoch"]

            if auc < best_auc:
                best_auc = auc
                best_epoch = epoch
                best_seed = seed

        print(f"Best seed: {best_seed} at epoch {best_epoch} with AUC {best_auc}")


if __name__ == "__main__":
    args = __parse_args()
    main(args)
