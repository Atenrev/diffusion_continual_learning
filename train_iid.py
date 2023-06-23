import os
import argparse
import torch
import numpy as np
import random

from torch.optim import Adam
from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler

from src.datasets.fashion_mnist import create_dataloader as create_fashion_mnist_dataloader
from src.datasets.mnist import create_dataloader as create_mnist_dataloader
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.models.vae import MlpVAE, VAE_loss
from src.losses.diffusion_losses import MSELoss, MinSNRLoss, SmoothL1Loss
from src.trainers.diffusion_training import DiffusionTraining
from src.trainers.diffusion_distillation import (
    GaussianDistillation,
    PartialGenerationDistillation,
    GenerationDistillation,
    NoDistillation
)
from src.trainers.generative_training import GenerativeTraining
from src.evaluators.generative_evaluator import GenerativeModelEvaluator


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 28 for vae, 32 for unet
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="fashion_mnist")

    parser.add_argument("--model_config_path", type=str,
                        default="configs/model/ddim_minimal.json")
    parser.add_argument("--training_type", type=str, default="diffusion",
                        help="Type of training to use (diffusion, generative)")
    parser.add_argument("--distillation_type", type=str, default=None,
                        help="Type of distillation to use (gaussian, generation, partial_generation, no_distillation)")
    parser.add_argument("--teacher_path", type=str, default=None, #"results/ddim_32_diffusion_None_mse_42",
                        help="Path to teacher model (only for distillation)")
    parser.add_argument("--criterion", type=str, default="mse",
                        help="Criterion to use for training (smooth_l1, mse, min_snr)")

    parser.add_argument("--generation_steps", type=int, default=20)
    parser.add_argument("--teacher_generation_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)

    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)

    parser.add_argument("--save_every", type=int, default=1,
                        help="Save model every n iterations (only for distillation)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    model_name = args.model_config_path.split("/")[-1].split(".")[0]
    run_name = f"{model_name}_{args.training_type}_{args.distillation_type}_{args.criterion}_{args.seed}"
    results_folder = os.path.join("results", run_name)
    os.makedirs(results_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    else:
        raise NotImplementedError

    model_config = get_configuration(args.model_config_path)

    evaluator = GenerativeModelEvaluator(
        device=device, save_images=20, save_path=results_folder)

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
                evaluator=evaluator
            )
            trainer.train(train_dataloader, test_dataloader,
                          save_path=results_folder)

        else:
            assert args.teacher_path is not None
            teacher_pipeline = DDIMPipeline.from_pretrained(args.teacher_path)
            teacher_pipeline.set_progress_bar_config(disable=True)
            teacher = teacher_pipeline.unet.to(device)
            wrap_in_pipeline(teacher, noise_scheduler, DDIMPipeline,
                             args.teacher_generation_steps, args.eta)

            if args.distillation_type == "gaussian":
                trainer_class = GaussianDistillation
            elif args.distillation_type == "generation":
                trainer_class = GenerationDistillation
            elif args.distillation_type == "partial_generation":
                trainer_class = PartialGenerationDistillation
            elif args.distillation_type == "no_distillation":
                trainer_class = NoDistillation
            else:
                raise NotImplementedError

            trainer = trainer_class(
                student=model,
                scheduler=noise_scheduler,
                optimizer=optimizer,
                criterion=criterion,
                train_mb_size=args.batch_size,
                train_iterations=args.num_epochs,
                eval_mb_size=args.eval_batch_size,
                device=device,
                train_timesteps=model_config.scheduler.train_timesteps,
                evaluator=evaluator
            )

            trainer.train(teacher, eval_loader=test_dataloader,
                          save_every=args.save_every, save_path=results_folder)

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

    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = __parse_args()
    main(args)
