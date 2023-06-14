import os
import argparse
import torch
import numpy as np
import random
import json

from copy import deepcopy
from torch.optim import Adam
from torchvision import transforms
from diffusers import UNet2DModel, DDIMScheduler

from src.datasets.fashion_mnist import create_dataloader as create_fashion_mnist_dataloader
from src.datasets.mnist import create_dataloader as create_mnist_dataloader
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.losses.diffusion_losses import MSELoss, MinSNRLoss
from src.trainers.diffusion_distillation import (
    GaussianDistillation,
    PartialGenerationDistillation,
    GenerationDistillation,
    NoDistillation
)
from src.evaluators.generative_evaluator import GenerativeModelEvaluator
from src.common.visual import plot_line_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 28 for vae, 32 for unet
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="mnist")

    parser.add_argument("--model_config_path", type=str,
                        default="configs/model/diffusion.json")
    parser.add_argument("--distillation_type", type=str, default="generation",
                        help="Type of distillation to use (gaussian, generation, partial_generation, no_distillation)")
    parser.add_argument("--teacher_path", type=str, default="results/diffusion_None_mse_42",
                        help="Path to teacher model (only for distillation)")
    parser.add_argument("--criterion", type=str, default="mse",
                        help="Criterion to use for training (mse, min_snr)")

    parser.add_argument("--generation_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--teacher_eta", type=float, default=1.0)

    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--eval_batch_size", type=int, default=256)

    parser.add_argument("--save_every", type=int, default=1,
                        help="Save model every n iterations (only for distillation)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_name = f"fid_vs_gensteps_{args.distillation_type}_{args.criterion}_eta_{args.teacher_eta}_{args.seed}"
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

    if args.criterion == "mse":
        criterion = MSELoss(noise_scheduler)
    elif args.criterion == "min_snr":
        criterion = MinSNRLoss(noise_scheduler)
    else:
        raise NotImplementedError

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

    assert args.teacher_path is not None
    teacher_pipeline = DDIMPipeline.from_pretrained(args.teacher_path)
    teacher_pipeline.set_progress_bar_config(disable=True)
    teacher = teacher_pipeline.unet.to(device)

    fid_list = []
    gen_steps = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    for gen_step in gen_steps:
        print(
            f"\n\n======= Training with {gen_step} generation steps =======\n")
        wrap_in_pipeline(teacher, noise_scheduler,
                         DDIMPipeline, gen_step, args.teacher_eta, output_type="torch_raw")
        student = deepcopy(model).to(device)
        wrap_in_pipeline(student, noise_scheduler, DDIMPipeline,
                         args.generation_steps, args.eta)
        optimizer = Adam(student.parameters(), lr=model_config.optimizer.lr)

        trainer = trainer_class(
            student=student,
            scheduler=noise_scheduler,
            optimizer=optimizer,
            criterion=criterion,
            train_mb_size=args.batch_size,
            train_iterations=args.num_epochs,
            eval_mb_size=args.eval_batch_size,
            device=device,
            train_timesteps=model_config.scheduler.train_timesteps,
            evaluator=None
        )

        trainer.train(teacher, eval_loader=test_dataloader,
                      save_every=args.save_every, save_path=results_folder)
        print(f"\n=== Evaluating with {gen_step} generation steps ===\n")
        fid = evaluator.evaluate(student, test_dataloader, gen_step)["fid"]
        fid_list.append(fid)

    # Save results as json
    results = {
        "config": {
            "args": vars(args),
            "model_config": model_config,
        },
        "results": {
            "fid_list": fid_list,
            "gen_steps": gen_steps
        }
    }

    with open(os.path.join(results_folder, "results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save graph
    plot_line_graph(
        gen_steps,
        fid_list,
        "Teacher's Generation Steps",
        "FID",
        "FID vs Teacher's Generation Steps",
        os.path.join(results_folder, "fid_vs_gensteps.png")
    )


if __name__ == "__main__":
    args = __parse_args()
    main(args)
