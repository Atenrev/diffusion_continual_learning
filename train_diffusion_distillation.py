import os
import argparse
import torch
import numpy as np
import random

from PIL import Image
from pathlib import Path
from torch.optim import Adam
from torch.nn import functional as F
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm

from src.pipelines.pipeline_ddim import DDIMPipeline
from src.common.diffusion_utils import evaluate_diffusion


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--distillation_type", type=str, default="generation")
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--generation_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--num_iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--save_image_epochs", type=int, default=50)
    return parser.parse_args()


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    experiment_name = f"diffusion_distillation_{args.distillation_type}_{args.seed}"
    results_folder = Path(f"./results/diffusion_distillation/{experiment_name}")
    results_folder.mkdir(exist_ok = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    teacher_pipeline = DDIMPipeline.from_pretrained("./results/diffusion").to("cuda")
    teacher_pipeline.set_progress_bar_config(disable=True)
    teacher = teacher_pipeline.unet.to(device)

    student = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.channels,  # the number of input channels, 3 for RGB images
        out_channels=args.channels,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=(16, 32, 32, 64),
        norm_num_groups=16,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    student = student.to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=args.timesteps)

    optimizer = Adam(student.parameters(), lr=args.learning_rate)

    bar = tqdm(range(args.num_iterations), desc="Training loop", total=args.num_iterations)
    for iteration in bar:
        optimizer.zero_grad()

        noise = torch.randn((args.batch_size, args.channels, args.image_size, args.image_size)).to(device)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (args.batch_size,), device=device
        ).long()

        if args.distillation_type == "gaussian":
            target = teacher(noise, timesteps, return_dict=False)[0]
            student_pred = student(noise, timesteps, return_dict=False)[0]
        elif args.distillation_type == "generation":
            generated_images = teacher_pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.generation_steps,
                eta=args.eta,
                output_type="np.array", 
            ).images
            generated_images = torch.from_numpy(generated_images).to(device)
            generated_images = generated_images.permute(0, 3, 1, 2)
            noisy_images = noise_scheduler.add_noise(generated_images, noise, timesteps)
            target = teacher(noisy_images, timesteps, return_dict=False)[0]
            student_pred = student(noisy_images, timesteps, return_dict=False)[0]
        elif args.distillation_type == "partial_generation":
            raise NotImplementedError
        elif args.distillation_type == "none":
            target = noise
            generated_images = teacher_pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.generation_steps,
                eta=args.eta,
            output_type="np.array", 
            ).images
            generated_images = torch.from_numpy(generated_images).to(device)
            generated_images = generated_images.permute(0, 3, 1, 2)
            noisy_images = noise_scheduler.add_noise(generated_images, noise, timesteps)
            student_pred = student(noisy_images, timesteps, return_dict=False)[0]

        sqrt_alphas_cumprod = (noise_scheduler.alphas_cumprod ** 0.5)
        sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod) ** 0.5
        alpha = _extract_into_tensor(sqrt_alphas_cumprod, timesteps, timesteps.shape)
        sigma = _extract_into_tensor(sqrt_one_minus_alpha_prod, timesteps, timesteps.shape)
        snr = (alpha / sigma) ** 2
        k = 5
        mse_loss_weight = torch.stack([snr, k * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr

        loss = mse_loss_weight * F.mse_loss(target, student_pred)
        loss = loss.sum()
        # loss = F.mse_loss(target, noise_pred)
        loss.backward()
        # Clip gradients to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)

        optimizer.step()

        bar.set_postfix(loss=loss.item())

        pipeline = DDIMPipeline(unet=student, scheduler=noise_scheduler)

        if (iteration + 1) % args.save_image_epochs == 0 or iteration == args.num_iterations - 1:
            evaluate_diffusion(results_folder, args.eval_batch_size, iteration, pipeline, seed=args.seed)
            pipeline.save_pretrained(results_folder)


if __name__ == "__main__":
    args = __parse_args()
    main(args)