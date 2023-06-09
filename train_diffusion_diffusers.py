import os
import argparse
import torch
import numpy as np
import random

from PIL import Image
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms
from torch.nn import functional as F
from diffusers import UNet2DModel, DDIMScheduler
from tqdm import tqdm
from dataclasses import dataclass

from src.datasets.fashion_mnist import create_dataloader
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.diffusion_utils import evaluate_diffusion


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=3e-3)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--save_image_epochs", type=int, default=1)
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

    results_folder = Path("./results/diffusion/")
    results_folder.mkdir(exist_ok = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # model = UNet2DModel(
    #     sample_size=args.image_size,  # the target image resolution
    #     in_channels=args.channels,  # the number of input channels, 3 for RGB images
    #     out_channels=args.channels,  # the number of output channels
    #     layers_per_block=2,  # how many ResNet layers to use per UNet block
    #     block_out_channels=(8, 16),
    #     norm_num_groups=8,
    #     down_block_types=("DownBlock2D", "AttnDownBlock2D"),
    #     up_block_types=("AttnUpBlock2D", "UpBlock2D"),
    # )
    model = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.channels,  # the number of input channels, 3 for RGB images
        out_channels=args.channels,  # the number of output channels
        layers_per_block=1,  # how many ResNet layers to use per UNet block
        block_out_channels=(16, 32, 32, 64),
        norm_num_groups=16,
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    model = model.to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=args.timesteps)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataloader = create_dataloader(args.batch_size, preprocess)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")

        bar = tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader))
        
        for step, clean_images in bar:
            optimizer.zero_grad()

            batch_size = clean_images["pixel_values"].shape[0]
            clean_images = clean_images["pixel_values"].to(device)

            noise = torch.randn(clean_images.shape).to(clean_images.device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

            # sqrt_alphas_cumprod = (noise_scheduler.alphas_cumprod ** 0.5)
            # sqrt_one_minus_alpha_prod = (1 - noise_scheduler.alphas_cumprod) ** 0.5
            # alpha = _extract_into_tensor(sqrt_alphas_cumprod, timesteps, timesteps.shape)
            # sigma = _extract_into_tensor(sqrt_one_minus_alpha_prod, timesteps, timesteps.shape)
            # snr = (alpha / sigma) ** 2
            # k = 5
            # mse_loss_weight = torch.stack([snr, k * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
            # loss = mse_loss_weight * F.mse_loss(noise_pred, noise)
            # loss = loss.mean()

            loss = F.mse_loss(noise_pred, noise)
            loss.backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            bar.set_postfix(loss=loss.item())

        pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)

        if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            evaluate_diffusion(results_folder, args.eval_batch_size, epoch, pipeline, seed=args.seed)
            pipeline.save_pretrained(results_folder)


if __name__ == "__main__":
    args = __parse_args()
    main(args)