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
from diffusers import UNet2DModel, DDIMScheduler, DDIMPipeline
from tqdm import tqdm
from dataclasses import dataclass

from src.datasets.fashion_mnist import create_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--save_image_epochs", type=int, default=2)
    return parser.parse_args()


def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def evaluate(output_dir, eval_batch_size, epoch, pipeline, seed: int = 42):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=eval_batch_size,
        generator=torch.manual_seed(seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_folder = Path("./results/diffusion/")
    results_folder.mkdir(exist_ok = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet2DModel(
        sample_size=args.image_size,  # the target image resolution
        in_channels=args.channels,  # the number of input channels, 3 for RGB images
        out_channels=args.channels,  # the number of output channels
        layers_per_block=2,  # how many ResNet layers to use per UNet block
        block_out_channels=(32, 64, 64, 128),
        down_block_types=("DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    )
    model = model.to(device)

    noise_scheduler = DDIMScheduler(num_train_timesteps=args.timesteps)

    optimizer = Adam(model.parameters(), lr=args.learning_rate)

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            # transforms.ToTensor(),
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
            loss = F.mse_loss(noise_pred, noise)

            loss.backward()
            # Clip gradients to avoid exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()

            bar.set_postfix(loss=loss.item())

        pipeline = DDIMPipeline(unet=model, scheduler=noise_scheduler)

        if (epoch + 1) % args.save_image_epochs == 0 or epoch == args.num_epochs - 1:
            evaluate(results_folder, args.eval_batch_size, epoch, pipeline, seed=args.seed)
            pipeline.save_pretrained(results_folder)


if __name__ == "__main__":
    args = __parse_args()
    main(args)