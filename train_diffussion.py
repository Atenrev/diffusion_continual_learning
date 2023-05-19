import argparse
import torch
import numpy as np
import random

from pathlib import Path
from torch.optim import Adam
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.unet import Unet
from src.models.diffussion import DiffusionModel
from src.datasets.fashion_mnist import create_dataloader


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_folder = Path("./results/diffussion/")
    results_folder.mkdir(exist_ok = True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoise_model = Unet(
        dim=args.image_size,
        channels=args.channels,
        dim_mults=(1, 2, 4,)
    )
    denoise_model.to(device)

    difussion_model = DiffusionModel(denoise_model, args.timesteps)

    optimizer = Adam(denoise_model.parameters(), lr=1e-3)

    dataloader = create_dataloader(args.batch_size)

    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")

        bar = tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader))
        
        for step, batch in bar:
            optimizer.zero_grad()

            batch_size = batch["pixel_values"].shape[0]
            batch = batch["pixel_values"].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, args.timesteps, (batch_size,), device=device).long()

            loss = difussion_model.p_losses(batch, t, loss_type="huber")

            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())

        # Sample from the model
        all_images = difussion_model.generate(args.image_size, batch_size=4, channels=args.channels, timesteps=args.timesteps)
        all_images = (all_images + 1) * 0.5
        save_image(all_images, str(results_folder / f'sample-{epoch}.png'), nrow = 6)


if __name__ == "__main__":
    args = __parse_args()
    main(args)