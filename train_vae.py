import os
import argparse
import torch
import numpy as np
import random

from matplotlib import pyplot as plt
from torch.optim import Adam
from torchvision import transforms
from tqdm import tqdm

from src.datasets.fashion_mnist import create_dataloader
from src.models.vae import MlpVAE, VAE_loss


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    results_folder = "./results/vae"
    os.makedirs(results_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MlpVAE(
        (1, args.image_size, args.image_size), 
        encoder_dims=(400, 400),
        decoder_dims=(400, 400),
        latent_dim=100,
        n_classes=10,
        device=device
    )
    model = model.to(device)
    print(model)

    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
    )

    preprocess = transforms.Compose(
        [
            # transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataloader = create_dataloader(args.batch_size, preprocess)

    for epoch in range(args.num_epochs):
        print(f"Epoch {epoch}")

        bar = tqdm(enumerate(dataloader), desc="Training loop", total=len(dataloader))
        avg_loss = 0.0
        
        for step, clean_images in bar:
            images = clean_images["pixel_values"].to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model(images)
            loss = VAE_loss(images, output)

            loss.backward()
            optimizer.step()

            bar.set_postfix(loss=loss.item())
            avg_loss += loss.item()

        print(f"Epoch completed. Average loss: {avg_loss/len(dataloader)}")

        # Save the images in the grid
        samples = model.generate(20)
        samples = samples.detach().cpu().numpy()
        fig, axs = plt.subplots(1, 20, figsize=(20, 1))
        for i in range(20):
            axs[i].imshow(samples[i][0], cmap="gray")
            axs[i].axis("off")
        plt.savefig(f"{results_folder}/GENERATOR_epoch_{epoch}.png")
        plt.close()
        

if __name__ == "__main__":
    args = __parse_args()
    main(args)