################################################################################
# Copyright (c) 2021 ContinualAI.                                              #
# Copyrights licensed under the MIT License.                                   #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-04-2022                                                             #
# Author(s): Florian Mies                                                      #
# E-mail: contact@continualai.org                                              #
# Website: avalanche.continualai.org                                           #
################################################################################

"""
This is a simple example on how to use the Replay strategy.
"""

import os
import argparse
import torch
import torch.optim.lr_scheduler
import numpy as np
import datetime

from matplotlib import pyplot as plt
from torch.optim import Adam
from torchvision import transforms
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    confusion_matrix_metrics,
)
from avalanche.logging import WandBLogger, InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin
 
from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin
from src.continual_learning.strategies import WeightedSoftGenerativeReplay, VAETraining
from src.continual_learning.metrics import ExperienceFIDMetric
from src.models.vae import MlpVAE, VAE_loss


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--epochs_generator", type=int, default=1) # 2000*128/12000 = ~21
    parser.add_argument("--epochs_solver", type=int, default=1) 
    parser.add_argument("--generator_lr", type=float, default=0.001)
    parser.add_argument("--solver_lr", type=float, default=0.001)
    parser.add_argument("--increasing_replay_size", type=bool, default=False)
    parser.add_argument("--replay_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--debug", action="store_true", default=True)
    return parser.parse_args()


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size), antialias=True),
        ]
    )
    benchmark = SplitMNIST(
        n_experiences=5, 
        seed=args.seed,
        train_transform=train_transform,
        eval_transform=train_transform,
    )
    # ---------

    run_name = "generative_replay_vae_baseline"
    run_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if args.debug:
        project_name = "master-thesis-debug"
    else:
        project_name = "master-thesis"
    wandb_logger = WandBLogger(
        project_name=project_name, 
        run_name=run_name, 
        config=vars(args)
    )

    # GENERATOR STRATEGY
    generator = MlpVAE(
        (1, args.image_size, args.image_size), 
        encoder_dims=(400, 400),
        decoder_dims=(400, 400),
        latent_dim=100,
        n_classes=benchmark.n_classes,
        device=device
    )
    optimizer_generator = Adam(
        generator.parameters(),
        lr=args.generator_lr,
        betas=(0.9, 0.999),
    )

    gen_eval_plugin = EvaluationPlugin(
        ExperienceFIDMetric(),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loggers=[wandb_logger, InteractiveLogger()],
    )

    generator_strategy = VAETraining(
        model=generator,
        optimizer=optimizer_generator,
        criterion=VAE_loss,
        train_mb_size=args.batch_size,
        train_epochs=args.epochs_generator,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=gen_eval_plugin,
        plugins=[
            UpdatedGenerativeReplayPlugin(
                replay_size=None,
                increasing_replay_size=False,
            )
        ],
    )

    # CLASSIFIER STRATEGY
    model = SimpleMLP(num_classes=benchmark.n_classes)
    optimizer_classifier = Adam(
        model.parameters(),
        lr=args.solver_lr,
        betas=(0.9, 0.999),
    )

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        loss_metrics(
            minibatch=True,
            epoch=True,
            epoch_running=True,
            experience=True,
            stream=True,
        ),
        forgetting_metrics(experience=True, stream=True),
        confusion_matrix_metrics(
            stream=True, wandb=True, class_names=[str(i) for i in range(10)]
        ),
        loggers=[wandb_logger],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = WeightedSoftGenerativeReplay(
        model,
        optimizer_classifier,
        CrossEntropyLoss(),
        generator_strategy=generator_strategy,
        train_mb_size=args.batch_size,
        train_epochs=args.epochs_solver,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    output_dir = "results/generative_replay/vae_baseline"
    os.makedirs(output_dir, exist_ok=True)
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        cl_strategy.eval(benchmark.test_stream)
        generator_strategy.eval(benchmark.test_stream)
        
        samples = cl_strategy.generator_strategy.model.generate(20)
        samples = samples.detach().cpu().numpy()

        # Save the images in the grid
        fig, axs = plt.subplots(1, 20, figsize=(20, 1))
        for i in range(20):
            axs[i].imshow(samples[i][0], cmap="gray")
            axs[i].axis("off")
        plt.savefig(f"{output_dir}/GENERATOR_exp_{experience.current_experience}.png")
        plt.close()

        print("Evaluation completed")


if __name__ == "__main__":
    args = __parse_args()
    main(args)