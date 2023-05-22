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
import json
import argparse
import torch
import torch.optim.lr_scheduler
import numpy as np

from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
# from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, GenerativeReplayPlugin

from src.continual_learning.strategies import GenerativeReplayWithDiffusion, DifussionTraining
from src.continual_learning.plugins import EfficientGenerativeReplayPlugin
from src.models.diffussion import DiffusionModel
from src.models.unet import Unet


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    return parser.parse_args()


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitMNIST(n_experiences=2, seed=42)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)
    denoise_model = Unet(
        dim=args.image_size, 
        channels=args.channels,
        dim_mults=(1, 2, 4,)
    )
    generator_model = DiffusionModel(denoise_model, (1, 28, 28), args.timesteps)

    # choose some metrics and evaluation method
    interactive_logger = InteractiveLogger()

    eval_plugin = EvaluationPlugin(
        accuracy_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        loss_metrics(minibatch=True, epoch=True, experience=True, stream=True),
        forgetting_metrics(experience=True),
        loggers=[interactive_logger],
    )

    # CREATE THE GENERATOR STRATEGY INSTANCE (DiffusionTraining)
    generator_strategy = DifussionTraining(
        generator_model,
        torch.optim.Adam(denoise_model.parameters(), lr=1e-4),
        train_mb_size=args.batch_size,
        train_epochs=1,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        timesteps=args.timesteps,
        plugins=[
            EfficientGenerativeReplayPlugin()
        ],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplayWithDiffusion(
        model,
        torch.optim.Adam(model.parameters(), lr=3e-3),
        CrossEntropyLoss(),
        train_mb_size=args.batch_size,
        train_epochs=4,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        generator_strategy=generator_strategy,
    )

    # OUTPUT DIRECTORY
    output_dir = "results/generative_replay/"
    os.makedirs(output_dir, exist_ok=True)

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    n_samples = 10
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))
        
        samples = cl_strategy.generator_strategy.model.generate(n_samples)
        # samples = (samples + 1) * 0.5
        samples = samples.detach().cpu().numpy()

        # Save plot of generated samples
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples, 1))
        for i in range(n_samples):
            axs[i].imshow(samples[i][0], cmap="gray")
            axs[i].axis("off")

        plt.savefig(os.path.join(output_dir, f"GENERATOR_output_exp_{experience.current_experience}.png"))

    print("Evaluation completed")

    # WRITE FINAL RESULTS TO A JSON FILE
    print("Saving results...")
    with open(os.path.join(output_dir, "metrics_per_epoch.json"), "w") as fp:
        json.dump(results, fp, indent=4)
        


if __name__ == "__main__":
    args = __parse_args()
    main(args)