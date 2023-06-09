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
import datetime

from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from avalanche.benchmarks import SplitMNIST
from avalanche.models import SimpleMLP
# from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
    cpu_usage_metrics,
    timing_metrics,
    gpu_usage_metrics,
    ram_usage_metrics,
    disk_usage_metrics,
    MAC_metrics,
    confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger, WandBLogger
from avalanche.training.plugins import EvaluationPlugin

from src.continual_learning.strategies import UpdatedGenerativeReplay, DiffusionTraining
from src.continual_learning.plugins import UpdatedGenerativeReplayPlugin
from src.models.diffusion_old import DiffusionModel
from src.models.unet import Unet


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--timesteps", type=int, default=500)
    parser.add_argument("--epochs_generator", type=int, default=16)
    parser.add_argument("--epochs_solver", type=int, default=4)
    parser.add_argument("--generator_lr", type=float, default=3e-4)
    parser.add_argument("--solver_lr", type=float, default=3e-3)
    parser.add_argument("--increasing_replay_size", type=bool, default=False)
    parser.add_argument("--replay_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    parser.add_argument("--debug", action="store_true", default=False)
    return parser.parse_args()


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitMNIST(n_experiences=5, seed=args.seed)
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
    run_name = "generative_replay_diffusion_baseline"
    run_name += f"_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    if args.debug:
        project_name = "master-thesis-debug"
    else:
        project_name = "master-thesis"
    interactive_logger = InteractiveLogger()
    wandb_logger = WandBLogger(
        project_name=project_name, 
        run_name=run_name, 
        config=vars(args)
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
        cpu_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        timing_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        ram_usage_metrics(
            every=0.5, minibatch=True, epoch=True, experience=True, stream=True
        ),
        gpu_usage_metrics(
            args.cuda,
            every=0.5,
            minibatch=True,
            epoch=True,
            experience=True,
            stream=True,
        ),
        disk_usage_metrics(
            minibatch=True, epoch=True, experience=True, stream=True
        ),
        MAC_metrics(minibatch=True, epoch=True, experience=True),
        loggers=[interactive_logger, wandb_logger],
    )

    # CREATE THE GENERATOR STRATEGY INSTANCE (DiffusionTraining)
    generator_strategy = DiffusionTraining(
        generator_model,
        torch.optim.Adam(denoise_model.parameters(), lr=args.generator_lr),
        train_mb_size=args.batch_size,
        train_epochs=args.epochs_generator,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        train_timesteps=args.timesteps,
        plugins=[
            UpdatedGenerativeReplayPlugin(
                increasing_replay_size=args.increasing_replay_size,
                replay_size=args.replay_size,
            )
        ],
    )

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = UpdatedGenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=args.solver_lr),
        CrossEntropyLoss(),
        train_mb_size=args.batch_size, # Caution: the batch size is doubled because of the replay
        train_epochs=args.epochs_solver,
        eval_mb_size=args.batch_size,
        device=device,
        evaluator=eval_plugin,
        generator_strategy=generator_strategy,
        increasing_replay_size=args.increasing_replay_size,
        replay_size=args.replay_size,
    )

    # OUTPUT DIRECTORY
    output_dir = f"results/generative_replay/diffusion_baseline/{run_name}"
    os.makedirs(output_dir, exist_ok=True)

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    n_samples = 20
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))
        
        print("Computing generated samples and saving them to disk")
        with torch.no_grad():
            samples = cl_strategy.generator_strategy.model.generate(n_samples)

        samples = samples.detach().cpu().numpy()

        # Save plot of generated samples
        fig, axs = plt.subplots(1, n_samples, figsize=(n_samples, 1))
        for i in range(n_samples):
            axs[i].imshow(samples[i][0], cmap="gray")
            axs[i].axis("off")

        plt.savefig(os.path.join(output_dir, f"GENERATOR_output_exp_{experience.current_experience}.png"))

    print("Evaluation completed")

    # WRITE FINAL RESULTS TO A JSON FILE
    # print("Saving results...")
    # with open(os.path.join(output_dir, "metrics_per_epoch.json"), "w") as fp:
    #     json.dump(results, fp, indent=4)
        


if __name__ == "__main__":
    args = __parse_args()
    main(args)