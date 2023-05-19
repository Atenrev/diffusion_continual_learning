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
from avalanche.training.supervised import GenerativeReplay
from avalanche.evaluation.metrics import (
    forgetting_metrics,
    accuracy_metrics,
    loss_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin


def main(args):
    # --- CONFIG
    device = torch.device(
        f"cuda:{args.cuda}"
        if torch.cuda.is_available() and args.cuda >= 0
        else "cpu"
    )

    # --- BENCHMARK CREATION
    benchmark = SplitMNIST(n_experiences=5, seed=42)
    # ---------

    # MODEL CREATION
    model = SimpleMLP(num_classes=benchmark.n_classes)

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

    # CREATE THE STRATEGY INSTANCE (GenerativeReplay)
    cl_strategy = GenerativeReplay(
        model,
        torch.optim.Adam(model.parameters(), lr=0.003),
        CrossEntropyLoss(),
        train_mb_size=256,
        train_epochs=4,
        eval_mb_size=256,
        device=device,
        evaluator=eval_plugin,
    )

    # TRAINING LOOP
    print("Starting experiment...")
    results = []
    f, axarr = plt.subplots(benchmark.n_experiences, 10)
    k = 0
    for experience in benchmark.train_stream:
        print("Start of experience ", experience.current_experience)
        cl_strategy.train(experience)
        print("Training completed")

        print("Computing accuracy on the whole test set")
        results.append(cl_strategy.eval(benchmark.test_stream))
        
        samples = cl_strategy.generator_strategy.model.generate(10)
        samples = samples.detach().cpu().numpy()

        for j in range(10):
            axarr[k, j].imshow(samples[j, 0], cmap="gray")
            axarr[k, 4].set_title("Generated images for experience " + str(k))

        np.vectorize(lambda ax: ax.axis("off"))(axarr)
        k += 1

    print("Evaluation completed")

    # WRITE FINAL RESULTS TO A JSON FILE
    output_dir = "results/generative_replay/"
    os.makedirs(output_dir, exist_ok=True)

    print("Saving results...")
    with open(os.path.join(output_dir, "metrics_per_epoch.json"), "w") as fp:
        json.dump(results, fp, indent=4)
    
    f.subplots_adjust(hspace=1.2)
    plt.savefig(os.path.join(output_dir, "GENERATOR_output_per_exp"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuda",
        type=int,
        default=0,
        help="Select zero-indexed cuda device. -1 to use CPU.",
    )
    args = parser.parse_args()
    main(args)