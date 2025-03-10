import os
import argparse
import torch
import numpy as np
import random
import json

from torchvision import transforms
from diffusers import DDIMScheduler

import sys
from pathlib import Path
# This script should be run from the root of the project
sys.path.append(str(Path(__file__).parent.parent))

from src.datasets.fashion_mnist import create_dataloader as create_fashion_mnist_dataloader
from src.datasets.mnist import create_dataloader as create_mnist_dataloader
from src.common.utils import get_configuration
from src.common.diffusion_utils import wrap_in_pipeline
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.standard_training.evaluators.generative_evaluator import GenerativeModelEvaluator
from src.common.visual import plot_line_std_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    # 28 for vae, 32 for unet
    parser.add_argument("--image_size", type=int, default=32)
    parser.add_argument("--channels", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="fashion_mnist")

    parser.add_argument("--model_config_path", type=str,
                        default="configs/model/ddim_medium.json")
    parser.add_argument("--model_path", type=str, default="results/fashion_mnist/diffusion/None/ddim_medium_mse/42/best_model",
                        help="Path to teacher model (only for distillation)")

    parser.add_argument("--eta", type=float, default=0.0)

    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    run_name = f"fid_vs_samples_eta_{args.eta}_seed_{args.seed}"
    results_folder = os.path.join("results", run_name)
    os.makedirs(results_folder, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    preprocess = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
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

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=model_config.scheduler.train_timesteps)

    assert args.model_path is not None
    model_pipeline = DDIMPipeline.from_pretrained(args.model_path)
    model_pipeline.set_progress_bar_config(disable=True)
    model = model_pipeline.unet.to(device)

    all_fid_list = []
    all_time_list = []
    num_samples = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    for n_samples in num_samples:
        print(f"Running for n_samples: {n_samples}")
        save_path = os.path.join(results_folder, f"gen_step_{n_samples}")
        os.makedirs(save_path, exist_ok=True)
        evaluator = GenerativeModelEvaluator(
            device=device, save_images=0, save_path=save_path)
        
        wrap_in_pipeline(model, noise_scheduler,
                         DDIMPipeline, n_samples, args.eta, def_output_type="torch")
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        fid_list = []
        time_list = []
        for _ in range(5):
            start.record()
            fid = evaluator.evaluate(model, test_dataloader, fid_images=n_samples, gensteps=20, compute_auc=False)["fid"]
            end.record()
            torch.cuda.synchronize()

            time_list.append(start.elapsed_time(end))
            fid_list.append(fid)

        all_fid_list.append(fid_list)
        all_time_list.append(time_list)

        print(f"Time taken: {np.mean(time_list) / 1000} +- {np.std(time_list) / 1000} s")
        print(f"FID: {np.mean(fid_list)} +- {np.std(fid_list)}")

    # Save results as json
    results = {
        "config": {
            "args": vars(args),
            "model_config": model_config,
        },
        "results": {
            "fid_list": all_fid_list,
            "time_list": all_time_list,
            "num_samples": num_samples
        }
    }

    with open(os.path.join(results_folder, "fid_vs_samples_results.json"), "w") as f:
        json.dump(results, f, indent=4)

    # Save graph
    plot_line_std_graph(
        num_samples,
        np.array([np.mean(t) for t in all_fid_list]),
        np.array([np.std(t) for t in all_fid_list]),
        "Num Samples",
        "FID",
        "FID vs Number of Samples",
        os.path.join(results_folder, "fid_vs_samples.png")
    )

    plot_line_std_graph(
        num_samples,
        np.array([np.mean(t) / 1000 for t in all_time_list]),
        np.array([np.std(t) / 1000 for t in all_time_list]),
        "Num Samples",
        "Time (s)",
        "Time vs Number of Samples",
        os.path.join(results_folder, "time_vs_samples.png")
    )


if __name__ == "__main__":
    args = __parse_args()
    main(args)
