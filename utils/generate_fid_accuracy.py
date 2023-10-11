import os
import pandas as pd
import numpy as np
import argparse

from typing import List

import sys
from pathlib import Path
# This script should be run from the root of the project
sys.path.append(str(Path(__file__).parent.parent))

from src.common.visual import plot_line_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_fuji/smasipca/generative_replay_gensteps/split_fmnist/")
    return parser.parse_args()


def plot_metrics(experiment_paths: List[str], save_path: str):
    metrics = {}
    
    for experiment_path in experiment_paths:  
        values = {}

        for seed_folder in os.listdir(experiment_path):
            seed_path = os.path.join(experiment_path, seed_folder)

            if not os.path.isdir(seed_path):
                continue

            seed_path = os.path.join(seed_path, 'logs')

            for file_name in os.listdir(seed_path):
                if file_name.endswith('.csv') and file_name.startswith('eval'):
                    file_path = os.path.join(seed_path, file_name)
                    df = pd.read_csv(file_path)
                    last_experience = df['training_exp'].max()
                    df = df[df['training_exp'] == last_experience]
                    df = df[df['eval_exp'] == last_experience]
                                    
                    for metric_name in ["Top1_Acc_Stream/eval_phase/test_stream/Task000", "StreamForgetting/eval_phase/test_stream", "stream_fid/eval_phase/test_stream/Task000"]:
                        # Get metric row
                        metric_row = df[df['metric_name'] == metric_name]
                        # Get metric value (last column)
                        metric_value = float(metric_row.iloc[0, -1])
                        if metric_name not in values:
                            values[metric_name] = []
                        values[metric_name].append(metric_value)

        for metric_name in values:
            if metric_name not in metrics:
                metrics[metric_name] = {}
                metrics[metric_name]['means'] = []
                metrics[metric_name]['stds'] = []
            metrics[metric_name]['means'].append(np.mean(values[metric_name]))
            metrics[metric_name]['stds'].append(np.std(values[metric_name]))
    
    # Plot fid vs accuracy
    output_path = os.path.join(save_path, f"fid_vs_accuracy.pgf")
    y_lim = [0, 1]
    x = metrics["stream_fid/eval_phase/test_stream/Task000"]["means"]
    y = metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"]["means"]
    x_index = np.argsort(x)
    x = np.array(x)[x_index]
    y = np.array(y)[x_index]
    plot_line_graph(x, y, "FID", "Accuracy", "FID vs Accuracy", output_path, log_x=False, y_lim=y_lim, size=(3, 2))

    # Plot gens vs accuracy
    output_path = os.path.join(save_path, f"gens_vs_accuracy.pgf")
    x = [20, 10, 5, 2]
    y = metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"]["means"]
    y = np.array(y)[x_index]
    plot_line_graph(x, y, "DDIM steps", "Accuracy", "DDIM steps vs Accuracy", output_path, x_ticks=x, log_x=False, y_lim=y_lim, size=(3, 2))

    # Plot gens vs forgetting
    output_path = os.path.join(save_path, f"gens_vs_forgetting.pgf")
    y = metrics["StreamForgetting/eval_phase/test_stream"]["means"]
    y = np.array(y)[x_index]
    plot_line_graph(x, y, "DDIM steps", "Forgetting", "DDIM steps vs Forgetting", output_path, x_ticks=x, log_x=False, y_lim=y_lim, size=(3, 2))
    
    # Plot fid vs forgetting
    output_path = os.path.join(save_path, f"fid_vs_forgetting.pgf")
    x = metrics["stream_fid/eval_phase/test_stream/Task000"]["means"]
    y = metrics["StreamForgetting/eval_phase/test_stream"]["means"]
    x = np.array(x)[x_index]
    y = np.array(y)[x_index]
    plot_line_graph(x, y, "FID", "Forgetting", "FID vs Forgetting", output_path, log_x=False, y_lim=y_lim, size=(3, 2))


if __name__ == '__main__':
    import matplotlib 
    matplotlib.use("pgf")
    matplotlib.rcParams.update({
        "pgf.texsystem": "pdflatex",
        'font.family': 'serif',
        'font.size': 8,
        'text.usetex': True,
        'pgf.rcfonts': False,
        'figure.autolayout': True,
    })
    args = __parse_args()
    experiment_names = []

    for experiment_name in os.listdir(args.experiments_path):
        if "lambd_4.0" in experiment_name:
            continue

        experiment_path = os.path.join(args.experiments_path, experiment_name)
        if os.path.isdir(experiment_path):
            experiment_names.append(experiment_path)

    plot_metrics(experiment_names, save_path=args.experiments_path)
        
        