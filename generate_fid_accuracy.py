import os
import pandas as pd
import numpy as np
import argparse

from typing import List

from src.common.visual import plot_line_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_fuji/smasipca/generative_replay_single/split_fmnist/")
    return parser.parse_args()


def plot_metrics(experiment_paths: List[str]):
    metrics = {}
    
    for experiment_path in experiment_paths:        
        for file_name in os.listdir(experiment_path):
            if file_name.endswith('.csv') and file_name.startswith('eval'):
                file_path = os.path.join(experiment_path, file_name)
                df = pd.read_csv(file_path)
                last_experience = df['training_exp'].max()
                df = df[df['training_exp'] == last_experience]
                df = df[df['eval_exp'] == last_experience]
                                
                for metric_name in ["Top1_Acc_Stream/eval_phase/test_stream/Task000", "StreamForgetting/eval_phase/test_stream", "stream_fid/eval_phase/test_stream/Task000"]:
                    # Get metric row
                    metric_row = df[df['metric_name'] == metric_name]
                    # Get metric value (last column)
                    metric_value = metric_row.iloc[0, -1]
                    if metric_name not in metrics:
                        metrics[metric_name] = []
                    metrics[metric_name].append(metric_value)
    
    # Plot fid vs accuracy
    output_path = f"fid_vs_accuracy.png"
    x = metrics["stream_fid/eval_phase/test_stream/Task000"]
    y = metrics["Top1_Acc_Stream/eval_phase/test_stream/Task000"]
    plot_line_graph(x, y, "FID", "Accuracy", "FID vs Accuracy", output_path, log_x=True)
    # Plot fid vs forgetting
    output_path = f"fid_vs_forgetting.png"
    x = metrics["stream_fid/eval_phase/test_stream/Task000"]
    y = metrics["StreamForgetting/eval_phase/test_stream"]
    plot_line_graph(x, y, "FID", "Forgetting", "FID vs Forgetting", output_path, log_x=True)


if __name__ == '__main__':
    args = __parse_args()

    results_for_table = {}
    experiment_names = [
        "gr_diffusion_full_generation_distillation_steps_2_lambd_1.0_cnn/42/logs",
        "gr_diffusion_full_generation_distillation_steps_5_lambd_1.0_cnn/42/logs",
        "gr_diffusion_full_generation_distillation_steps_10_lambd_1.0_cnn/42/logs",
        "gr_diffusion_full_generation_distillation_steps_20_lambd_1.0_cnn/42/logs",
    ]
    experiment_names = [os.path.join(args.experiments_path, experiment_name) for experiment_name in experiment_names]
    plot_metrics(experiment_names)
        
        