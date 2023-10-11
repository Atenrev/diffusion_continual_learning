import os
import json
import pandas as pd
import numpy as np
import argparse
import regex as re
import torch
import matplotlib 

from diffusers import DDIMScheduler

import sys
from pathlib import Path
# This script should be run from the root of the project
sys.path.append(str(Path(__file__).parent.parent))

from src.common.visual import plot_line_std_graph
from src.pipelines.pipeline_ddim import DDIMPipeline
from src.common.diffusion_utils import wrap_in_pipeline, generate_diffusion_samples


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        help="Path to the folder containing the experiment folders",
                        default="results_fuji/smasipca/iid_results/comparison_pvf/diffusion/")
    parser.add_argument("--generate_images", 
                        help="Generate images for the best seed of each experiment",
                        action="store_true")
    return parser.parse_args()


def plot_metrics(experiment_path: str):
    metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue
        
        for file_name in os.listdir(seed_path):
            if file_name.endswith('.csv') and file_name.startswith('test'):
                file_path = os.path.join(seed_path, file_name)
                df = pd.read_csv(file_path)
                
                for metric, group in df.groupby('metric'):
                    if metric not in metrics:
                        metrics[metric] = []
                    metrics[metric].append(group['value'].values)
    
    # Plot each metric and calculate average and std
    for metric, values_list in metrics.items():
        x = np.arange(len(values_list[0]))
        means = np.array(values_list).mean(axis=0)
        stds = np.array(values_list).std(axis=0)
        output_path = os.path.join(experiment_path, f"{metric}.png")
        plot_line_std_graph(x, means, stds, 'Epoch', metric, metric, output_path)


def create_summary(experiment_path: str):
    best_seed = None
    best_epoch = None
    best_auc = float('inf')
    best_metrics = {}
    best_seeds_metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue

        # best_seed_metrics = {}
        # best_seed_auc = float('inf')
        last_seed_metrics = {}
        
        for file_name in os.listdir(seed_path):
            if file_name.endswith('.csv') and file_name.startswith('test'):
                file_path = os.path.join(seed_path, file_name)
                df = pd.read_csv(file_path)
                
                # for epoch in df['epoch'].unique():
                    # epoch_data = df[df['epoch'] == epoch]
                    # auc_value = epoch_data[epoch_data['metric'] == 'auc']['value'].values[0]
                
                # Get last epoch
                epoch = df['epoch'].max()
                epoch_data = df[df['epoch'] == epoch]
                auc_value = epoch_data[epoch_data['metric'] == 'auc']['value'].values[0]
                last_seed_metrics = epoch_data.set_index('metric')['value'].to_dict()

                if auc_value < best_auc:
                    best_seed = seed_folder
                    best_epoch = epoch
                    best_auc = auc_value
                    best_metrics = epoch_data.set_index('metric')['value'].to_dict()

                    # if auc_value < best_seed_auc:
                    #     best_seed_auc = auc_value
                    #     best_seed_metrics = epoch_data.set_index('metric')['value'].to_dict()

        # best_seeds_metrics[seed_folder] = best_seed_metrics
        best_seeds_metrics[seed_folder] = last_seed_metrics

    # Calculate mean and std for each metric
    metrics_mean_std = {}
    for metric in best_seeds_metrics[list(best_seeds_metrics.keys())[0]]:
        metric_values = [metrics[metric] for metrics in best_seeds_metrics.values()]
        metrics_mean_std[metric] = {
            'mean': pd.Series(metric_values).mean(),
            'std': pd.Series(metric_values).std(),
            'sem': pd.Series(metric_values).std() / np.sqrt(len(metric_values))
        }

    # Save best seed and epoch to json
    with open(os.path.join(experiment_path, 'summary.json'), 'w') as f:
        json.dump({
            'best_seed': int(best_seed),
            'best_epoch': int(best_epoch),
            'best_metrics': best_metrics,
            'metrics_mean_std': metrics_mean_std
        }, f, indent=4)

    return best_seed, best_epoch, best_metrics, metrics_mean_std


def create_table(experiments_path: str, metrics: dict):
    """
    Create a table with the results of all experiments 
    and save it to a csv file and a latex file.

    The numbers are rounded to 3 decimal places.

    Table format:
    Experiment Name | Metric 1 | Metric 2 | ... | Metric N
    ------------------------------------------------------
    Experiment 1    | mean±std | mean±std | ... | mean±std
    Experiment 2    | mean±std | mean±std | ... | mean±std
    ...
    
    Args:
        experiment_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {mean: float, std: float}, ...}, ...}
    """
    # Get the metrics names
    metrics_names = set()
    for metric in metrics.values():
        metrics_names.update(metric.keys())
    metrics_names = sorted(metrics_names, key=lambda x: (int(re.findall(r'\d+', x)[0] if re.findall(r'\d+', x) else "0"), x))

    # Identify the best values for each metric
    best_values = {}
    for metric_name in metrics_names:
        best_value = float('inf')  # Initialize with a very large value
        for experiment_metrics in metrics.values():
            mean = experiment_metrics[metric_name]['mean']
            if mean < best_value:
                best_value = mean
        best_values[metric_name] = best_value

    # Create the table
    table = []
    for experiment_name, experiment_metrics in metrics.items():
        experiment_name = experiment_name.replace('_', ' ')
        experiment_name = ' '.join([word.capitalize() for word in experiment_name.split()])
        row = [experiment_name]
        for metric_name in metrics_names:
            mean = experiment_metrics[metric_name]['mean']
            sem = experiment_metrics[metric_name]['sem']
            formatted_mean = f"{mean:.3f}±{sem:.3f}"
            if mean == best_values[metric_name]:
                formatted_mean = f"\\textbf{{{formatted_mean}}}"
            row.append(f"{formatted_mean}")
        table.append(row)
    table = np.array(table)

    # Create a dataframe with the table
    metrics_names = [re.sub(r'(\d+)', r'{\1}', metric_name) for metric_name in metrics_names]
    metrics_names = [metric_name.replace('AUC', 'FID_{\\text{auc}}') for metric_name in metrics_names]
    metrics_names = [ "$" + metric_name.upper() + "$" for metric_name in metrics_names]
    df = pd.DataFrame(table, columns=['Experiment Name'] + metrics_names)
    df = df.set_index('Experiment Name').rename_axis(None)

    # Save the dataframe to csv
    
    df.to_csv(os.path.join(experiments_path, 'summary.csv'))

    # Save the dataframe to latex
    final_table = df.to_latex()
    final_table = f"""
\\begin{{table}}[]
    \centering
    {final_table}
    \caption{{Caption}}
    \label{{tab:my_label}}
\end{{table}}
"""
    with open(os.path.join(experiments_path, 'summary.tex'), 'w') as f:
        f.write(final_table)


def create_comparison_plots(experiments_path: str):
    """
    Create a plot with the comparison of the metrics for each 
    experiment and save it to a png file.

    Args:
        experiments_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {mean: float, std: float}, ...}, ...}
    """
    all_metrics = {}

    for root_experiment in os.listdir(experiments_path):
        root_experiment_path = os.path.join(experiments_path, root_experiment)
        if not os.path.isdir(root_experiment_path):
            continue

        for experiment_name in os.listdir(root_experiment_path):
            experiment_path = os.path.join(root_experiment_path, experiment_name)
            if not os.path.isdir(experiment_path):
                continue

            metrics = {}
            
            for seed_folder in os.listdir(experiment_path):
                seed_path = os.path.join(experiment_path, seed_folder)
                if not os.path.isdir(seed_path):
                    continue
                
                for file_name in os.listdir(seed_path):
                    if file_name.endswith('.csv') and file_name.startswith('test'):
                        file_path = os.path.join(seed_path, file_name)
                        df = pd.read_csv(file_path)
                        
                        for metric, group in df.groupby('metric'):
                            if metric not in metrics:
                                metrics[metric] = []
                            metrics[metric].append(group['value'].values)
            
            # Plot each metric and calculate average and std
            for metric, values_list in metrics.items():
                exp_name = root_experiment
                means = np.array(values_list).mean(axis=0)
                stds = np.array(values_list).std(axis=0)

                x = np.arange(len(values_list[0]))
                if "base" in exp_name: 
                    x_ticks = (x+1) * 5 * 469 // 1000 
                else:
                    x_ticks = x
                output_path = os.path.join(experiment_path, f"{metric}.png")
                plot_line_std_graph(x, means, stds, 'Steps (in thousands)', metric, metric, output_path, x_ticks=x_ticks)
                
                if metric not in all_metrics:
                    all_metrics[metric] = {
                        "x": [],
                        "means": [],
                        "stds": [],
                        "experiment_names": []
                    }

                all_metrics[metric]["x"].append(np.array(x_ticks))
                all_metrics[metric]["means"].append(means)
                all_metrics[metric]["stds"].append(stds)

                if "base_model" in exp_name: 
                    exp_name = exp_name.replace('_', ' ')
                else:
                    exp_name = exp_name.replace('_', '-')

                    if "distillation" not in exp_name:
                        exp_name = exp_name + " distillation"
                    else:
                        exp_name = "student"

                teacher_ddim_steps = re.findall(r'\d+', experiment_name)
                exp_name = [exp_name + f" ({teacher_ddim_steps[0]} teacher DDIM steps)" if teacher_ddim_steps else exp_name][0]
                all_metrics[metric]["experiment_names"].append(exp_name)

    # Create the plot
    for metric_name in all_metrics:
        all_means = all_metrics[metric_name]["means"]
        all_sems = all_metrics[metric_name]["stds"] / np.sqrt(len(all_metrics[metric_name]["stds"]))
        x = all_metrics[metric_name]["x"]
        if "auc" not in metric_name.lower():
            x_ticks = np.arange(max([max(a) for a in x]) + 1) + 1
            x_labels = None
        else:
            # Tick every 2
            x_ticks = np.arange(0, max([max(a) for a in x]) + 1, 2)
            x_labels = x_ticks + 1
        y_labels = all_metrics[metric_name]["experiment_names"]
        output_path = os.path.join(experiments_path, f"{metric_name}.pgf")
        metric_name = metric_name.replace('auc', '$FID_{auc}$')
        plot_line_std_graph(x, all_means, all_sems, 'Steps (in thousands)', metric_name, f"Comparison - {metric_name}", output_path, x_ticks=x_ticks, x_labels=x_labels, y_labels=y_labels, size=(4.8, 2.2))


def generate_images(experiment_path: str, best_seed: int):
    # Model is in last_model in the seed inside the experiment path
    model_path = os.path.join(experiment_path, str(best_seed), 'last_model')
    # Load the best model
    model_pipeline = DDIMPipeline.from_pretrained(model_path)
    model_pipeline.set_progress_bar_config(disable=True)
    model = model_pipeline.unet
    # if cuda use it
    if torch.cuda.is_available():
        model = model.cuda()
    noise_scheduler = DDIMScheduler(num_train_timesteps=1000)

    for steps in [2, 5, 10, 20, 50, 100]:
        wrap_in_pipeline(model, noise_scheduler, DDIMPipeline,
                            steps, 0.0, def_output_type="torch_raw")
        output_dir = os.path.join("samples", "_".join(experiment_path.split('/')[-2:]) + f"_{best_seed}")
        os.makedirs(output_dir, exist_ok=True)
        generate_diffusion_samples(output_dir, 100, steps, model, steps, seed=1)


if __name__ == '__main__':
    # Document text width: 7.1413 inches
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

    results_for_table = {}
    # Iterate the experiments path and plot the metrics for each experiment
    # Each experiment might have several subexperiments
    # each one of them is a folder with the results of 5 different seeds
    for root_experiment in os.listdir(args.experiments_path):
        root_experiment_path = os.path.join(args.experiments_path, root_experiment)
        if not os.path.isdir(root_experiment_path):
            continue
        
        for experiment in os.listdir(root_experiment_path):
            experiment_path = os.path.join(root_experiment_path, experiment)
            if not os.path.isdir(experiment_path):
                continue
            
            try:
                print(f"Creating summary for {experiment_path}...")
                best_seed, _, _, metrics_mean_std = create_summary(experiment_path)
                if args.generate_images:
                    print(f"Generating images for {experiment_path}...")
                    generate_images(experiment_path, best_seed)
                experiment_name = root_experiment + '_' + experiment
                results_for_table[experiment_name] = metrics_mean_std
            except Exception as e:
                continue

    print(f"Creating table with final results...")
    create_table(args.experiments_path, results_for_table)
    print(f"Creating comparison plots...")
    create_comparison_plots(args.experiments_path)