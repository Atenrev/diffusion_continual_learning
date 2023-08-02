import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import argparse
import regex as re

from src.common.visual import plot_line_std_graph


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_fuji/smasipca/iid_results/fashion_mnist/diffusion/")
    return parser.parse_args()


def plot_metrics(experiment_path: str):
    sns.set(style='whitegrid')
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
        x = df['epoch'].unique()
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

        best_seed_metrics = {}
        best_seed_auc = float('inf')
        
        for file_name in os.listdir(seed_path):
            if file_name.endswith('.csv') and file_name.startswith('test'):
                file_path = os.path.join(seed_path, file_name)
                df = pd.read_csv(file_path)
                
                for epoch in df['epoch'].unique():
                    epoch_data = df[df['epoch'] == epoch]
                    auc_value = epoch_data[epoch_data['metric'] == 'auc']['value'].values[0]
                    
                    if auc_value < best_auc:
                        best_seed = seed_folder
                        best_epoch = epoch
                        best_auc = auc_value
                        best_metrics = epoch_data.set_index('metric')['value'].to_dict()

                    if auc_value < best_seed_auc:
                        best_seed_auc = auc_value
                        best_seed_metrics = epoch_data.set_index('metric')['value'].to_dict()

        best_seeds_metrics[seed_folder] = best_seed_metrics
    
    print(f"Best Seed: {best_seed}, Best Epoch: {best_epoch}")
    print("Best Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")

    # Calculate mean and std for each metric
    metrics_mean_std = {}
    for metric in best_seeds_metrics[list(best_seeds_metrics.keys())[0]]:
        metric_values = [metrics[metric] for metrics in best_seeds_metrics.values()]
        metrics_mean_std[metric] = {
            'mean': pd.Series(metric_values).mean(),
            'std': pd.Series(metric_values).std()
        }
    
    # Print mean and std for each metric
    print("Metrics mean and std:")
    for metric, values in metrics_mean_std.items():
        print(f"{metric} - {values['mean']}±{values['std']}")

    # Save best seed and epoch to json
    with open(os.path.join(experiment_path, 'summary.json'), 'w') as f:
        json.dump({
            'best_seed': int(best_seed),
            'best_epoch': int(best_epoch),
            'best_metrics': best_metrics,
            'metrics_mean_std': metrics_mean_std
        }, f, indent=4)

    return best_seed, best_epoch, best_metrics, metrics_mean_std


def create_table(experiment_path: str, metrics: dict):
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
            std = experiment_metrics[metric_name]['std']
            formatted_mean = f"{mean:.3f}"
            if mean == best_values[metric_name]:
                formatted_mean = f"\\textbf{{{mean:.3f}}}"
            row.append(f"{formatted_mean}±{std:.3f}")
        table.append(row)
    table = np.array(table)

    # Create a dataframe with the table
    metric_names = [re.sub(r'(\d+)', r'{DDIM(\1)}', metric_name) for metric_name in metrics_names]
    metric_names = [ "$" + metric_name.upper() + "$" for metric_name in metric_names]
    df = pd.DataFrame(table, columns=['Experiment Name'] + metric_names)
    df = df.set_index('Experiment Name').rename_axis(None)

    # Save the dataframe to csv
    df.to_csv(os.path.join(experiment_path, 'summary.csv'))

    # Save the dataframe to latex
    with open(os.path.join(experiment_path, 'summary.tex'), 'w') as f:
        f.write(df.to_latex())


if __name__ == '__main__':
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
                print(f"Plotting metrics for {experiment_path}...")
                plot_metrics(experiment_path)
                print(f"Creating summary for {experiment_path}...")
                _, _, _, metrics_mean_std = create_summary(experiment_path)
                experiment_name = root_experiment + '_' + experiment
                results_for_table[experiment_name] = metrics_mean_std
            except Exception as e:
                continue

    print(f"Creating table with final results...")
    create_table(args.experiments_path, results_for_table)