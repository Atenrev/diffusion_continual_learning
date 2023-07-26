import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.common.visual import plot_line_std_graph

def plot_metrics(experiment_path):
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
        plot_line_std_graph(x, means, stds, 'Epoch', metric, metric, f"{metric}_std.png")


def find_best_seed_and_epoch(experiment_path):
    best_seed = None
    best_epoch = None
    best_auc = float('inf')
    best_metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue
        
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
    
    print(f"Best Seed: {best_seed}, Best Epoch: {best_epoch}")
    print("Best Metrics:")
    for metric, value in best_metrics.items():
        print(f"{metric}: {value}")


def compute_metrics_for_best_epoch(experiment_path):
    best_seeds_metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)
        if not os.path.isdir(seed_path):
            continue
        
        best_auc = float('inf')
        best_metrics = {}
        
        for file_name in os.listdir(seed_path):
            if file_name.endswith('.csv') and file_name.startswith('test'):
                file_path = os.path.join(seed_path, file_name)
                df = pd.read_csv(file_path)
                
                for epoch in df['epoch'].unique():
                    epoch_data = df[df['epoch'] == epoch]
                    auc_value = epoch_data[epoch_data['metric'] == 'auc']['value'].values[0]
                    
                    if auc_value < best_auc:
                        best_auc = auc_value
                        best_metrics = epoch_data.set_index('metric')['value'].to_dict()
        
        best_seeds_metrics[seed_folder] = best_metrics
    
    # Calculate mean and std for each metric
    metrics_mean_std = {}
    for metric in best_seeds_metrics[list(best_seeds_metrics.keys())[0]]:
        metric_values = [metrics[metric] for metrics in best_seeds_metrics.values()]
        metrics_mean_std[metric] = {
            'mean': pd.Series(metric_values).mean(),
            'std': pd.Series(metric_values).std()
        }
    
    # Print mean and std for each metric
    for metric, values in metrics_mean_std.items():
        print(f"{metric} - Mean: {values['mean']}, Std: {values['std']}")


if __name__ == '__main__':
    experiment_path = "results/fashion_mnist/diffusion/None/ddim_medium_mse"
    print("Plotting metrics...")
    plot_metrics(experiment_path)
    print("Finding best seed and epoch...")
    find_best_seed_and_epoch(experiment_path)
    print("Computing mean and std for the best models...")
    compute_metrics_for_best_epoch(experiment_path)



#"results/fashion_mnist/diffusion/None/ddim_medium_mse"
