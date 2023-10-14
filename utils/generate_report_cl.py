import os
import json
import pandas as pd
import numpy as np
import argparse
import regex as re
import torch
import matplotlib 

from typing import Union, List

import sys
from pathlib import Path
# This script should be run from the root of the project
sys.path.append(str(Path(__file__).parent.parent))

from src.common.visual import plot_line_std_graph, plot_bar


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_fuji/smasipca/generative_replay/split_fmnist/")
    parser.add_argument("--only_average", action='store_true', default=True)
    parser.add_argument("--filter_condition", type=Union[str, List[str]], 
                        help="Filter the experiments to plot. If empty string, all experiments are plotted.",
                        default=[
                            "gr_cnn_naive",
                            "gr_cnn_cumulative",
                            "gr_diffusion_naive",
                            "gr_diffusion_cumulative",
                            # "diffusion_no_distillation_steps_10,_teacher_DDIM_steps_2",
                            "diffusion_no_distillation",
                            "lwf_distillation_steps_10_lambd_0.25_cnn", 
                            "full_generation_distillation_steps_10_lambd_3.0_cnn", 
                            # "partial_generation_distillation_steps_10_lambd_5.0_cnn", 
                            "gaussian_distillation_steps_10_lambd_24.0_cnn", 
                            # "gaussian_symmetry_distillation_steps_10_lambd_12.0_cnn"
                            ])
    return parser.parse_args()


def format_experiment_name(experiment_name: str) -> str:
    if "gr_cnn_naive" in experiment_name:
        experiment_name = "Finetune (classifier)"
    elif "gr_cnn_cumulative" in experiment_name:
        experiment_name = "Joint (classifier)"
    elif "gr_diffusion_naive" in experiment_name:
        experiment_name = "Finetune (generator)"
    elif "gr_diffusion_cumulative" in experiment_name:
        experiment_name = "Joint (generator)"
    else:
        experiment_name = experiment_name.replace("gr_diffusion_", "").replace("_cnn", "").replace('_', ' ').replace(' lambd ', ', lambda=') # .replace("_steps_10", "", 1)
        experiment_name = ' '.join([word for word in experiment_name.split()])

    return experiment_name


def get_short_name(experiment_name: str) -> str:
    if "gr_cnn_naive" in experiment_name:
        experiment_name = "Naive"
    elif "gr_cnn_cumulative" in experiment_name:
        experiment_name = "Joint"
    elif "gr_diffusion_naive" in experiment_name:
        experiment_name = "Naive"
    elif "gr_diffusion_cumulative" in experiment_name:
        experiment_name = "Joint"
    elif "full_generation" in experiment_name:
        experiment_name = "GD"
    elif "partial_generation" in experiment_name:
        experiment_name = "PGD"
    elif "gaussian_distillation" in experiment_name:
        experiment_name = "GGD"
    elif "gaussian_symmetry" in experiment_name:
        experiment_name = "GGSD"
    elif "lwf" in experiment_name:
        experiment_name = "LwF"
    elif "no_distillation" in experiment_name:
        steps = re.findall(r'\d+', experiment_name)[1]
        experiment_name = f"GR({steps})"

    return experiment_name


def plot_metrics(experiment_path: str):
    metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)

        if not os.path.isdir(seed_path):
            continue

        seed_path = os.path.join(seed_path, 'logs')
        
        for file_name in os.listdir(seed_path):
            if not file_name.endswith('.csv') or not file_name.startswith('eval'):
                continue

            file_path = os.path.join(seed_path, file_name)
            df = pd.read_csv(file_path)
            last_experience = df['training_exp'].max()


            for metric_name in ['avg FID', 'avg KLD', 'avg ACC', 'avg FORGETTING', 'stream_hist_pred', 'stream_hist_true']:
                if metric_name not in metrics:
                    metrics[metric_name] = []
                metrics[metric_name].append([])

            for experience in df['training_exp'].unique():
                experience_data = df[df['training_exp'] == experience]
                experience_data = experience_data[experience_data['eval_exp'] == last_experience]
                if not experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"].empty:
                    fid_row = experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"]
                    fid_value = fid_row.iloc[-1, -1]
                    kld_row = experience_data[experience_data['metric_name'] == "stream_kld/eval_phase/test_stream/Task000"]
                    kld_value = kld_row.iloc[-1, -1]
                    stream_hist_pred_row = experience_data[experience_data['metric_name'] == "stream_hist_pred/eval_phase/test_stream/Task000"]
                    stream_hist_pred_value = stream_hist_pred_row.iloc[-1, -1]
                    stream_hist_true_row = experience_data[experience_data['metric_name'] == "stream_hist_true/eval_phase/test_stream/Task000"]
                    stream_hist_true_value = stream_hist_true_row.iloc[-1, -1]
                    metrics['avg FID'][-1].append(float(fid_value))
                    metrics['avg KLD'][-1].append(float(kld_value))
                    metrics['stream_hist_true'][-1].append([float(x) for x in stream_hist_true_value[1:-1].strip().split()])
                    metrics['stream_hist_pred'][-1].append([float(x) for x in stream_hist_pred_value[1:-1].strip().split()])

                if not experience_data[experience_data['metric_name'] == "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000"].empty:
                    top1_acc_row = experience_data[experience_data['metric_name'] == "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000"]
                    top1_acc_value = top1_acc_row.iloc[-1, -1]
                    stream_forgetting_row = experience_data[experience_data['metric_name'] == "StreamForgetting/eval_phase/test_stream"]
                    stream_forgetting_value = stream_forgetting_row.iloc[-1, -1]
                    metrics['avg ACC'][-1].append(float(top1_acc_value))
                    metrics['avg FORGETTING'][-1].append(float(stream_forgetting_value))

    # Join both stream_hist_pred and stream_hist_true
    if 'stream_hist_true' in metrics:
        metrics['stream_hist'] = []
        for i in range(len(metrics['stream_hist_pred'])):
            metrics['stream_hist'].append([])
            for j in range(len(metrics['stream_hist_pred'][i])):
                metrics['stream_hist'][i].append([metrics['stream_hist_pred'][i][j], metrics['stream_hist_true'][i][j]])
        del metrics['stream_hist_pred']
        del metrics['stream_hist_true']
    
    # Plot each metric and calculate average and std
    # for each experience
    for metric, values_list in metrics.items():
        experiment_name = format_experiment_name(experiment_path.split('/')[-1])
        title = metric + " - " + experiment_name
        values_list = np.array(values_list).squeeze()

        if values_list.size == 0:
            continue

        if "hist" not in metric:
            x = np.array(range(last_experience+1))
            if values_list.ndim == 2:
                means = values_list.mean(axis=0)
                stds = values_list.std(axis=0)
            else:
                means = values_list
                stds = np.zeros_like(means)
            output_path = os.path.join(experiment_path, f"{metric}.pgf")
            y_lim = None
            if metric == 'avg FID':
                y_lim = (0, 100)
            elif metric == 'avg KLD':
                max_idx = np.argmax(means)
                max_kld = means[max_idx] + stds[max_idx]
                y_lim = (0, max_kld + 0.2)
            elif metric == 'avg ACC' or metric == 'avg FORGETTING':
                y_lim = (0, 1)
            plot_line_std_graph(x, means, stds, 'Task', metric, title, output_path, x_ticks=x, y_lim=y_lim, size=(3, 2))
        else:
            x = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

            if values_list.ndim == 4:
                for seed in range(values_list.shape[0]):
                    for i in range(values_list.shape[1]):
                        output_path = os.path.join(experiment_path, f"{metric}_exp_{i}_seed_{seed}.pgf")
                        plot_bar(x, values_list[seed, i, :], 'Class', 'Frequency', title, output_path, y_labels=('True', 'Predicted'), size=(3, 2))
            else:
                means = values_list

                for i in range(means.shape[0]):
                    output_path = os.path.join(experiment_path, f"{metric}_exp_{i}.pgf")
                    plot_bar(x, means[i, :], 'Class', 'Frequency', title, output_path, y_labels=('True', 'Predicted'), size=(3, 2))


def create_comparison_histogram(experiments_path: str, filter_condition: Union[str, List[str]] = ""):
    """
    Creates a histogram with the comparison of stream_hist_pred and stream_hist_true
    for each experiment and save it to a png file.
    """    
    true_added = [False] * 5
    experiment_names = []
    metrics = [[], [], [], [], []]
    root_experiments = os.listdir(experiments_path)
    root_experiments = sorted(root_experiments, key=lambda x: 0 if "cumulative" in x or "naive" in x else 1 if "no_distillation" in x else 2 if "lwf" in x else 3 if "partial_generation" in x else 4 if "full_generation" in x else 5 if "gaussian_distillation" in x else 6 if "gaussian_symmetry" in x else 7)

    for root_experiment in root_experiments:
        filter_condition = [condition for condition in filter_conditions if condition in root_experiment]

        if args.filter_condition is not None and not filter_condition:
            continue
        
        experiment_path = os.path.join(args.experiments_path, root_experiment)
        if not os.path.isdir(experiment_path):
            continue

        for seed_folder in os.listdir(experiment_path):
            if seed_folder != "42": 
                continue 
            
            seed_path = os.path.join(experiment_path, seed_folder)

            if not os.path.isdir(seed_path):
                continue

            seed_path = os.path.join(seed_path, 'logs')
            added_experiment_name = False
            
            for file_name in os.listdir(seed_path):
                if not file_name.endswith('.csv') or not file_name.startswith('eval'):
                    continue

                file_path = os.path.join(seed_path, file_name)
                df = pd.read_csv(file_path)
                last_experience = df['training_exp'].max()

                for i, experience in enumerate(df['training_exp'].unique()):
                    experience_data = df[df['training_exp'] == experience]
                    experience_data = experience_data[experience_data['eval_exp'] == last_experience]
                    
                    if experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"].empty:
                        continue
                    
                    if not added_experiment_name:
                        experiment_names.append(root_experiment)
                        added_experiment_name = True

                    stream_hist_pred_row = experience_data[experience_data['metric_name'] == "stream_hist_pred/eval_phase/test_stream/Task000"]
                    stream_hist_pred_value = stream_hist_pred_row.iloc[-1, -1]
                    stream_hist_true_row = experience_data[experience_data['metric_name'] == "stream_hist_true/eval_phase/test_stream/Task000"]
                    stream_hist_true_value = stream_hist_true_row.iloc[-1, -1]
                    if not true_added[i]:
                        true_added[i] = True
                        metrics[i].append([float(x) for x in stream_hist_pred_value[1:-1].strip().split()])
                    metrics[i].append([float(x) for x in stream_hist_true_value[1:-1].strip().split()])

    x = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    title = "Comparison - Histogram"
    experiment_names = [format_experiment_name(experiment_name) for experiment_name in experiment_names]
    y_labels = ["True"] + experiment_names
    
    for i, metric in enumerate(metrics):
        output_path = os.path.join(experiments_path, f"comparison_hist_exp_{i}.pgf")
        plot_bar(x, metric, 'Class', 'Frequency', title, output_path, y_labels=y_labels, size=(6, 2.5))


def create_summary(experiment_path: str):
    best_seed_fgt = None
    best_seed_fidauc = None
    best_fgt = float('inf')
    best_fidauc = float('inf')
    best_metrics_fgt = {}
    best_metrics_fidauc = {}
    seeds_metrics = {}
    
    for seed_folder in os.listdir(experiment_path):
        seed_path = os.path.join(experiment_path, seed_folder)

        if not os.path.isdir(seed_path):
            continue

        seed_path = os.path.join(seed_path, 'logs')
        last_seed_metrics = {}
        
        for file_name in os.listdir(seed_path):
            if not file_name.endswith('.csv') or not file_name.startswith('eval'):
                continue

            file_path = os.path.join(seed_path, file_name)
            df = pd.read_csv(file_path)
            last_experience = df['training_exp'].max()

            fids = []
            klds = []
            for experience in df['training_exp'].unique():
                experience_data = df[df['training_exp'] == experience]
                experience_data = experience_data[experience_data['eval_exp'] == last_experience]
                top1_acc_value = None

                if not experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"].empty:
                    fid_row = experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"]
                    fid_value = float(fid_row.iloc[-1, -1])
                    fids.append(fid_value)
                    kld_row = experience_data[experience_data['metric_name'] == "stream_kld/eval_phase/test_stream/Task000"]
                    kld_value = float(kld_row.iloc[-1, -1])
                    klds.append(kld_value)
                    last_seed_metrics[f'stream_fid_{experience}'] = fid_value
                    last_seed_metrics[f'stream_kld_{experience}'] = kld_value

                if not experience_data[experience_data['metric_name'] == "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000"].empty:
                    top1_acc_row = experience_data[experience_data['metric_name'] == "Accuracy_On_Trained_Experiences/eval_phase/test_stream/Task000"]
                    top1_acc_value = float(top1_acc_row.iloc[-1, -1])
                    stream_forgetting_row = experience_data[experience_data['metric_name'] == "StreamForgetting/eval_phase/test_stream"]
                    stream_forgetting_value = float(stream_forgetting_row.iloc[-1, -1])
                    last_seed_metrics[f'stream_acc_{experience}'] = top1_acc_value
                    last_seed_metrics[f'stream_forgetting_{experience}'] = stream_forgetting_value

                if experience == last_experience:
                    if fids:
                        last_seed_metrics['avg FID'] = fid_value
                        last_seed_metrics['avg KLD'] = kld_value
                    if top1_acc_value:
                        last_seed_metrics['avg ACC'] = top1_acc_value
                        last_seed_metrics['avg FORGETTING'] = stream_forgetting_value

            stream_fidauc = torch.trapz(torch.tensor(fids)).item()
            # last_seed_metrics['avg FID AUC'] = stream_fidauc
            stream_kldauc = torch.trapz(torch.tensor(klds)).item()
            # last_seed_metrics['avg KLD AUC'] = stream_kldauc

            if 'stream_forgetting_value' in last_seed_metrics and last_seed_metrics['avg FORGETTING'] < best_fgt:
                best_seed_fgt = seed_folder
                best_fgt = last_seed_metrics['avg FORGETTING'] 
                best_metrics_fgt = last_seed_metrics

            if stream_fidauc < best_fidauc:
                best_seed_fidauc = seed_folder
                best_fidauc = stream_fidauc
                best_metrics_fidauc = last_seed_metrics

        seeds_metrics[seed_folder] = last_seed_metrics

    # Calculate mean and std for each metric
    metrics_mean_std = {}
    for metric in seeds_metrics[list(seeds_metrics.keys())[0]]:
        metric_values = [metrics[metric] for metrics in seeds_metrics.values()]
        metrics_mean_std[metric] = {
            'mean': np.mean(metric_values),
            'std': np.std(metric_values),
            'sem': np.std(metric_values) / np.sqrt(len(metric_values))
        }

    # Save best seed and method to json
    with open(os.path.join(experiment_path, 'summary.json'), 'w') as f:
        json.dump({
            'best_seed_fgt': int(best_seed_fgt) if best_seed_fgt else None,
            'best_metrics_fgt': best_metrics_fgt if best_metrics_fgt else None,
            'best_seed_fidauc': int(best_seed_fidauc) if best_seed_fidauc else None,
            'best_metrics_fidauc': best_metrics_fidauc if best_metrics_fidauc else None,
            'metrics_mean_std': metrics_mean_std if metrics_mean_std else None
        }, f, indent=4)

    return best_seed_fgt, best_metrics_fgt, best_seed_fidauc, best_metrics_fidauc, metrics_mean_std


def create_table(experiment_path: str, metrics: dict, only_average: bool = False, filter_condition: Union[str, List[str]] = ""):
    """
    Create a table with the results of all experiments 
    and save it to a csv file and a latex file.

    The numbers are rounded to 3 decimal places.

    Table format:
    Experiment Name | Metric 1 | Metric 2 | ... | Metric N
    ------------------------------------------------------
    Experiment 1    | mean±sem | mean±sem | ... | mean±sem
    Experiment 2    | mean±sem | mean±sem | ... | mean±sem
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
    metrics_names = sorted(metrics_names, reverse=True)
    # First FID, then KLD, then ACC, then FORGETTING
    metrics_names = sorted(metrics_names, key=lambda x: 0 if 'fid' in x.lower() else 1 if 'kld' in x.lower() else 2 if 'acc' in x.lower() else 3 if 'forgetting' in x.lower() else 4)
    metrics_names = [metric_name for metric_name in metrics_names if 'avg' in metric_name.lower() or not only_average]

    # Identify the best values for each metric
    best_values = {}
    best_sems = {}
    for metric_name in metrics_names:
        best_value = float('inf')
        best_sem = float('inf')
        for experiment_name, experiment_metrics in metrics.items():
            if "cumulative" in experiment_name or "naive" in experiment_name:
                continue
            if metric_name not in experiment_metrics:
                continue
            mean = experiment_metrics[metric_name]['mean']
            mean = -mean if 'acc' in metric_name.lower() else mean
            if mean < best_value:
                best_value = mean
                best_sem = experiment_metrics[metric_name]['sem']

        best_values[metric_name] = -best_value if 'acc' in metric_name.lower() else best_value
        best_sems[metric_name] = best_sem

    # Given the best values, mark as best values the ones that 
    # overlap with the best values considering the sem
    multiple_best_values = {}
    for metric_name in metrics_names:
        multiple_best_values[metric_name] = []
        for experiment_name, experiment_metrics in metrics.items():
            if metric_name not in experiment_metrics:
                continue
            mean = experiment_metrics[metric_name]['mean']
            sem = experiment_metrics[metric_name]['sem']
            
            if (mean + sem >= best_values[metric_name] - best_sems[metric_name] and mean < best_values[metric_name]
                or mean - sem <= best_values[metric_name] + best_sems[metric_name] and mean > best_values[metric_name]
                or mean == best_values[metric_name]):
                multiple_best_values[metric_name].append(experiment_name)

    # Create the table
    table = []
    experiment_metrics = metrics.items()
    # experiment_metrics = sorted(experiment_metrics, key=lambda x: [int(i) for i in re.findall(r'\d+', x[0])] if re.findall(r'\d+', x[0]) else x[0])
    experiment_metrics = sorted(experiment_metrics)
    for experiment_name, experiment_metrics in experiment_metrics:
        formatted_experiment_name = format_experiment_name(experiment_name)
        # experiment_name = f"{filter_condition.replace('_', ' ')}, lambda=" + re.findall(r'\d+', experiment_name)[1]
        row = [formatted_experiment_name]
        for metric_name in metrics_names:
            if metric_name not in experiment_metrics:
                row.append('-')
                continue

            mean = experiment_metrics[metric_name]['mean']
            sem = experiment_metrics[metric_name]['sem']
            formatted_mean = f"{mean:.3f}±{sem:.3f}"
            # if mean == best_values[metric_name]:
            #     formatted_mean = f"\\textbf{{{formatted_mean}}}"
            if experiment_name in multiple_best_values[metric_name]:
                formatted_mean = f"\\textbf{{{formatted_mean}}}"
            row.append(f"{formatted_mean}")
        table.append(row)
    table = np.array(table)

    # Create a dataframe with the table
    # metrics_names = [re.sub(r'(\d+)', r'{DDIM(\1)}', metric_name) for metric_name in metrics_names]
    metrics_names = [ "$" + metric_name.upper() + "$" for metric_name in metrics_names]
    df = pd.DataFrame(table, columns=['Experiment Name'] + metrics_names)
    df = df.set_index('Experiment Name').rename_axis(None)

    # Save the dataframe to csv
    if isinstance(filter_condition, list):
        filter_condition = "comparison"
    df.to_csv(os.path.join(experiment_path, f'summary_{filter_condition}.csv'))

    # Save the dataframe to latex
    final_table = df.to_latex()
    final_table = f"""
\\begin{{table}}[]
    \centering
    \caption{{Caption}}
    {final_table}
    \label{{tab:my_label}}
\end{{table}}
"""
    with open(os.path.join(experiment_path, f'summary_{filter_condition}.tex'), 'w') as f:
        f.write(final_table)


def create_comparison_plots(experiments_path: str, metrics: dict, filter_condition: Union[str, List[str]] = ""):
    """
    Create a plot with the comparison of the stream metrics for each 
    experiment and save it to a png file.

    Args:
        experiments_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {mean: float, std: float}, ...}, ...}
    """
    experiment_names = list(metrics.keys())
    # experiment_names = sorted(experiment_names, key=lambda x: [int(i) for i in re.findall(r'\d+', x)] if re.findall(r'\d+', x) else x)
    experiment_names = sorted(experiment_names, key=lambda x: 0 if "cumulative" in x or "naive" in x else 1 if "no_distillation" in x else 2 if "lwf" in x else 3 if "partial_generation" in x else 4 if "full_generation" in x else 5 if "gaussian_distillation" in x else 6 if "gaussian_symmetry" in x else 7)
    metric_names = list(set(sum([list(metrics[en].keys()) for en in experiment_names], [])))
    metric_names = [metric_name for metric_name in metric_names if 'stream' in metric_name]
    metric_names = [metric_name.split('_')[1] for metric_name in metric_names]
    num_experiences = len([metric_name for metric_name in metric_names if metric_name == metric_names[0]])
    metric_names = sorted(set(metric_names))
    
    x = np.array(range(num_experiences))

    # Create the plot
    for metric_name in metric_names:
        all_experiment_means = []
        all_experiment_sems = []

        experiments_wo_metric = []
        for experiment_name in experiment_names:
            experiment_metrics = metrics[experiment_name]
            experiment_metric_means = []
            experiment_metric_stds = []

            if f'stream_{metric_name}_0' not in experiment_metrics:
                experiments_wo_metric.append(experiment_name)
                continue

            for i in range(num_experiences):
                experiment_metric_means.append(experiment_metrics[f'stream_{metric_name}_{i}']['mean'])
                experiment_metric_stds.append(experiment_metrics[f'stream_{metric_name}_{i}']['sem'])

            all_experiment_means.append(experiment_metric_means)
            all_experiment_sems.append(experiment_metric_stds)

        all_experiment_means = np.array(all_experiment_means)
        all_experiment_sems = np.array(all_experiment_sems)

        if isinstance(filter_condition, list):
            filter_condition = "comparison"

        output_path = os.path.join(experiments_path, f"stream_{metric_name}_{filter_condition}.pgf")
        metric_name = f"avg {metric_name.upper()}"
        y_labels = []

        for experiment_name in experiment_names:
            if experiment_name in experiments_wo_metric:
                continue
            experiment_name = get_short_name(experiment_name)
            y_labels.append(experiment_name)

        y_lim = None

        if metric_name == 'avg FID':
            y_lim = (0, 180)
        elif metric_name == 'avg KLD':
            max_idx = np.argmax(all_experiment_means)
            max_kld = all_experiment_means.flatten()[max_idx] + all_experiment_sems.flatten()[max_idx]
            y_lim = (0, 4)
        elif metric_name == 'avg ACC' or metric_name == 'avg FORGETTING':
            y_lim = (0, 1)
        metric_name = metric_name.replace('avg ', '')
        plot_line_std_graph(x, all_experiment_means, all_experiment_sems, 'Task', metric_name, None, output_path, x_ticks=x+1, y_labels=y_labels, y_lim=y_lim, size=(3, 1.7), annotate_last=True)


if __name__ == '__main__':
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
    filter_conditions = args.filter_condition
    if filter_conditions is None:
        filter_conditions = ""
    if isinstance(filter_conditions, str):
        filter_conditions = [filter_conditions]

    # create_comparison_histogram(args.experiments_path, args.filter_condition)

    all_experiments_results = {}
    # Iterate the experiments path and plot the metrics for each experiment
    # Each experiment might have several subexperiments
    # each one of them is a folder with the results of 5 different seeds
    for root_experiment in os.listdir(args.experiments_path):
        filter_condition = [condition for condition in filter_conditions if condition in root_experiment]

        if args.filter_condition is not None and not filter_condition:
            continue
        
        experiment_path = os.path.join(args.experiments_path, root_experiment)
        if not os.path.isdir(experiment_path):
            continue
        
        print(f"Plotting metrics for {root_experiment}...")
        plot_metrics(experiment_path)
        print(f"Creating summary for {root_experiment}...")
        _, _, _, _, metrics_mean_std = create_summary(experiment_path)
        experiment_name = root_experiment
        all_experiments_results[experiment_name] = metrics_mean_std

    print(f"Creating table with final results...")
    create_table(args.experiments_path, all_experiments_results, args.only_average, args.filter_condition)
    print(f"Creating comparison plots...")
    create_comparison_plots(args.experiments_path, all_experiments_results, args.filter_condition)