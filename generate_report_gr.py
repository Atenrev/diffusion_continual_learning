import os
import json
import pandas as pd
import numpy as np
import argparse
import regex as re
import torch

from src.common.visual import plot_line_std_graph, plot_bar


def __parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()    
    parser.add_argument("--experiments_path", type=str, 
                        default="results_fuji/smasipca/generative_replay_single/split_fmnist/")
    parser.add_argument("--only_average", action='store_true', default=True)
    parser.add_argument("--filter_condition", type=str, default="partial")
    return parser.parse_args()


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
            
            # TODO: plot the histograms stream_hist_pred/eval_phase/test_stream/Task000
            # and stream_hist_true/eval_phase/test_stream/Task000 at each experience
            for experience in df['training_exp'].unique():
                experience_data = df[df['training_exp'] == experience]
                experience_data = experience_data[experience_data['eval_exp'] == last_experience]
                fid_row = experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"]
                fid_value = fid_row.iloc[-1, -1]
                kld_row = experience_data[experience_data['metric_name'] == "stream_kld/eval_phase/test_stream/Task000"]
                kld_value = kld_row.iloc[-1, -1]
                top1_acc_row = experience_data[experience_data['metric_name'] == "Top1_Acc_Stream/eval_phase/test_stream/Task000"]
                top1_acc_value = top1_acc_row.iloc[-1, -1]
                stream_forgetting_row = experience_data[experience_data['metric_name'] == "StreamForgetting/eval_phase/test_stream"]
                stream_forgetting_value = stream_forgetting_row.iloc[-1, -1]
                stream_hist_pred_row = experience_data[experience_data['metric_name'] == "stream_hist_pred/eval_phase/test_stream/Task000"]
                stream_hist_pred_value = stream_hist_pred_row.iloc[-1, -1]
                stream_hist_true_row = experience_data[experience_data['metric_name'] == "stream_hist_true/eval_phase/test_stream/Task000"]
                stream_hist_true_value = stream_hist_true_row.iloc[-1, -1]

                metrics['avg FID'][-1].append(float(fid_value))
                metrics['avg KLD'][-1].append(float(kld_value))
                metrics['avg ACC'][-1].append(float(top1_acc_value))
                metrics['avg FORGETTING'][-1].append(float(stream_forgetting_value))
                metrics['stream_hist_pred'][-1].append([float(x) for x in stream_hist_pred_value[1:-1].strip().split()])
                metrics['stream_hist_true'][-1].append([float(x) for x in stream_hist_true_value[1:-1].strip().split()])

    # Join both stream_hist_pred and stream_hist_true
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
        values_list = np.array(values_list).squeeze()
        if "hist" not in metric:
            x = np.array(range(last_experience+1))
            if values_list.ndim == 2:
                means = values_list.mean(axis=0)
                stds = values_list.std(axis=0)
            else:
                means = values_list
                stds = np.zeros_like(means)
            output_path = os.path.join(experiment_path, f"{metric}.png")
            plot_line_std_graph(x, means, stds, 'Task', metric, metric, output_path, x_ticks=x)
        else:
            x = np.array(range(len(values_list[0][0])))
            if values_list.ndim == 4:
                means = values_list.mean(axis=0)
                stds = values_list.std(axis=0)
            else:
                means = values_list
                stds = np.zeros_like(means)
            for i in range(means.shape[0]):
                output_path = os.path.join(experiment_path, f"{metric}_exp_{i}.png")
                plot_bar(x, means[i, :], 'Class', 'Frequency', metric, output_path, y_labels=('True', 'Predicted'))


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
            for experience in df['training_exp'].unique():
                experience_data = df[df['training_exp'] == experience]
                experience_data = experience_data[experience_data['eval_exp'] == last_experience]
                fid_row = experience_data[experience_data['metric_name'] == "stream_fid/eval_phase/test_stream/Task000"]
                fid_value = float(fid_row.iloc[-1, -1])
                fids.append(fid_value)
                kld_row = experience_data[experience_data['metric_name'] == "stream_kld/eval_phase/test_stream/Task000"]
                kld_value = float(kld_row.iloc[-1, -1])
                top1_acc_row = experience_data[experience_data['metric_name'] == "Top1_Acc_Stream/eval_phase/test_stream/Task000"]
                top1_acc_value = float(top1_acc_row.iloc[-1, -1])
                stream_forgetting_row = experience_data[experience_data['metric_name'] == "StreamForgetting/eval_phase/test_stream"]
                stream_forgetting_value = float(stream_forgetting_row.iloc[-1, -1])
                last_seed_metrics[f'stream_fid_{experience}'] = fid_value
                last_seed_metrics[f'stream_kld_{experience}'] = kld_value
                last_seed_metrics[f'stream_acc_{experience}'] = top1_acc_value
                last_seed_metrics[f'stream_forgetting_{experience}'] = stream_forgetting_value

                if experience == last_experience:
                    last_seed_metrics['avg FID'] = fid_value
                    last_seed_metrics['avg KLD'] = kld_value
                    last_seed_metrics['avg ACC'] = top1_acc_value
                    last_seed_metrics['avg FORGETTING'] = stream_forgetting_value

            stream_fidauc = torch.trapz(torch.tensor(fids)).item()
            last_seed_metrics['avg FID AUC'] = stream_fidauc

            if stream_forgetting_value < best_fgt:
                best_seed_fgt = seed_folder
                best_fgt = stream_forgetting_value
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
            'std': np.std(metric_values)
        }

    # Save best seed and method to json
    with open(os.path.join(experiment_path, 'summary.json'), 'w') as f:
        json.dump({
            'best_seed_fgt': int(best_seed_fgt),
            'best_metrics_fgt': best_metrics_fgt,
            'best_seed_fidauc': int(best_seed_fidauc),
            'best_metrics_fidauc': best_metrics_fidauc,
            'metrics_mean_std': metrics_mean_std
        }, f, indent=4)

    return best_seed_fgt, best_metrics_fgt, best_seed_fidauc, best_metrics_fidauc, metrics_mean_std


def create_table(experiment_path: str, metrics: dict, only_average: bool = False, filter_condition: str = ""):
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
    metrics_names = sorted(metrics_names)
    metrics_names = [metric_name for metric_name in metrics_names if 'avg' in metric_name.lower() or not only_average]

    # Identify the best values for each metric
    best_values = {}
    for metric_name in metrics_names:
        best_value = float('inf')
        for experiment_metrics in metrics.values():
            mean = experiment_metrics[metric_name]['mean']
            mean = -mean if 'acc' in metric_name.lower() else mean
            if mean < best_value:
                best_value = mean
        best_values[metric_name] = -best_value if 'acc' in metric_name.lower() else best_value

    # Create the table
    table = []
    experiment_metrics = metrics.items()
    experiment_metrics = sorted(experiment_metrics, key=lambda x: [int(i) for i in re.findall(r'\d+', x[0])] if re.findall(r'\d+', x[0]) else x[0])
    for experiment_name, experiment_metrics in experiment_metrics:
        experiment_name = experiment_name.replace('_', ' ')
        experiment_name = ' '.join([word.capitalize() for word in experiment_name.split()])
        row = [experiment_name]
        for metric_name in metrics_names:
            mean = experiment_metrics[metric_name]['mean']
            std = experiment_metrics[metric_name]['std']
            formatted_mean = f"{mean:.3f}±{std:.3f}"
            if mean == best_values[metric_name]:
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
    df.to_csv(os.path.join(experiment_path, f'summary_{filter_condition}.csv'))

    # Save the dataframe to latex
    with open(os.path.join(experiment_path, f'summary_{filter_condition}.tex'), 'w') as f:
        f.write(df.to_latex())


def create_comparison_plots(experiments_path: str, metrics: dict, filter_condition: str = ""):
    """
    Create a plot with the comparison of the stream metrics for each 
    experiment and save it to a png file.

    Args:
        experiments_path (str): path to the experiments
        metrics (dict): dictionary with the metrics for each experiment
            format: {experiment_name: {metric: {mean: float, std: float}, ...}, ...}
    """
    experiment_names = list(metrics.keys())
    experiment_names = sorted(experiment_names, key=lambda x: [int(i) for i in re.findall(r'\d+', x)] if re.findall(r'\d+', x) else x)
    metric_names = list(metrics[experiment_names[0]].keys())
    metric_names = [metric_name for metric_name in metric_names if 'stream' in metric_name]
    metric_names = [metric_name.split('_')[1] for metric_name in metric_names]
    num_experiences = len([metric_name for metric_name in metric_names if metric_name == metric_names[0]])
    metric_names = sorted(set(metric_names))
    
    x = np.array(range(num_experiences))

    # Create the plot
    for metric_name in metric_names:
        all_experiment_means = []
        all_experiment_stds = []

        for experiment_name in experiment_names:
            experiment_metrics = metrics[experiment_name]
            experiment_metric_means = []
            experiment_metric_stds = []

            for i in range(num_experiences):
                experiment_metric_means.append(experiment_metrics[f'stream_{metric_name}_{i}']['mean'])
                experiment_metric_stds.append(experiment_metrics[f'stream_{metric_name}_{i}']['std'])

            all_experiment_means.append(experiment_metric_means)
            all_experiment_stds.append(experiment_metric_stds)

        all_experiment_means = np.array(all_experiment_means)
        all_experiment_stds = np.array(all_experiment_stds)
        output_path = os.path.join(experiments_path, f"stream_{metric_name}_{filter_condition}.png")
        metric_name = f"avg {metric_name.upper()}"
        plot_line_std_graph(x, all_experiment_means, all_experiment_stds, 'Task', metric_name, f"Comparison - {metric_name}", output_path, x_ticks=x, y_labels=experiment_names)


if __name__ == '__main__':
    args = __parse_args()

    all_experiments_results = {}
    # Iterate the experiments path and plot the metrics for each experiment
    # Each experiment might have several subexperiments
    # each one of them is a folder with the results of 5 different seeds
    for root_experiment in os.listdir(args.experiments_path):
        if args.filter_condition is not None and args.filter_condition not in root_experiment:
            continue

        experiment_path = os.path.join(args.experiments_path, root_experiment)
        if not os.path.isdir(experiment_path):
            continue
        
        try:
            print(f"Plotting metrics for {root_experiment}...")
            plot_metrics(experiment_path)
            print(f"Creating summary for {root_experiment}...")
            _, _, _, _, metrics_mean_std = create_summary(experiment_path)
            experiment_name = root_experiment
            all_experiments_results[experiment_name] = metrics_mean_std
        except Exception as e:
            print("Warning:", root_experiment, "failed")
            continue

    print(f"Creating table with final results...")
    create_table(args.experiments_path, all_experiments_results, args.only_average, args.filter_condition)
    print(f"Creating comparison plots...")
    create_comparison_plots(args.experiments_path, all_experiments_results, args.filter_condition)