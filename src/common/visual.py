import numpy as np
import matplotlib.pyplot as plt


def plot_bar(x, y, x_label, y_label, title, save_path, color='skyblue', y_labels=None, size=(14, 8)):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(size[0], size[1])
    colors = ['black', 'skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'yellow', 'magenta']
    width = 1.0
    num_sublists = 1

    if isinstance(y[0], list) or isinstance(y[0], np.ndarray):
        # Plotting multiple bars for each sublist
        num_sublists = len(y)
        width = 1.0 / (num_sublists+2)  # Adjust the width based on the number of sublists
        x_positions = np.arange(len(x))

        for i, sublist_y in enumerate(y):
            lbl = f"{y_label} {i+1}" 
            if y_labels is not None:
                lbl = y_labels[i]
            ax.bar(x_positions + i * width, sublist_y, width=width, label=lbl, color=colors[i])

        ax.legend()

    else:
        # Plotting a single bar graph
        ax.bar(x, y, color=color)

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjusting the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Rotating the x-axis labels if necessary
    # plt.xticks(rotation=45)

    # Setting the x-axis tick positions and labels
    ax.set_xticks([i + width * (num_sublists-1) / 2 for i in range(len(x))])
    ax.set_xticklabels(x)

    # Save the graph to disk
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_line_graph(x, y, x_label, y_label, title, save_path, color='skyblue', log_x=False, x_ticks=None, second_x=None, second_x_label=None, y_lim=None, size=(8, 6)):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    fig.set_size_inches(size[0], size[1])

    # Plotting the line graph with small markers
    ax.plot(x, y, marker='o', linestyle='-', color=color, markersize=3)

    # Setting the x-axis scale
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Setting the x-axis tick positions and labels
    if x_ticks is not None:
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks)

    # Setting the y-axis scale
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjusting the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    if second_x is not None and second_x_label is not None:
        ax2 = ax.twiny()
        ax2.set_xlabel(second_x_label)
        ax2.set_xticks(np.arange(0, len(second_x)))
        ax2.set_xticklabels(second_x)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

    plt.savefig(save_path)
    plt.close()


def plot_line_std_graph(x, y, std, x_label, y_label, title, save_path, colors=None, log_x=False, x_ticks=None, x_labels=None, y_labels=None, y_lim=None, size=(12, 8), annotate_last=False):
    # Set up the figure and axis
    fig, ax = plt.subplots()
    # Make figure larger
    fig.set_size_inches(size[0], size[1])

    # Plotting multiple line graphs
    if colors is None:
        if isinstance(y[0], list) or isinstance(y[0], np.ndarray):
            # Different colors for each sublist
            colors = ['skyblue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'black', 'yellow', 'magenta']
        else:
            colors = 'skyblue'
    
    if not isinstance(y[0], list) and not isinstance(y[0], np.ndarray):
        y = [y]
        std = [std]
        colors = [colors]

    x = np.array(x)
    if len(x.shape) == 1:
        x = [x] * len(y)

    for i, y_vals in enumerate(y):
        color = colors[i]
        if y_labels is not None and not annotate_last:
            label = y_labels[i]
        else:
            label = None
        
        ax.plot(x[i], y_vals, linestyle='-', color=color, label=label)
        ax.fill_between(x[i], y_vals - std[i], y_vals + std[i], color=color, alpha=0.2)

    if annotate_last and y_labels is not None:
        y_arr = np.array(y)[:,-1]
        # Order the labels by the last value
        y_labels = [y_labels[i] for i in np.argsort(y_arr)]
        colors = [colors[i] for i in np.argsort(y_arr)]
        offset = 0.1 * y_lim[1]
        
        # Annotate the labels in order
        for i in range(len(y_labels)):
            y_val = offset * i + offset
            ax.annotate(y_labels[i], xy=(x[i][-1], y_val), xytext=(x[i][-1]+0.25, y_val), va='center', color=colors[i], weight='bold')

    # Setting the x-axis scale
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(x[0])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    # Setting the x-axis tick positions and labels
    if x_ticks is not None:
        if x_labels is None:
            ax.set_xticks(np.arange(0, len(x_ticks)))
            ax.set_xticklabels(x_ticks)
        else:
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)

    # Setting the y-axis scale
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

    # Adding labels, legend, and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if y_labels is not None and not annotate_last:
        ax.legend()

    # Adjusting the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the graph to disk
    # plt.tight_layout()
    plt.savefig(save_path)
    plt.close()