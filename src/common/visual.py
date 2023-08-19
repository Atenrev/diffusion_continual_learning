import numpy as np
import matplotlib.pyplot as plt


def plot_bar(x, y, x_label, y_label, title, save_path, color='skyblue', y_labels=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    if isinstance(y[0], list) or isinstance(y[0], np.ndarray):
        # Plotting multiple bars for each sublist
        num_sublists = len(y)
        width = 0.8 / num_sublists  # Adjust the width based on the number of sublists
        x_positions = np.arange(len(x))

        for i, sublist_y in enumerate(y):
            lbl = f"{y_label} {i+1}" 
            if y_labels is not None:
                lbl = y_labels[i]
            ax.bar(x_positions + i * width, sublist_y, width=width, label=lbl)

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
    plt.xticks(rotation=45)

    # Setting the x-axis tick positions and labels
    ax.set_xticks(np.arange(0, len(x)))
    ax.set_xticklabels(x)

    # Save the graph to disk
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_line_graph(x, y, x_label, y_label, title, save_path, color='skyblue', log_x=False, second_x=None, second_x_label=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the line graph
    ax.plot(x, y, marker='o', linestyle='-', color=color)

    # Setting the x-axis scale
    if log_x:
        ax.set_xscale('log')
        ax.set_xticks(x)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

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

    # Save the graph to disk
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_line_std_graph(x, y, std, x_label, y_label, title, save_path, color='skyblue', x_ticks=None):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the line graph
    ax.plot(x, y, linestyle='-', color=color)
    ax.fill_between(x, y - std, y + std, color=color, alpha=0.2)

    # Setting the x-axis tick positions and labels
    if x_ticks is not None:
        ax.set_xticks(np.arange(0, len(x)))
        ax.set_xticklabels(x_ticks)

    # Adding labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    # Adjusting the appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save the graph to disk
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()