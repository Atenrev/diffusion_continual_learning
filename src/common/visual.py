import numpy as np
import matplotlib.pyplot as plt


def plot_bar(x, y, x_label, y_label, title, save_path, color='skyblue'):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the bar graph
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


def plot_line_graph(x, y, x_label, y_label, title, save_path, color='skyblue'):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the line graph
    ax.plot(x, y, marker='o', linestyle='-', color=color)

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


def plot_line_std_graph(x, y, std, x_label, y_label, title, save_path, color='skyblue'):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the line graph
    ax.plot(x, y, linestyle='-', color=color)
    ax.fill_between(x, y - std, y + std, color=color, alpha=0.2)

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