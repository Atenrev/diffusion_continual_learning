import numpy as np
import matplotlib.pyplot as plt


def plot_bar(x, y, x_label, y_label, title, save_path):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the bar graph
    ax.bar(x, y, color='skyblue')

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


def plot_line_graph(x, y, x_label, y_label, title, save_path):
    # Set up the figure and axis
    fig, ax = plt.subplots()

    # Plotting the FID vs generation steps
    ax.plot(x, y, marker='o', linestyle='-', color='skyblue')

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
