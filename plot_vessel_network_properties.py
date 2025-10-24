import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import matplotlib.patches as patches

# --------------------------------------------
# Config (USER-EDITABLE SECTION)
csv_file_path = r"C:\..\intervessel_distances.csv"
manual_labels =  ['Skeleton_1', 'Skeleton_2', 'Skeleton_3', 'Skeleton_4', 'Skeleton_5', 'Skeleton_6', 'Skeleton_7', 'Skeleton_8', 'Skeleton_9', 'Skeleton_10', 'Skeleton_11']   # Set to None for import from .csv column labels. Note: the no. labels must match the no. data columns in the .csv file.
y_axis_bounds = (None, None)  # Set to (None, None) for auto-scaling, or (ymin, ymax) for custom
y_major_step = None   # Set to None for default spacing
y_minor_step = None    # Set to None for no minor grid
output_plot_path = "IVD_violin_plot.png"
plot_title = "IVD by Algorithm"
y_axis_label = "Intervessel distance (microns)"
# --------------------------------------------

def load_and_reshape_data(filepath, manual_labels=None):
    df = pd.read_csv(filepath, header=0, index_col=0)
    if manual_labels:
        df.columns = manual_labels
    df_long = df.melt(var_name="Algorithm", value_name="Value")
    df_long.dropna(inplace=True)
    return df_long

def plot_combined(df_long, title, ylabel, y_bounds=None, y_major=None, y_minor=None):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Get unique algorithms and assign colorblind-safe colors
    unique_algorithms = df_long["Algorithm"].unique()
    colors = sns.color_palette("colorblind", n_colors=len(unique_algorithms))
    color_map = dict(zip(unique_algorithms, colors))

    # Violin plots (semi-opaque)
    for i, algorithm in enumerate(unique_algorithms):
        subset = df_long[df_long["Algorithm"] == algorithm]
        sns.violinplot(x="Algorithm", y="Value", data=subset, inner=None,
                       linewidth=0, color=color_map[algorithm], alpha=0.4)

    # Calculate medians first for custom boxplots
    medians = {}
    for i, algorithm in enumerate(unique_algorithms):
        subset = df_long[df_long["Algorithm"] == algorithm]
        medians[algorithm] = subset["Value"].median()

    # Create custom box plots with gaps at median
    ax = plt.gca()
    box_width = 0.25
    gap_width = 0.2  

    for i, algorithm in enumerate(unique_algorithms):
        subset = df_long[df_long["Algorithm"] == algorithm]
        
        # Calculate quartiles
        q1 = subset["Value"].quantile(0.25)
        median = medians[algorithm]
        q3 = subset["Value"].quantile(0.75)
        
        color = color_map[algorithm]
        
        # Draw the lower part of the box (from q1 to median - gap/2)
        rect_lower = patches.Rectangle(
            (i - box_width/2, q1),
            box_width, median - q1 - gap_width/2,
            facecolor=color, alpha=0.5, edgecolor=None
        )
        ax.add_patch(rect_lower)
        
        # Draw the upper part of the box (from median + gap/2 to q3)
        rect_upper = patches.Rectangle(
            (i - box_width/2, median + gap_width/2),
            box_width, q3 - (median + gap_width/2),
            facecolor=color, alpha=0.5, edgecolor=None
        )
        ax.add_patch(rect_upper)

    # Scatter points
    for i, algorithm in enumerate(unique_algorithms):
        subset = df_long[df_long["Algorithm"] == algorithm]
        x_vals = np.full(len(subset), i)
        plt.scatter(x_vals, subset["Value"], color=color_map[algorithm], s=5, alpha=0.6, edgecolor="none")

    # Mean (dot) and median (line)
    for i, algorithm in enumerate(unique_algorithms):
        subset = df_long[df_long["Algorithm"] == algorithm]
        mean_val = subset["Value"].mean()
        median_val = medians[algorithm]
        plt.scatter(i, mean_val, color="black", s=20, zorder=10)
        plt.plot([i - 0.1, i + 0.1], [median_val, median_val], color="black", linewidth=1.0, zorder=9)

    # Axis formatting
    plt.xticks(ticks=range(len(unique_algorithms)), labels=unique_algorithms, fontsize=12)
    plt.title(title, fontsize=14)
    plt.ylabel(ylabel)
    plt.xlabel("")

    # Y-axis bounds
    if y_bounds is not None and any(b is not None for b in y_bounds):
        plt.ylim(*y_bounds)

    # Grid spacing and tick locators
    if y_major is not None:
        ax.yaxis.set_major_locator(ticker.MultipleLocator(y_major))
        ax.grid(which="major", axis="y", linestyle="-", linewidth=1.0, alpha=0.7)
    if y_minor is not None:
        ax.yaxis.set_minor_locator(ticker.MultipleLocator(y_minor))
        ax.grid(which="minor", axis="y", linestyle=":", linewidth=0.8, alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    plt.show()

def main():
    df_long = load_and_reshape_data(csv_file_path, manual_labels)
    plot_combined(df_long, title=plot_title, ylabel=y_axis_label,
                  y_bounds=y_axis_bounds, y_major=y_major_step, y_minor=y_minor_step)

if __name__ == "__main__":
    main()