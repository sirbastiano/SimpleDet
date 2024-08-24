import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def set_style(scale_factor=1.5, font_family='STIXGeneral', dpi=500, fig_width=15, fig_height=5):
    """
    Sets the styling parameters for Matplotlib and Seaborn plots, including figure dimensions.

    Parameters:
    scale_factor (float, optional): Factor to scale the font sizes. Default is 1.5.
    font_family (str, optional): The font family to use for the plots. Default is 'STIXGeneral'.
    dpi (int, optional): Dots per inch (DPI) setting for the figure. Default is 500.
    fig_width (float, optional): Width of the figure in inches. Default is 15.
    fig_height (float, optional): Height of the figure in inches. Default is 5.
    """
    plt.rcParams['font.family'] = font_family
    plt.rcParams['figure.dpi'] = dpi
    
    font_size = 13 * scale_factor
    plt.rcParams.update({
        'font.size': font_size,
        'axes.titlesize': font_size,
        'axes.labelsize': font_size,
        'xtick.labelsize': font_size,
        'ytick.labelsize': font_size,
        'legend.fontsize': font_size,
        'figure.titlesize': font_size,
        'figure.figsize': (fig_width * scale_factor, fig_height * scale_factor)
    })

    sns.set_context("paper", rc={
        "font.size": font_size,
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size
    })
    
 
set_style()

def smooth_curve(y, window_size=5):
    """Smooth the curve using a moving average with a given window size."""
    return np.convolve(y, np.ones(window_size)/window_size, mode='same')

def plot_precision_recall(ax, pr_x_band, axins_true=False):
    """
    Helper function to plot precision-recall curves with shaded areas for standard deviation.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The subplot axes to plot on.
    pr_x_band (list): A list of evaluation results for different spectral bands.
    """
    
    if len(pr_x_band) == 4:
        print('4 labels active') 
        V = [pr_x_band[f'b{i}'] for i in [2,3,4,8]]
        coco_eval_lists = V
        labels = [f"$B_{{{i}}}$" for i in [2,3,4,8]]
    
    elif len(pr_x_band) == 12:
        V = [pr_x_band[f'b{i}'] for i in range(1, 13)]
        coco_eval_lists = V
        labels = [f"$B_{{{i}}}$" for i in range(1, 13)]
    
    else:
        identifiers = list(pr_x_band.keys())
        V = [pr_x_band[i] for i in identifiers]
        coco_eval_lists = V
        labels = identifiers

    colors = sns.color_palette("colorblind", len(labels) + 5) 
    recall_list = []
    lower_bound_list = []
    upper_bound_list = []
    mean_precision_list = []
    
    for i, coco_eval_list in enumerate(coco_eval_lists):
        # Extract precision values and recall
        all_precisions = []
        recall = np.arange(0.0, 1.01, 0.01)
        recall_list.append(recall)

        for coco_eval in coco_eval_list:
            precision = coco_eval.eval['precision'][0, :, 0, 0, 2]  # precision for IoU=0.50:0.95 and area=all
            all_precisions.append(precision)

        # Convert list of all precisions to a numpy array for easier manipulation
        all_precisions = np.array(all_precisions)

        # Compute the mean precision and the standard deviation for shading
        mean_precision = np.mean(all_precisions, axis=0)
        mean_precision_list.append(mean_precision)
        std_precision = np.std(all_precisions, axis=0, ddof=1)
        std_precision_smooth = smooth_curve(std_precision, window_size=5)
        mean_precision_smooth = smooth_curve(mean_precision, window_size=5)

        upper_bound = mean_precision + std_precision_smooth
        lower_bound = mean_precision - std_precision_smooth
        
        upper_bound_list.append(upper_bound)
        lower_bound_list.append(lower_bound)

        # Plot shaded area and mean precision line
        ax.fill_between(recall, lower_bound, upper_bound, color=colors[i], alpha=0.3)
        ax.plot(recall, mean_precision, label=reformat(labels[i]), color=colors[i])

    if axins_true:
        # Add zoomed-in plot
        axins = ax.inset_axes([0.05, 0.5, 0.35, 0.35])
        
        # Iterate over all curves to add them to the zoomed-in plot
        for i in range(len(recall_list)):
            recall = recall_list[i]
            lower_bound = lower_bound_list[i]
            upper_bound = upper_bound_list[i]
            mean_precision = mean_precision_list[i]
            axins.fill_between(recall, lower_bound, upper_bound, color=colors[i], alpha=0.3)
            axins.plot(recall, mean_precision, label=labels[i], color=colors[i])
        
        axins.set_xlim(0.72, 0.85)
        axins.set_ylim(0.8, 1)
        axins.set_xticklabels('')
        axins.set_yticklabels('')
        ax.indicate_inset_zoom(axins)
    # ax.set_title('Precision-Recall Curves by Band')
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    
    ax.legend()
    ax.set_ylim([0,1.1])
    ax.grid(True)

def reformat(s):
    s = s.split('_')
    # Capitalize
    x = [i.split('b')[-1] for i in s]
    
    x = [f'$B_{{{i}}}$' for i in x]
    x = '-'.join(x)
    return x


def plot_error_bars(ax, grouped, labels=None):
    """
    Helper function to plot error bars for COCO bbox mAP metrics by spectral band.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The subplot axes to plot on.
    grouped (pandas.DataFrame): DataFrame containing mean and std of coco/bbox mAP metrics by spectral band.
    """
    
    # Plotting the error bars for each metric
    ax.errorbar(grouped['Band'], grouped['mean_mAP'], yerr=grouped['std_mAP'], 
                 label='$AP$', fmt='-o', capsize=5)

    ax.errorbar(grouped['Band'], grouped['mean_mAP_50'], yerr=grouped['std_mAP_50'], 
                 label='$AP_{50}$', fmt='-s', capsize=5)

    ax.errorbar(grouped['Band'], grouped['mean_mAP_75'], yerr=grouped['std_mAP_75'], 
                 label='$AP_{75}$', fmt='-^', capsize=5)


    ax.set_ylim([0,1.15])
    
    if labels is not None:
        ax.set_xticks(range(len(labels)), labels=labels, fontsize=15)
    ax.set_xlabel('Spectral Band')
    ax.set_ylabel('Mean Metric Value')
    ax.legend()
    ax.grid(True)

# Main plotting function
def combined_plot(pr_x_band, all_band, savepath):
    """
    Main function to generate subplots for precision-recall curves and error bars for COCO bbox mAP metrics.

    Parameters:
    pr_x_band (list): List of evaluation results for different spectral bands.
    all_band (pandas.DataFrame): DataFrame containing coco/bbox mAP metrics grouped by spectral band.
    """
    # Create figure and subplots
    scale_factor = 1.5
    plt.figure(figsize=(15 * scale_factor, 5 * scale_factor))

    # First subplot: Precision-Recall Curves
    ax1 = plt.subplot(1, 2, 1)
    plot_precision_recall(ax1, pr_x_band)

    

    # Group the data by 'Band' and calculate the mean and standard deviation for second subplot
    grouped = all_band.groupby('Band').agg(
        mean_mAP=('coco/bbox_mAP', 'mean'),
        std_mAP=('coco/bbox_mAP', 'std'),
        mean_mAP_50=('coco/bbox_mAP_50', 'mean'),
        std_mAP_50=('coco/bbox_mAP_50', 'std'),
        mean_mAP_75=('coco/bbox_mAP_75', 'mean'),
        std_mAP_75=('coco/bbox_mAP_75', 'std')
    ).reset_index()

    labels = [reformat(idx[0]) for idx in list(all_band.groupby(['Band','LR','BS','ME']).mean().index)]

    # Second subplot: Error bars for COCO bbox mAP metrics
    ax2 = plt.subplot(1, 2, 2)
    plot_error_bars(ax2, grouped, labels)


    # Add text (a) and (b) under axes
    ax1.text(0.25, -0.14, '(a) Precision-Recall Curves (VDVRaw)', transform=ax1.transAxes, va='top')
    ax2.text(0.25, -0.14, '(b) Error Plot of BBox Metrics (VDVRaw)', transform=ax2.transAxes, va='top')

    # Adjust layout and display plot
    plt.subplots_adjust(hspace=0.45)
    plt.savefig(savepath, bbox_inches='tight')
    plt.show()
    return grouped