import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

def set_style(scale_factor=1.5, font_family='STIXGeneral', dpi=500, fig_width=15, fig_height=5, fontsize=16):
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
    
    font_size = fontsize * scale_factor
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