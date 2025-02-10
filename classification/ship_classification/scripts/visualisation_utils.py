import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_csv_values(csv_file_path, x_column, y_column):
    """
    Reads a CSV file into a DataFrame and plots specified columns using Seaborn.

    Parameters:
    csv_file_path (str): The file path to the CSV file.
    x_column (str): The column to be plotted on the x-axis.
    y_column (str): The column to be plotted on the y-axis.

    Returns:
    matplotlib.figure.Figure: The figure object containing the plot, or None if an error occurs.
    """
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_file_path)
        
        # Check if the specified columns exist in the DataFrame
        if x_column not in data.columns or y_column not in data.columns:
            print(f"Error: Specified columns '{x_column}' or '{y_column}' do not exist in the CSV.")
            return None
        
        # Check if the specified columns are numeric
        if not pd.api.types.is_numeric_dtype(data[x_column]) or not pd.api.types.is_numeric_dtype(data[y_column]):
            print(f"Error: Specified columns must be numeric.")
            return None

        # Create a Seaborn plot
        sns.set_theme(style="darkgrid")
        fig, ax = plt.subplots()
        sns.lineplot(data=data, x=x_column, y=y_column, ax=ax)
        
        # Set the title and labels
        ax.set_title(f'{y_column} vs. {x_column}')
        ax.set_xlabel(x_column)
        ax.set_ylabel(y_column)
        
        # Show the plot
        plt.show()
        return fig
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def save_metrics_to_csv(metrics_list, file_path):
    """
    Saves a list of dictionaries with test metrics into a CSV file.

    Parameters:
    metrics_list (list): A list of dictionaries where each dictionary contains metric values.
    file_name (str): The file path or name where the CSV will be saved.
    """
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(metrics_list)

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=False)

def plot_all_data(csv_file_path, output_dir):
    y_fields = ['val_loss',
                'val_acc',
                'lr',
                'precision',
                'recall',
                'f1_score',
                'cohen_kappa',
                'balanced_accuracy']
    
    for y in y_fields:
        fig = plot_csv_values(csv_file_path, 'epoch', y)
        if fig is not None:
            fig.savefig(f"{output_dir}/{y}.png")
            plt.close(fig)
            

def main():
    print('Starting main function...')
    
    plot_csv_values('/Data_large/marine/Dom/classification/train_64_perfect/lightning_logs/venus/vessels/64x64_perfect/fp32_bands_5_10_seed_42/version_0/metrics.csv', 'epoch', 'val_loss')


   
if __name__ == '__main__':
    main()
    
