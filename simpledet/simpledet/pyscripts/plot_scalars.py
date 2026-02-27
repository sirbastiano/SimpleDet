#!/usr/bin/env python3
"""
Script to read scalars.json file and generate bbox_mAP_50 plot.
Saves the plot in the same directory as the scalars.json file.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse
from typing import Optional, List
from pathlib import Path


def read_and_plot_bbox_map50(json_file_path: str, save_plot: bool = True) -> None:
    """
    Read scalars.json file and plot bbox_mAP_50 values over training steps.
    
    Args:
        json_file_path (str): Path to the scalars.json file
        save_plot (bool): Whether to save the plot to file
    """
    assert os.path.exists(json_file_path), f'File does not exist: {json_file_path}'
    
    steps = []
    bbox_map_50_values = []
    
    # Read the JSON file line by line
    with open(json_file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                # Check if this line contains bbox_mAP_50
                if 'coco/bbox_mAP_50' in data and 'step' in data:
                    steps.append(data['step'])
                    bbox_map_50_values.append(data['coco/bbox_mAP_50'])
            except json.JSONDecodeError:
                # Skip invalid JSON lines
                continue
    
    if not bbox_map_50_values:
        print('No bbox_mAP_50 values found in the file!')
        return
    
    # Create the plot
    plt.figure(figsize=(12, 8), dpi=300)
    plt.plot(steps, bbox_map_50_values, 'b-o', linewidth=2, markersize=6)
    plt.title('COCO bbox mAP@0.5 Over Training Steps', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step (Epoch)', fontsize=14)
    plt.ylabel('bbox mAP@0.5', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, max(bbox_map_50_values) * 1.1)  # Set y-axis limit with some padding
    
    # Add value annotations on points
    for step, value in zip(steps, bbox_map_50_values):
        plt.annotate(f'{value:.3f}', 
                    (step, value), 
                    textcoords='offset points', 
                    xytext=(0, 10), 
                    ha='center',
                    fontsize=10)
    
    plt.tight_layout()
    
    if save_plot:
        # Save plot in the same directory as the scalars.json file
        output_dir = Path(json_file_path).parent
        output_path = output_dir / 'bbox_mAP_50_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to: {output_path}')
    
    plt.show()
    
    # Print summary statistics
    print(f'\nbbox_mAP_50 Summary:')
    print(f'Number of evaluation points: {len(bbox_map_50_values)}')
    print(f'Best mAP@0.5: {max(bbox_map_50_values):.3f} at step {steps[np.argmax(bbox_map_50_values)]}')
    print(f'Final mAP@0.5: {bbox_map_50_values[-1]:.3f}')
    print(f'Average mAP@0.5: {np.mean(bbox_map_50_values):.3f}')


def read_and_plot_metrics(json_file_path: str, metrics_to_plot: Optional[List[str]] = None, save_plot: bool = True) -> None:
    """
    Read scalars.json file and plot multiple COCO metrics.
    
    Args:
        json_file_path (str): Path to the scalars.json file
        metrics_to_plot (Optional[List[str]]): List of metrics to plot. If None, plots common COCO metrics.
        save_plot (bool): Whether to save the plot to file
    """
    assert os.path.exists(json_file_path), f'File does not exist: {json_file_path}'
    
    if metrics_to_plot is None:
        metrics_to_plot = ['coco/bbox_mAP', 'coco/bbox_mAP_50', 'coco/bbox_mAP_75']
    
    # Dictionary to store data for each metric
    metrics_data = {metric: {'steps': [], 'values': []} for metric in metrics_to_plot}
    
    # Read the JSON file line by line
    with open(json_file_path, 'r') as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                if 'step' in data:
                    for metric in metrics_to_plot:
                        if metric in data:
                            metrics_data[metric]['steps'].append(data['step'])
                            metrics_data[metric]['values'].append(data[metric])
            except json.JSONDecodeError:
                continue
    
    # Create subplots
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(14, 5 * len(metrics_to_plot)), dpi=300)
    if len(metrics_to_plot) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, metric in enumerate(metrics_to_plot):
        if not metrics_data[metric]['values']:
            print(f'No data found for {metric}')
            continue
            
        steps = metrics_data[metric]['steps']
        values = metrics_data[metric]['values']
        
        axes[i].plot(steps, values, color=colors[i % len(colors)], marker='o',
                    linewidth=2, markersize=6, label=metric)
        axes[i].set_title(f'{metric} Over Training Steps', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Training Step (Epoch)', fontsize=12)
        axes[i].set_ylabel(metric.split('/')[-1], fontsize=12)
        axes[i].grid(True, alpha=0.3)
        axes[i].legend()
        
        # Add best value annotation
        best_idx = np.argmax(values)
        axes[i].annotate(f'Best: {values[best_idx]:.3f}', 
                        xy=(steps[best_idx], values[best_idx]),
                        xytext=(10, 10), 
                        textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    if save_plot:
        # Save plot in the same directory as the scalars.json file
        output_dir = Path(json_file_path).parent
        output_path = output_dir / 'coco_metrics_plot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Metrics plot saved to: {output_path}')
    
    plt.show()
    
    # Print summary for bbox_mAP_50 specifically
    if 'coco/bbox_mAP_50' in metrics_data and metrics_data['coco/bbox_mAP_50']['values']:
        values = metrics_data['coco/bbox_mAP_50']['values']
        steps = metrics_data['coco/bbox_mAP_50']['steps']
        print(f'\nbbox_mAP_50 Summary:')
        print(f'Best: {max(values):.3f} at step {steps[np.argmax(values)]}')
        print(f'Final: {values[-1]:.3f}')
        print(f'Improvement: {values[-1] - values[0]:.3f}')


def main():
    """Main function to parse arguments and generate plots."""
    parser = argparse.ArgumentParser(
        description='Generate plots from scalars.json file',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python plot_scalars.py /path/to/scalars.json
  python plot_scalars.py /path/to/scalars.json --plot-type bbox_only
  python plot_scalars.py /path/to/scalars.json --plot-type all_metrics --no-save
        """
    )
    
    parser.add_argument(
        'json_path',
        type=str,
        help='Path to the scalars.json file'
    )
    
    parser.add_argument(
        '--plot-type',
        choices=['bbox_only', 'all_metrics'],
        default='bbox_only',
        help='Type of plot to generate (default: bbox_only)'
    )
    
    parser.add_argument(
        '--metrics',
        nargs='+',
        default=None,
        help='Specific metrics to plot (only used with all_metrics type)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save the plot to file'
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.json_path):
        print(f'Error: File does not exist: {args.json_path}')
        return 1
    
    if not args.json_path.endswith('scalars.json'):
        print(f'Warning: File does not appear to be a scalars.json file: {args.json_path}')
    
    save_plot = not args.no_save
    
    try:
        if args.plot_type == 'bbox_only':
            read_and_plot_bbox_map50(args.json_path, save_plot=save_plot)
        elif args.plot_type == 'all_metrics':
            read_and_plot_metrics(args.json_path, metrics_to_plot=args.metrics, save_plot=save_plot)
    except Exception as e:
        print(f'Error generating plot: {e}')
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())