import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from style import set_style

# Data for BS, S, Avg Latency, and Throughput
data = {
    "BS": [1, 2, 4, 1, 2, 4, 1, 2, 4],
    "S": [128, 128, 128, 256, 256, 256, 352, 352, 352],
    "Avg Latency (ms)": [244.72, 490.67, 982.75, 685.37, 1388.11, 1908.84, 1837.88, 3613.55, 6857.57],
    "Throughput (FPS)": [16.28, 16.24, 16.20, 5.81, 5.73, 4.15, 2.16, 2.18, 2.27]
}

# Convert the data into a DataFrame
df = pd.DataFrame(data)

set_style(scale_factor=1)

# Adjust the layout to have one plot on top of another on the left and a new subplot on the right
fig, axes = plt.subplots(2, 2, figsize=(15, 4.5))

# Plot 1: Average Latency for different BS and S (top-left)
sns.barplot(x="BS", y="Avg Latency (ms)", hue="S", data=df, ax=axes[0, 0], palette="Greys", edgecolor='black', linewidth=1)
axes[0, 0].set_title('(a)', pad=-20, loc='left', fontweight='bold', x=0.025)
axes[0, 0].set_ylabel('Avg Latency (ms)')
axes[0, 0].set_xlabel('Batch Size (BS)')
axes[0, 0].legend(title='Input Size', loc='upper right', title_fontsize='small')
axes[0, 0].set_xlim([-.75, 3.25])
axes[0, 0].set_ylim([0, 8100])

# Add values on top of bars for Plot 1
for p in axes[0, 0].patches:
    width = p.get_width()
    if width > 0:
        axes[0, 0].annotate(format(p.get_height(), '.0f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9), textcoords='offset points')

# Plot 2: Throughput for different BS and S (bottom-left)
sns.barplot(x="BS", y="Throughput (FPS)", hue="S", data=df, ax=axes[1, 0], palette="Greys", edgecolor='black', linewidth=1)
axes[1, 0].set_title('(b)', pad=-20, loc='left', fontweight='bold', x=0.025)
axes[1, 0].set_ylabel('Throughput (FPS)')
axes[1, 0].set_xlabel('Batch Size (BS)')
axes[1, 0].set_xlim([-.75, 3.25])
axes[1, 0].set_ylim([0, 20])
axes[1, 0].legend(title='Input Size', loc='upper right', title_fontsize='small')

# Add values on top of bars for Plot 2
for p in axes[1, 0].patches:
    width = p.get_width()
    if width > 0:
        axes[1, 0].annotate(format(p.get_height(), '.1f'),
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 9), textcoords='offset points')

# Placeholder for the new subplot on the right (top-right and bottom-right combined)
axes[0, 1].remove()
axes[1, 1].remove()
ax_new = fig.add_subplot(1, 2, 2)

# Example new plot: Scatter plot of Avg Latency vs Throughput
# Calculate the number of windows required to cover a 2048x2048 image for different window sizes (S)
df['Num Windows'] = (2048 / df['S']) ** 2

# Calculate the new metric: Number of windows divided by FPS
df['Windows per FPS'] = df['Num Windows'] / df['Throughput (FPS)']

# Plotting the scatter plot for Avg Latency vs Throughput, with the new metric Windows per FPS
sns.barplot(x="Windows per FPS", y="BS", hue="S", data=df, ax=ax_new, palette="Greys", orient='h', width=0.6, edgecolor='black', linewidth=1)
ax_new.set_title('(c)', pad=-20, loc='left', fontweight='bold', x=0.025)
ax_new.set_ylabel('Batch Size (BS)')
ax_new.set_xlabel('Total Inference Time (s)')
ax_new.set_xlim([0, 24])
ax_new.set_ylim([-0.5, 2.75])
ax_new.legend(title='Input Size', loc='upper right', title_fontsize='small')

# Add values on top of bars for the new plot
for p in ax_new.patches:
    width = p.get_width()
    if width > 0:
        ax_new.annotate(format(width, '.1f'),
                        (width, p.get_y() + p.get_height() / 2.),
                        ha='center', va='center',
                        xytext=(15, 0), textcoords='offset points')

plt.tight_layout()
plt.savefig('Fig15.png', dpi=400, bbox_inches='tight')
plt.show()
