# matplotlib plot functions

import matplotlib.pyplot as plt
import numpy as np



x = ['3', '10', '30', 'dynamic']

amzn_y = [0.54, 0.56, 0.36, 0.64]

apple_y = [0.64, 0.60, 0.52, 0.60]

msft_y = [0.5, 0.58, 0.38, 0.34]

nvidia_y = [0.60, 0.44, 0.54, 0.48]



# Total number of bar groups
num_groups = len(x)

# Setting up the figure and axes
fig, ax = plt.subplots()

# Creating a bar width
bar_width = 0.2

# Creating an index for the groups
index = np.arange(num_groups)

# Plotting each company's bars
bars_amzn = ax.bar(index, amzn_y, bar_width, label='Amazon')
bars_apple = ax.bar(index + bar_width, apple_y, bar_width, label='Apple')
bars_msft = ax.bar(index + 2*bar_width, msft_y, bar_width, label='Microsoft')
bars_nvidia = ax.bar(index + 3*bar_width, nvidia_y, bar_width, label='Nvidia')

# Adding x-axis and y-axis labels
ax.set_xlabel('Sequence Length (in days)')
ax.set_ylabel('Accuracy')

# Title of the plot
ax.set_title('Accuracy by Sequence Length and Company')

# Adding x-tick marks
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(x)

# Adding a legend
ax.legend()

# Function to add labels on the bars
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Call the function for each set of bars
add_labels(bars_amzn)
add_labels(bars_apple)
add_labels(bars_msft)
add_labels(bars_nvidia)

# Showing the plot
plt.show()