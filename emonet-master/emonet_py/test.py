import os.path
import matplotlib.pyplot as plt
import explanations_emonet, global_analysis, local_analysis, emonet
import pandas as pd
import numpy as np

# Your dictionary of emotions and objects
data = {
    'Romance': {'Human face': 0.2308, 'Person': 0.6923, 'Clothing': 0.0769},
    'Amusement': {'Person': 0.6441, 'Clothing': 0.2542, 'Human face': 0.0847, 'Furniture': 0.0169},
    'Joy': {'Human face': 0.3, 'Person': 0.4667, 'Clothing': 0.1667, 'Vehicle': 0.0333, 'Sports equipment': 0.0333},
    'Excitement': {'Person': 0.8824, 'Clothing': 0.0588, 'Sports equipment': 0.0588},
    'Confusion': {'Clothing': 0.4444, 'Human face': 0.1111, 'Person': 0.4444},
    'Awe': {'Person': 1.0},
    'Sadness': {'Person': 0.7273, 'Clothing': 0.2727},
    'Disgust': {'Person': 0.5, 'Clothing': 0.5},
    'Boredom': {'Human face': 0.3889, 'Person': 0.5556, 'Clothing': 0.0556},
    'Fear': {'Person': 0.5, 'Clothing': 0.5},
    'Anxiety': {'Person': 1.0},
    'Interest': {'Person': 1.0}
}

# Create a figure and axis object
fig, ax = plt.subplots(figsize=(12, 8))

# List to store the positions of the bars
bar_positions = []

# Iterate through each emotion
for i, (emotion, objects) in enumerate(data.items()):
    # Get the objects and their values for the current emotion
    objects_list = list(objects.keys())
    values_list = list(objects.values())

    # Position of the bars for the current emotion
    pos = [p + i for p in range(len(objects_list))]

    # Plot the bars
    bars = ax.bar(pos, values_list, label=emotion)

    # Add object names at each bar
    for bar, obj in zip(bars, objects_list):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), obj, ha='center', va='bottom', rotation=90)

    # Store the bar positions for later use
    bar_positions.extend(pos)

# Set the x-axis labels
ax.set_xticks(bar_positions)
ax.set_xticklabels([])

# Set labels and title
ax.set_xlabel('Objects')
ax.set_ylabel('Values')
ax.set_title('Emotions and Objects')

# Show plot
plt.tight_layout()
plt.show()
