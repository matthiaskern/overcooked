import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('human_trials.csv')

# Filter for horizon lengths 3 and 5 only
df_filtered = df[df['Horizon Length'].isin([3, 5])]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

# Get unique participants
participants = sorted(df_filtered['Participant'].unique())

# Set up positions for bars
x = np.arange(len(participants))
width = 0.35

# Get data for each horizon length
horizon_3_steps = []
horizon_5_steps = []

for participant in participants:
    h3_data = df_filtered[(df_filtered['Participant'] == participant) & (df_filtered['Horizon Length'] == 3)]
    h5_data = df_filtered[(df_filtered['Participant'] == participant) & (df_filtered['Horizon Length'] == 5)]
    
    horizon_3_steps.append(h3_data['Steps'].values[0] if len(h3_data) > 0 else 0)
    horizon_5_steps.append(h5_data['Steps'].values[0] if len(h5_data) > 0 else 0)

# Create bars
bars1 = ax.bar(x - width/2, horizon_3_steps, width, label='3', color='blue')
bars2 = ax.bar(x + width/2, horizon_5_steps, width, label='5', color='green')

# Customize the plot
ax.set_xlabel('Participant')
ax.set_ylabel('Number of Steps')
ax.set_title('Steps at Horizon Length 3 vs. 5')
ax.set_xticks(x)
ax.set_xticklabels(participants)
ax.legend(title='Horizon Length')
ax.grid(True, alpha=0.3)

# Set y-axis limits to match the image (0 to 40)
ax.set_ylim(0, 40)

# Add some padding to the plot
plt.tight_layout()

# Save the plot
plt.savefig('horizon_comparison.png', dpi=300, bbox_inches='tight')
plt.show()