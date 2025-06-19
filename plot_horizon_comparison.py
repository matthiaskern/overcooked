import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("agent_comparison_results.csv")

# Check unique values in completed column
print("Unique values in completed column:", df['completed'].unique())

# Convert completion to numeric
df['completed_num'] = df['completed'].map({True: 1, False: 0, 'TRUE': 1, 'FALSE': 0})

print("After conversion:", df['completed_num'].unique())

# Calculate metrics
pivot_steps = df.pivot_table(
    index='horizon length', 
    columns='agent', 
    values='steps', 
    aggfunc='mean'
)

pivot_completion = df.pivot_table(
    index='horizon length', 
    columns='agent', 
    values='completed_num', 
    aggfunc='mean'
)

print("Steps pivot:")
print(pivot_steps)
print("\nCompletion pivot:")
print(pivot_completion)

# Create plot
fig, ax = plt.subplots(figsize=(10, 6))

# Plot bars
pivot_steps.plot(kind='bar', color=['#FF6B6B', '#4ECDC4'], ax=ax)

# Add completion rate as text on bars
for i, horizon in enumerate(pivot_steps.index):
    for j, agent in enumerate(pivot_steps.columns):
        steps = pivot_steps.loc[horizon, agent]
        completion_rate = pivot_completion.loc[horizon, agent]
        
        # Position text on bar
        x_pos = i + (j - 0.5) * 0.4
        ax.text(x_pos, steps + 1, f'{completion_rate:.1%}', 
               ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.xlabel('Horizon Length')
plt.ylabel('Number of Steps (max 50)')
plt.title('Steps by Agent Type at Horizon Length 3 vs 5 vs 9\n(Completion rates shown on bars)')
plt.xticks(rotation=0)
plt.legend(title='Agent Type')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('horizon_comparison_plot.png', dpi=300, bbox_inches='tight')
plt.show()
