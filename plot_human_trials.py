import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("human_trials.csv")

df.columns = df.columns.str.strip()

df['Completed_num'] = df['Completed'].map({'Yes': 1, 'No': 0})

df_filtered = df[df['Horizon Length'] != 9]

pivot_steps = df_filtered.pivot_table(
    index='Horizon Length', 
    values='Steps', 
    aggfunc='mean'
)

pivot_completion = df_filtered.pivot_table(
    index='Horizon Length', 
    values='Completed_num', 
    aggfunc='mean'
)

pivot_rating = df_filtered.pivot_table(
    index='Horizon Length', 
    values='Rating', 
    aggfunc='mean'
)

print("Steps by horizon:")
print(pivot_steps)
print("\nCompletion rates by horizon:")
print(pivot_completion)
print("\nRating by horizon:")
print(pivot_rating)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

pivot_steps.plot(kind='bar', color='#4ECDC4', ax=ax1)
for i, horizon in enumerate(pivot_steps.index):
    steps = pivot_steps.loc[horizon, 'Steps']
    completion_rate = pivot_completion.loc[horizon, 'Completed_num']
    
    ax1.text(i, steps + 1, f'{completion_rate:.1%}', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xlabel('Horizon Length')
ax1.set_ylabel('Number of Steps')
ax1.set_title('Human Trial Steps by Horizon Length\n(Completion rates shown on bars)')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=0)
ax1.grid(True, alpha=0.3)

pivot_rating.plot(kind='bar', color='#FF6B6B', ax=ax2)
for i, horizon in enumerate(pivot_rating.index):
    rating = pivot_rating.loc[horizon, 'Rating']
    
    ax2.text(i, rating + 0.1, f'{rating:.1f}', 
           ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xlabel('Horizon Length')
ax2.set_ylabel('Rating (1-5)')
ax2.set_title('Human Trial Ratings by Horizon Length')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 5.5)

plt.tight_layout()
plt.savefig('human_trials_plot.png', dpi=300, bbox_inches='tight')
plt.show()