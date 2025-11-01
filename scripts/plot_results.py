import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Si ya tienes results.csv, úsalo; si no, generamos una tabla de ejemplo (simulada)
csv_path = 'results.csv'
try:
    df = pd.read_csv(csv_path)
except FileNotFoundError:
    df = pd.DataFrame({
        'model':['DeepPixBiS','DeepPixBiS-Adv'],
        'acc':[0.75, 0.82],   # <- replace with real values
        'tdr':[0.72, 0.85],
        'fpr':[0.08, 0.04],
        'fnr':[0.28, 0.15],
        'auc':[0.88, 0.93]
    })
    print("Using simulated dataframe; replace with results.csv for real plot.")

# plot TDR / FPR side-by-side
labels = df['model']
x = np.arange(len(labels))
width = 0.25

fig, ax = plt.subplots(figsize=(8,5))
ax.bar(x - width, df['tdr'], width, label='TDR')
ax.bar(x, df['fpr'], width, label='FPR')
ax.bar(x + width, df['acc'], width, label='Accuracy')

ax.set_ylabel('Score')
ax.set_title('Comparación métricas (ejemplo)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.tight_layout()
plt.savefig('plots/metrics_comparison.png', dpi=200)
print("Saved plots/metrics_comparison.png")
