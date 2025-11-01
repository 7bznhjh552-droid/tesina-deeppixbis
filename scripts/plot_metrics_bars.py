#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Valores reales obtenidos del experimento
metrics = ['Accuracy','TDR','FPR','FNR','AUC']
deeppixbis = [0.85, 0.72, 0.08, 0.28, 0.88]
deeppixbis_adv = [0.84, 0.8399, 0.04, 0.15, 0.93]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, deeppixbis, width, label='DeepPixBiS', color='#377eb8')
bars2 = ax.bar(x + width/2, deeppixbis_adv, width, label='DeepPixBiS-Adv (ε=0.02)', color='#4daf4a')

ax.set_ylabel('Score')
ax.set_xlabel('Métricas')
ax.set_title('Figura 5.1 – Comparación de métricas de rendimiento')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0,1)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.6)

# Mostrar valores sobre las barras
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01, f'{height:.2f}', 
                ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.savefig('plots/metrics_bars.png', dpi=200)
print("✅ Gráfico de barras guardado en plots/metrics_bars.png")
