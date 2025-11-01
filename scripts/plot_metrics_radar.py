#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Métricas transformadas para que todas sean "mejor = más alto"
labels = np.array(['Accuracy','TDR','1-FPR','1-FNR','AUC'])
deeppixbis = np.array([0.85, 0.72, 1-0.08, 1-0.28, 0.88])
deeppixbis_adv = np.array([0.84, 0.8399, 1-0.04, 1-0.15, 0.93])

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
deeppixbis = np.concatenate((deeppixbis, [deeppixbis[0]]))
deeppixbis_adv = np.concatenate((deeppixbis_adv, [deeppixbis_adv[0]]))
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
ax.plot(angles, deeppixbis, 'o-', linewidth=2, label='DeepPixBiS', color='#377eb8')
ax.fill(angles, deeppixbis, alpha=0.25, color='#377eb8')
ax.plot(angles, deeppixbis_adv, 'o-', linewidth=2, label='DeepPixBiS-Adv (ε=0.02)', color='#4daf4a')
ax.fill(angles, deeppixbis_adv, alpha=0.25, color='#4daf4a')

ax.set_thetagrids(np.degrees(angles[:-1]), labels)
ax.set_ylim(0,1)
ax.legend(loc='upper right', bbox_to_anchor=(1.3,1.1))
plt.tight_layout()
plt.savefig('plots/metrics_radar.png', dpi=200)
print("✅ Gráfico radar guardado en plots/metrics_radar.png")
