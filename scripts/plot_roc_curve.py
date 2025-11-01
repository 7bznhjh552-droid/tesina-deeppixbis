#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np

# Datos provistos por René (simulación coherente)
fpr = np.array([0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60, 0.80, 1.00])
tpr_base = np.array([0.00, 0.25, 0.40, 0.55, 0.65, 0.70, 0.80, 0.88, 1.00])
tpr_adv  = np.array([0.00, 0.60, 0.78, 0.88, 0.92, 0.95, 0.97, 0.99, 1.00])

auc_base = np.trapz(tpr_base, fpr)
auc_adv  = np.trapz(tpr_adv, fpr)

plt.figure(figsize=(7,6))
plt.plot(fpr, tpr_base, color='#377eb8', lw=2.5, marker='o',
         label=f'DeepPixBiS (AUC ≈ {auc_base:.2f})')
plt.plot(fpr, tpr_adv, color='#4daf4a', lw=2.5, marker='o',
         label=f'DeepPixBiS-Adv (AUC ≈ {auc_adv:.2f})')
plt.plot([0,1],[0,1],'k--',lw=1,alpha=0.7)

plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.legend(loc="lower right")
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig('plots/roc_comparison.png', dpi=200)
print(f"✅ Curva ROC guardada en plots/roc_comparison.png")
print(f"AUC Base: {auc_base:.3f} | AUC Adversarial: {auc_adv:.3f}")
