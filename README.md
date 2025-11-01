# Tesina - Fortalecimiento de Sistemas de Reconocimiento Facial mediante Entrenamiento Adversarial (DeepPixBiS)

[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-Academic-lightgrey.svg)](./LICENSE)
[![GitHub Repo](https://img.shields.io/badge/GitHub-tesina--deeppixbis-black.svg?logo=github)](https://github.com/7bznhjh552-droid/tesina-deeppixbis)

---

Este repositorio contiene los **scripts, modelos y resultados experimentales** utilizados en la tesina de:

> **René Antonio León Cofré – Universidad Técnica Federico Santa María (2025)**  
> *Fortalecimiento de sistemas de reconocimiento facial contra la suplantación de identidad mediante entrenamiento adversarial*

---

## 📋 Descripción general

El objetivo de este trabajo es evaluar el impacto del **entrenamiento adversarial (FGSM)** en el modelo **DeepPixBiS** para la detección de ataques de presentación (*Presentation Attack Detection – PAD*).  
Se comparan dos versiones del mismo modelo: una entrenada **sin adversarial** y otra **con adversarial training**, utilizando datasets públicos combinados (MSU-MFSD y Monitors-Replay).

---

## 🧩 Estructura del proyecto

```text
tesina-deeppixbis/
├── data/
│   ├── bonafide/
│   ├── attack/
│   ├── processed/
│   └── raw_kaggle/
├── downloads/
│   └── MSU-MFSD/
├── models/
├── src/
│   ├── attacks.py
│   ├── train_deeppixbis.py
│   ├── evaluate_deeppixbis.py
│   ├── unify_datasets.py
│   └── data_prepare.py
├── scripts/
│   ├── plot_metrics_bars.py
│   ├── plot_metrics_radar.py
│   └── plot_roc_curve.py
├── plots/
├── logs/
├── requirements.txt
└── README.md
```

---

## ⚙️ Preparación del entorno

```bash
pyenv install 3.10.12
pyenv virtualenv 3.10.12 tesis-deeppixbis-3.10
pyenv activate tesis-deeppixbis-3.10
pip install -r requirements.txt
```

**Verificar soporte MPS (Mac Apple Silicon):**

```bash
python -c "import torch; print(torch.__version__, torch.backends.mps.is_available())"
```

---

## 📥 Descarga de datasets

### 🔹 MSU-MFSD

Repositorio oficial:  
https://github.com/sunny3/MSU-MFSD

```bash
mkdir -p downloads
cd downloads
git clone https://github.com/sunny3/MSU-MFSD.git
cd ..
```

### 🔹 Monitors-Replay (Kaggle)

Página del dataset:  
https://www.kaggle.com/datasets/tapakah68/monitors-replay-attacks-dataset

```bash
pip install kaggle
mkdir -p data/raw_kaggle
cd data/raw_kaggle
kaggle datasets download -d tapakah68/monitors-replay-attacks-dataset
unzip monitors-replay-attacks-dataset.zip -d monitors-replay
cd ../..
```

---

## 🧠 Unificación y normalización

```bash
python src/extract_kaggle_attacks.py
python src/unify_datasets.py
python src/data_prepare.py
```

**Resultado esperado:** `data/processed/metadata.csv`

---

## 🧪 Entrenamiento y evaluación

**Entrenamiento estándar**
```bash
python src/train_deeppixbis.py --epochs 5 --batch-size 8
```

**Entrenamiento adversarial (FGSM)**
```bash
python src/train_deeppixbis.py --adv --epochs 5 --batch-size 8 --epsilon 0.02
```

**Evaluación**
```bash
python src/evaluate_deeppixbis.py
```

---

## 📊 Visualización de resultados

```bash
python scripts/plot_metrics_bars.py
python scripts/plot_metrics_radar.py
python scripts/plot_roc_curve.py
```

**Figuras generadas:**

- `plots/metrics_bars.png`
- `plots/metrics_radar.png`
- `plots/roc_comparison.png`

---

## 🧮 Resumen de métricas (esperadas)

| Modelo                   | Accuracy | TDR  | FPR  | FNR  | AUC  |
|--------------------------|:--------:|:----:|:----:|:----:|:----:|
| DeepPixBiS (base)        | 0.85     | 0.72 | 0.08 | 0.28 | 0.88 |
| DeepPixBiS-Adv (ε=0.02)  | 0.84     | 0.84 | 0.04 | 0.15 | 0.93 |

> *Nota:* Los valores son de referencia para validar la tubería de entrenamiento/evaluación y pueden variar según semillas aleatorias y particiones.

---

## ⚠️ Archivos grandes

Por límite de 100 MB de GitHub, los datasets no se incluyen directamente.

| Dataset         | Fuente             | Tamaño aprox. | Enlace                                                                 |
|-----------------|--------------------|---------------|------------------------------------------------------------------------|
| MSU-MFSD        | GitHub (Sunny3)    | ~200 MB       | https://github.com/sunny3/MSU-MFSD                                     |
| Monitors-Replay | Kaggle (Tapakah68) | ~600 MB       | https://www.kaggle.com/datasets/tapakah68/monitors-replay-attacks-dataset |

---

## 📚 Licencia y citación

León Cofré, R. A. (2025). *Fortalecimiento de sistemas de reconocimiento facial contra la suplantación de identidad mediante entrenamiento adversarial*. Universidad Técnica Federico Santa María.

Repositorio oficial:  
🔗 https://github.com/7bznhjh552-droid/tesina-deeppixbis

**Tag:** `v1.0-tesina-2025`

---

## 🧱 Créditos

- **Autor:** René Antonio León Cofré  
- **Año:** 2025  
- **Institución:** Universidad Técnica Federico Santa María

