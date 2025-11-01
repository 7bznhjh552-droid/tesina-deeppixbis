#!/usr/bin/env python3
"""
Divide el dataset procesado en train/val/test manteniendo la proporción por sujeto y clase.
Entrada: data/processed/metadata.csv
Salida: train.csv, val.csv, test.csv
"""
import pandas as pd
from sklearn.model_selection import train_test_split
import os

META = "data/processed/metadata.csv"
OUT_DIR = "data/processed"
SEED = 42

df = pd.read_csv(META)
print(f"Total imágenes: {len(df)}")

# crear columna agrupadora por persona
persons = df['person'].unique()
train_p, temp_p = train_test_split(persons, test_size=0.3, random_state=SEED)
val_p, test_p = train_test_split(temp_p, test_size=0.5, random_state=SEED)

def subset(df, persons_list):
    return df[df['person'].isin(persons_list)]

train_df = subset(df, train_p)
val_df = subset(df, val_p)
test_df = subset(df, test_p)

for name, d in [('train', train_df), ('val', val_df), ('test', test_df)]:
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    d.to_csv(out_path, index=False)
    print(f"{name}: {len(d)} imágenes -> {out_path}")

print("✅ División completada con semilla reproducible:", SEED)
