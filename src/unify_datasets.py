#!/usr/bin/env python3
"""
Unifica datasets MSU-MFSD y Kaggle Monitors-Replay
Genera:
  data/bonafide/<dataset>_<person>/
  data/attack/<dataset>_<person>/
y produce data/combined_manifest.csv con path,label,dataset,person
"""
import os, glob, csv, shutil

ROOT = os.getcwd()
OUT_B = os.path.join(ROOT, 'data', 'bonafide')
OUT_A = os.path.join(ROOT, 'data', 'attack')
os.makedirs(OUT_B, exist_ok=True)
os.makedirs(OUT_A, exist_ok=True)
rows = []

def copy_images(src_dir, out_dir, dataset_tag, label):
    if not os.path.isdir(src_dir):
        print("⚠️  Skipping missing folder:", src_dir)
        return
    for root, _, files in os.walk(src_dir):
        for f in files:
            if not f.lower().endswith(('.jpg','.png','.jpeg')): 
                continue
            src = os.path.join(root, f)
            person = os.path.splitext(f)[0][:12]
            dst_dir = os.path.join(out_dir, f"{dataset_tag}_{person}")
            os.makedirs(dst_dir, exist_ok=True)
            dst = os.path.join(dst_dir, f)
            shutil.copy2(src, dst)
            rows.append([dst, label, dataset_tag, f"{dataset_tag}_{person}"])

# 1️⃣ MSU-MFSD (reales + ataques)
msu_real = os.path.join(ROOT, 'downloads', 'MSU-MFSD', 'pics', 'real')
msu_attack = os.path.join(ROOT, 'downloads', 'MSU-MFSD', 'pics', 'attack')
copy_images(msu_real, OUT_B, 'msu', 'bonafide')
copy_images(msu_attack, OUT_A, 'msu', 'attack')

# 2️⃣ Kaggle Monitors Replay (solo ataques)
kaggle_attack = os.path.join(ROOT, 'data', 'attack')
copy_images(kaggle_attack, OUT_A, 'kaggle', 'attack')

# Guardar CSV combinado
manifest = os.path.join(ROOT, 'data', 'combined_manifest.csv')
with open(manifest, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['path','label','dataset','person'])
    writer.writerows(rows)

print(f"✅ Unificación completada. Total registros: {len(rows)}")
print(f"📄 Archivo generado: {manifest}")
