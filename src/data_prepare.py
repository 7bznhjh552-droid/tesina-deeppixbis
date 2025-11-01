#!/usr/bin/env python3
"""
Crea data/processed con recortes de cara usando MTCNN (facenet-pytorch)
Espera estructura en data/ con subfolders por clase:
data/bonafide/<person>/img1.jpg
data/attack/<person>/img1.jpg
Genera data/processed/{bonafide,attack}/{person}/img_XXXXX.jpg
También genera data/processed/metadata.csv con columnas: path,label,person
"""
import os
from facenet_pytorch import MTCNN
from PIL import Image
import csv
from tqdm import tqdm

SRC = "data"
DST = "data/processed"
LABELS = ["bonafide", "attack"]

os.makedirs(DST, exist_ok=True)
mtcnn = MTCNN(keep_all=False, device='cpu')  # si torch MPS está disponible, cambiar device

meta_rows = []
for label in LABELS:
    src_label = os.path.join(SRC, label)
    if not os.path.isdir(src_label):
        continue
    for person in os.listdir(src_label):
        pdir = os.path.join(src_label, person)
        if not os.path.isdir(pdir):
            continue
        out_person_dir = os.path.join(DST, label, person)
        os.makedirs(out_person_dir, exist_ok=True)
        images = [f for f in os.listdir(pdir) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for i, fname in enumerate(tqdm(images, desc=f"{label}/{person}")):
            try:
                img_path = os.path.join(pdir, fname)
                img = Image.open(img_path).convert('RGB')
                face = mtcnn(img)
                if face is None:
                    # si no detecta rostro, opcional: guardar resized full image
                    face_img = img.resize((224,224))
                else:
                    # face es tensor CxHxW -> convertir a PIL
                    face_img = Image.fromarray((face.permute(1,2,0).int().numpy()).astype('uint8'), 'RGB') if hasattr(face, 'permute') else img.resize((224,224))
                out_path = os.path.join(out_person_dir, f"img_{i:05d}.jpg")
                face_img.save(out_path)
                meta_rows.append([out_path, label, person])
            except Exception as e:
                print("ERROR processing", img_path, e)

# guarda CSV
csv_path = os.path.join(DST, "metadata.csv")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["path","label","person"])
    writer.writerows(meta_rows)

print("Done. Processed images:", len(meta_rows))
