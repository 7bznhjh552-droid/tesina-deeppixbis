#!/usr/bin/env python3
"""
Entrenamiento del modelo DeepPixBiS (implementación LivenessNet adaptada)
con y sin entrenamiento adversarial (FGSM).
Usa los splits generados (train.csv / val.csv).
"""
import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from attacks import fgsm_attack

class SpoofDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {'bonafide': 0, 'attack': 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.label_map[row['label']]

def train(args):
    # Selección de dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'
                          if getattr(torch.backends, 'mps', None)
                          and torch.backends.mps.is_available() else 'cpu')
    print("📟 Device:", device)

    # Importar modelo adaptado (3 canales RGB)
    from deeppixbis_adapter import LivenessNetRGB
    model = LivenessNetRGB().to(device)

    # Transformaciones
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # Datasets y dataloaders
    train_ds = SpoofDataset('data/processed/train.csv', transform=tf)
    val_ds = SpoofDataset('data/processed/val.csv', transform=tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Configuración de entrenamiento
    loss_fn = nn.BCELoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.float().to(device)

            # Entrenamiento adversarial opcional
            if args.adv:
                adv_imgs = fgsm_attack(model, loss_fn, imgs, labels, epsilon=args.epsilon)
                imgs = torch.cat([imgs, adv_imgs])
                labels = torch.cat([labels, labels])

            preds = model(imgs).squeeze()

            # Ajustar tamaño de salida (mapa 28x28 -> valor promedio por imagen)
            if preds.ndim == 3:
                preds = preds.mean(dim=[1, 2])

            loss = loss_fn(preds, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())

        # Guardar checkpoint
        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), f"models/deeppixbis_epoch{epoch+1}{'_adv' if args.adv else ''}.pt")

    print("✅ Entrenamiento completado.")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--adv', action='store_true', help='Usar entrenamiento adversarial FGSM')
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--epsilon', type=float, default=0.02)
    args = ap.parse_args()
    train(args)
