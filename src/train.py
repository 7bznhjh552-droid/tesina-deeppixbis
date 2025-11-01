#!/usr/bin/env python3
import argparse
import os
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm
from attacks import fgsm_attack

class SpoofDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {'bonafide':0, 'attack':1}
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        label = self.label_map[row['label']]
        return img, label

def build_simple_cnn(num_classes=2):
    # modelo pequeño para pruebas
    return nn.Sequential(
        nn.Conv2d(3,16,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(32*56*56, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu')
    print("Using device:", device)
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    csv_path = os.path.join('data','processed','metadata.csv')
    ds = SpoofDataset(csv_path, transform=tf)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,num_workers=2)
    model = build_simple_cnn().to(device)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)
            if args.adv:
                # generar adversario FGSM y entrenar con ambos
                adv_imgs = fgsm_attack(model, loss_fn, imgs, labels, epsilon=args.epsilon)
                inputs = torch.cat([imgs, adv_imgs], dim=0)
                labs = torch.cat([labels, labels], dim=0)
            else:
                inputs = imgs
                labs = labels

            outputs = model(inputs)
            loss = loss_fn(outputs, labs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=loss.item())
        # guardar checkpoint por epoch
        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f"models/model_epoch_{epoch+1}{'_adv' if args.adv else ''}.pt")
    print("Training finished.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--adv', action='store_true', help='Entrenar con adversarial (FGSM) por batch')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--epsilon', type=float, default=0.02)
    args = parser.parse_args()
    train(args)
