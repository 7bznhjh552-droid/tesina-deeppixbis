#!/usr/bin/env python3
"""
Evalúa los modelos DeepPixBiS estándar y adversarial sobre el set de test.
Calcula TDR, FPR, FNR, Accuracy y AUC.
"""
import torch, os
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix
import numpy as np
from tqdm import tqdm

class SpoofDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.label_map = {'bonafide': 0, 'attack': 1}
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row['path']).convert('RGB')
        if self.transform: img = self.transform(img)
        return img, self.label_map[row['label']]

def evaluate(model_path, device):
    from deeppixbis_adapter import LivenessNetRGB
    model = LivenessNetRGB().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor()])
    ds = SpoofDataset('data/processed/test.csv', transform=tf)
    dl = DataLoader(ds, batch_size=4)

    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in tqdm(dl, desc=f"Evaluando {os.path.basename(model_path)}"):
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze()
            if outputs.ndim == 3:
                outputs = outputs.mean(dim=[1,2])
            probs = outputs.detach().cpu().numpy()
            y_pred.extend(probs)
            y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_hat = (y_pred > 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_hat).ravel()
    acc = (tp + tn) / (tp + tn + fp + fn)
    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)
    tdr = tp / (tp + fn + 1e-8)
    auc = roc_auc_score(y_true, y_pred)

    return dict(model=model_path, acc=acc, fpr=fpr, fnr=fnr, tdr=tdr, auc=auc,
                tp=tp, tn=tn, fp=fp, fn=fn, total=len(y_true))

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps'
                          if getattr(torch.backends, 'mps', None)
                          and torch.backends.mps.is_available() else 'cpu')

    results = []
    for m in ["models/deeppixbis_epoch1.pt", "models/deeppixbis_epoch1_adv.pt"]:
        res = evaluate(m, device)
        results.append(res)

    print("\n📊 Resultados comparativos:")
    for r in results:
        print(f"\nModelo: {os.path.basename(r['model'])}")
        print(f"  Accuracy: {r['acc']:.3f}")
        print(f"  TDR:      {r['tdr']:.3f}")
        print(f"  FPR:      {r['fpr']:.3f}")
        print(f"  FNR:      {r['fnr']:.3f}")
        print(f"  AUC:      {r['auc']:.3f}")
