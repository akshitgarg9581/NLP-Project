import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

SEED, EPOCHS, BATCH_SIZE, LR = 42, 5, 32, 3e-5
VISION_MODEL = "google/vit-base-patch16-224"
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./workspace_data"
DATA_DIR, RESULTS_DIR = os.path.join(BASE_DIR, "processed_data"), os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORKERS = 2 if os.name != "nt" else 0
ASPECTS = ["Person", "Location", "Org/Brand", "Product/Misc"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]


class ImageDataset(Dataset):
    def __init__(self, csv_path, tfm):
        self.df = pd.read_csv(csv_path)
        self.paths = self.df["image_path"].tolist()
        self.aspects = torch.tensor(self.df["aspect_id"].values, dtype=torch.long)
        self.sentiments = torch.tensor(self.df["sentiment_id"].values, dtype=torch.long)
        self.tfm = tfm

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))
        return self.tfm(img), self.aspects[i], self.sentiments[i]


class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.vit = AutoModel.from_pretrained(VISION_MODEL)
        dim = self.vit.config.hidden_size
        self.drop = nn.Dropout(0.3)
        self.aspect_head = nn.Linear(dim, 4)
        self.sentiment_head = nn.Linear(dim, 3)

    def forward(self, pixel_values):
        out = self.vit(pixel_values=pixel_values).last_hidden_state
        pooled = self.drop(0.5 * (out[:, 0] + out[:, 1:].mean(1)))
        return self.aspect_head(pooled), self.sentiment_head(pooled)


def get_weights(labels, n_cls):
    counts = np.bincount(labels, minlength=n_cls).astype(np.float32)
    w = np.sqrt(counts.sum() / (n_cls * np.maximum(counts, 1.0)))
    return torch.tensor(w / w.mean(), device=DEVICE)


def evaluate(model, loader):
    model.eval()
    a_true, a_pred, s_true, s_pred = [], [], [], []
    with torch.no_grad():
        for px, ay, sy in loader:
            al, sl = model(px.to(DEVICE))
            a_true.extend(ay.numpy()); a_pred.extend(al.argmax(-1).cpu().numpy())
            s_true.extend(sy.numpy()); s_pred.extend(sl.argmax(-1).cpu().numpy())

    a_true, a_pred, s_true, s_pred = map(np.array, (a_true, a_pred, s_true, s_pred))
    ap, ar, af1, _ = precision_recall_fscore_support(a_true, a_pred, average="macro", zero_division=0)
    sp, sr, sf1, _ = precision_recall_fscore_support(s_true, s_pred, average="macro", zero_division=0)
    j_true, j_pred = a_true * 3 + s_true, a_pred * 3 + s_pred
    _, _, jf1, _ = precision_recall_fscore_support(j_true, j_pred, average="macro", zero_division=0)

    metrics = {
        "model_name": "Image-Only (ViT)",
        "aspect_acc": float(accuracy_score(a_true, a_pred)), "aspect_precision": float(ap), "aspect_recall": float(ar), "aspect_macro_f1": float(af1),
        "sentiment_acc": float(accuracy_score(s_true, s_pred)), "sentiment_precision": float(sp), "sentiment_recall": float(sr), "sentiment_macro_f1": float(sf1),
        "joint_acc": float(accuracy_score(j_true, j_pred)), "joint_macro_f1": float(jf1),
    }
    return metrics, a_true, a_pred, s_true, s_pred


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
    train_tfm = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.85, 1.0)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(proc.image_mean, proc.image_std)])
    eval_tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(proc.image_mean, proc.image_std)])

    train_ds = ImageDataset(os.path.join(DATA_DIR, "train.csv"), train_tfm)
    val_ds = ImageDataset(os.path.join(DATA_DIR, "val.csv"), eval_tfm)
    test_ds = ImageDataset(os.path.join(DATA_DIR, "test.csv"), eval_tfm)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, num_workers=WORKERS)

    model = ImageOnlyModel().to(DEVICE)
    crit_a = nn.CrossEntropyLoss(weight=get_weights(train_ds.df["aspect_id"], 4))
    crit_s = nn.CrossEntropyLoss(weight=get_weights(train_ds.df["sentiment_id"], 3))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best_f1, ckpt = -1.0, os.path.join(RESULTS_DIR, "best_image_only.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for px, ay, sy in train_loader:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                al, sl = model(px.to(DEVICE))
                loss = crit_a(al, ay.to(DEVICE)) + crit_s(sl, sy.to(DEVICE))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            total_loss += loss.item()

        val_m, *_ = evaluate(model, val_loader)
        score = 0.5 * (val_m["aspect_macro_f1"] + val_m["sentiment_macro_f1"])
        print(f"Epoch {ep}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Val Aspect F1: {val_m['aspect_macro_f1']*100:.2f}% | Val Sent F1: {val_m['sentiment_macro_f1']*100:.2f}%")
        if score > best_f1:
            best_f1 = score
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    test_m, a_true, a_pred, s_true, s_pred = evaluate(model, test_loader)
    print("\nTEST RESULTS (Image-Only):", json.dumps({k: round(v * 100, 2) if isinstance(v, float) else v for k, v in test_m.items()}, indent=2))

    with open(os.path.join(RESULTS_DIR, "image_only_metrics.json"), "w") as f:
        json.dump(test_m, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(a_true, a_pred), display_labels=ASPECTS).plot(ax=axes[0], cmap="Oranges", colorbar=False)
    axes[0].set_title("Image-Only: Aspect")
    ConfusionMatrixDisplay(confusion_matrix(s_true, s_pred), display_labels=SENTIMENTS).plot(ax=axes[1], cmap="Oranges", colorbar=False)
    axes[1].set_title("Image-Only: Sentiment")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "image_only_cm.png"), dpi=150); plt.show()
