import os
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, ConfusionMatrixDisplay

SEED, EPOCHS, BATCH_SIZE, LR = 42, 5, 32, 2e-5
MODEL_NAME = "roberta-base"
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./workspace_data"
DATA_DIR, RESULTS_DIR = os.path.join(BASE_DIR, "processed_data"), os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ASPECTS = ["Person", "Location", "Org/Brand", "Product/Misc"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]


class TextDataset(Dataset):
    def __init__(self, csv_path, tokenizer):
        self.df = pd.read_csv(csv_path)
        self.enc = tokenizer(self.df["implicit_text"].astype(str).tolist(), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
        self.aspects = torch.tensor(self.df["aspect_id"].values, dtype=torch.long)
        self.sentiments = torch.tensor(self.df["sentiment_id"].values, dtype=torch.long)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        return self.enc["input_ids"][i], self.enc["attention_mask"][i], self.aspects[i], self.sentiments[i]


class TextOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(MODEL_NAME)
        dim = self.roberta.config.hidden_size
        self.drop = nn.Dropout(0.3)
        self.aspect_head = nn.Linear(dim, 4)
        self.sentiment_head = nn.Linear(dim, 3)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = self.drop((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
        return self.aspect_head(pooled), self.sentiment_head(pooled)


def get_weights(labels, n_cls):
    counts = np.bincount(labels, minlength=n_cls).astype(np.float32)
    w = np.sqrt(counts.sum() / (n_cls * np.maximum(counts, 1.0)))
    return torch.tensor(w / w.mean(), device=DEVICE)


def evaluate(model, loader):
    model.eval()
    a_true, a_pred, s_true, s_pred = [], [], [], []
    with torch.no_grad():
        for ids, mask, ay, sy in loader:
            al, sl = model(ids.to(DEVICE), mask.to(DEVICE))
            a_true.extend(ay.numpy()); a_pred.extend(al.argmax(-1).cpu().numpy())
            s_true.extend(sy.numpy()); s_pred.extend(sl.argmax(-1).cpu().numpy())

    a_true, a_pred, s_true, s_pred = map(np.array, (a_true, a_pred, s_true, s_pred))
    ap, ar, af1, _ = precision_recall_fscore_support(a_true, a_pred, average="macro", zero_division=0)
    sp, sr, sf1, _ = precision_recall_fscore_support(s_true, s_pred, average="macro", zero_division=0)
    j_true, j_pred = a_true * 3 + s_true, a_pred * 3 + s_pred
    _, _, jf1, _ = precision_recall_fscore_support(j_true, j_pred, average="macro", zero_division=0)

    metrics = {
        "model_name": "Text-Only (RoBERTa)",
        "aspect_acc": float(accuracy_score(a_true, a_pred)), "aspect_precision": float(ap), "aspect_recall": float(ar), "aspect_macro_f1": float(af1),
        "sentiment_acc": float(accuracy_score(s_true, s_pred)), "sentiment_precision": float(sp), "sentiment_recall": float(sr), "sentiment_macro_f1": float(sf1),
        "joint_acc": float(accuracy_score(j_true, j_pred)), "joint_macro_f1": float(jf1),
    }
    return metrics, a_true, a_pred, s_true, s_pred


if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = TextDataset(os.path.join(DATA_DIR, "train.csv"), tokenizer)
    val_ds = TextDataset(os.path.join(DATA_DIR, "val.csv"), tokenizer)
    test_ds = TextDataset(os.path.join(DATA_DIR, "test.csv"), tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = TextOnlyModel().to(DEVICE)
    crit_a = nn.CrossEntropyLoss(weight=get_weights(train_ds.df["aspect_id"], 4))
    crit_s = nn.CrossEntropyLoss(weight=get_weights(train_ds.df["sentiment_id"], 3))

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * len(train_loader) * EPOCHS), len(train_loader) * EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
    best_f1, ckpt = -1.0, os.path.join(RESULTS_DIR, "best_text_only.pt")

    for ep in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for ids, mask, ay, sy in train_loader:
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=(DEVICE.type == "cuda")):
                al, sl = model(ids.to(DEVICE), mask.to(DEVICE))
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
    print("\nTEST RESULTS (Text-Only):", json.dumps({k: round(v * 100, 2) if isinstance(v, float) else v for k, v in test_m.items()}, indent=2))

    with open(os.path.join(RESULTS_DIR, "text_only_metrics.json"), "w") as f:
        json.dump(test_m, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    ConfusionMatrixDisplay(confusion_matrix(a_true, a_pred), display_labels=ASPECTS).plot(ax=axes[0], cmap="Blues", colorbar=False)
    axes[0].set_title("Text-Only: Aspect")
    ConfusionMatrixDisplay(confusion_matrix(s_true, s_pred), display_labels=SENTIMENTS).plot(ax=axes[1], cmap="Blues", colorbar=False)
    axes[1].set_title("Text-Only: Sentiment")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS_DIR, "text_only_cm.png"), dpi=150); plt.show()
