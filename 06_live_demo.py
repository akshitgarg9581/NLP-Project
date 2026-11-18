import glob
import os
import subprocess
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel

try:
    import gradio as gr
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
    import gradio as gr

TEXT_MODEL, VISION_MODEL = "roberta-base", "google/vit-base-patch16-224"
BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./workspace_data"
DATA_DIR, RESULTS_DIR = os.path.join(BASE_DIR, "processed_data"), os.path.join(BASE_DIR, "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ASPECTS = ["Person", "Location", "Org/Brand", "Product/Misc"]
SENTIMENTS = ["Negative", "Neutral", "Positive"]


class TextOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(TEXT_MODEL)
        dim = self.roberta.config.hidden_size
        self.drop = nn.Dropout(0.3)
        self.aspect_head = nn.Linear(dim, 4)
        self.sentiment_head = nn.Linear(dim, 3)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()
        pooled = self.drop((out * mask).sum(1) / mask.sum(1).clamp(min=1e-9))
        return self.aspect_head(pooled), self.sentiment_head(pooled)


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


class MultimodalImplicitMABSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(TEXT_MODEL)
        self.vit = AutoModel.from_pretrained(VISION_MODEL)
        dim = 768
        self.t2v_attn = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1, batch_first=True)
        self.v2t_attn = nn.MultiheadAttention(dim, num_heads=8, dropout=0.1, batch_first=True)
        self.norm_t = nn.LayerNorm(dim)
        self.norm_v = nn.LayerNorm(dim)
        self.gate = nn.Sequential(nn.Linear(dim * 2, dim), nn.Sigmoid())
        self.fusion = nn.Sequential(nn.Linear(dim * 3, dim), nn.GELU(), nn.Dropout(0.3), nn.LayerNorm(dim))
        self.aspect_head = nn.Linear(dim, 4)
        self.sentiment_head = nn.Linear(dim, 3)

    def forward(self, input_ids, attention_mask, pixel_values):
        t_seq = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        v_seq = self.vit(pixel_values=pixel_values).last_hidden_state
        t_cross, _ = self.t2v_attn(t_seq, v_seq, v_seq)
        v_cross, _ = self.v2t_attn(v_seq, t_seq, t_seq, key_padding_mask=(attention_mask == 0))
        t_seq = self.norm_t(t_seq + t_cross)
        v_seq = self.norm_v(v_seq + v_cross)
        mask = attention_mask.unsqueeze(-1).float()
        z_t = (t_seq * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        z_v = 0.5 * (v_seq[:, 0] + v_seq[:, 1:].mean(1))
        g = self.gate(torch.cat([z_t, z_v], dim=-1))
        fused = (g * z_t + (1.0 - g) * z_v) + self.fusion(torch.cat([z_t, z_v, z_t * z_v], dim=-1))
        return self.aspect_head(fused), self.sentiment_head(fused)


def load_ckpt(cls, filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        matches = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
        if not matches:
            return None
        path = matches[0]
    model = cls().to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model


tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL)
proc = AutoImageProcessor.from_pretrained(VISION_MODEL)
img_tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(proc.image_mean, proc.image_std)])

m_text = load_ckpt(TextOnlyModel, "best_text_only.pt")
m_img = load_ckpt(ImageOnlyModel, "best_image_only.pt")
m_mm = load_ckpt(MultimodalImplicitMABSA, "best_multimodal.pt")


def format_out(al, sl):
    ap, sp = F.softmax(al, dim=-1)[0], F.softmax(sl, dim=-1)[0]
    ai, si = ap.argmax().item(), sp.argmax().item()
    return f"Aspect: {ASPECTS[ai]} ({ap[ai]*100:.1f}%)  |  Sentiment: {SENTIMENTS[si]} ({sp[si]*100:.1f}%)"


def predict(image, implicit_text):
    if image is None or not str(implicit_text).strip():
        return "Please provide both an image and implicit text.", "", ""
    enc = tokenizer(str(implicit_text), truncation=True, padding="max_length", max_length=64, return_tensors="pt")
    ids, mask = enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)
    px = img_tfm(image.convert("RGB")).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        r_t = format_out(*m_text(ids, mask)) if m_text else "Checkpoint missing (run 02_train_text_only.py)"
        r_i = format_out(*m_img(px)) if m_img else "Checkpoint missing (run 03_train_image_only.py)"
        r_m = format_out(*m_mm(ids, mask, px)) if m_mm else "Checkpoint missing (run 04_train_multimodal.py)"
    return r_t, r_i, r_m


sample_examples = []
test_csv = os.path.join(DATA_DIR, "test.csv")
if os.path.exists(test_csv):
    tdf = pd.read_csv(test_csv)
    for aid in range(4):
        sub = tdf[tdf["aspect_id"] == aid]
        if len(sub) > 0 and os.path.exists(sub.iloc[0]["image_path"]):
            sample_examples.append([sub.iloc[0]["image_path"], sub.iloc[0]["implicit_text"]])

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Post Image"),
        gr.Textbox(label="Implicit Caption (aspect hidden)", placeholder="e.g., Not what I expected from this!"),
    ],
    outputs=[
        gr.Textbox(label="Model 1: Text-Only (RoBERTa)"),
        gr.Textbox(label="Model 2: Image-Only (ViT)"),
        gr.Textbox(label="Model 3: Multimodal ImplicitMABSA (Ours)"),
    ],
    examples=sample_examples if sample_examples else None,
    title="ImplicitMABSA: Multimodal Implicit Aspect & Sentiment Demo",
    description="Upload an image and enter an implicit caption to compare Text-Only, Image-Only, and Multimodal predictions.",
)

if __name__ == "__main__":
    demo.launch(share=True, inline=True)
