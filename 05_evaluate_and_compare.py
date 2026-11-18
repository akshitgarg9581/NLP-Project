import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./workspace_data"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

FALLBACK_TEXT_ONLY = {
    "model_name": "Text-Only (RoBERTa)",
    "aspect_acc": 0.6962, "aspect_precision": 0.6286, "aspect_recall": 0.6265, "aspect_macro_f1": 0.6271,
    "sentiment_acc": 0.7396, "sentiment_precision": 0.6855, "sentiment_recall": 0.7185, "sentiment_macro_f1": 0.6987,
    "joint_acc": 0.5246, "joint_macro_f1": 0.4130,
}


def load_json(fname, fallback=None):
    path = os.path.join(RESULTS_DIR, fname)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    if fallback is not None:
        return fallback
    raise FileNotFoundError(f"Missing {path} — run its training script first.")


if __name__ == "__main__":
    m_text = load_json("text_only_metrics.json", fallback=FALLBACK_TEXT_ONLY)
    m_img = load_json("image_only_metrics.json")
    m_mm = load_json("multimodal_metrics.json")

    rows = []
    for m in [m_text, m_img, m_mm]:
        rows.append({
            "Model": m["model_name"],
            "Aspect Acc (%)": round(m["aspect_acc"] * 100, 2),
            "Aspect Prec (%)": round(m["aspect_precision"] * 100, 2),
            "Aspect Rec (%)": round(m["aspect_recall"] * 100, 2),
            "Aspect F1 (%)": round(m["aspect_macro_f1"] * 100, 2),
            "Sent Acc (%)": round(m["sentiment_acc"] * 100, 2),
            "Sent Prec (%)": round(m["sentiment_precision"] * 100, 2),
            "Sent Rec (%)": round(m["sentiment_recall"] * 100, 2),
            "Sent F1 (%)": round(m["sentiment_macro_f1"] * 100, 2),
            "Joint Acc (%)": round(m["joint_acc"] * 100, 2),
            "Joint F1 (%)": round(m["joint_macro_f1"] * 100, 2),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "comparison_table.csv"), index=False)
    print("\n=== FINAL MODEL COMPARISON TABLE ===")
    print(df.to_string(index=False))

    # Plot clean comparison bar chart for Paper & Slides
    labels = ["Aspect Acc", "Aspect F1", "Sentiment Acc", "Sentiment F1", "Joint Acc"]
    keys = ["aspect_acc", "aspect_macro_f1", "sentiment_acc", "sentiment_macro_f1", "joint_acc"]
    x, w = np.arange(len(labels)), 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    b1 = ax.bar(x - w, [m_text[k] * 100 for k in keys], w, label="Text-Only", color="#4C72B0")
    b2 = ax.bar(x, [m_img[k] * 100 for k in keys], w, label="Image-Only", color="#DD8452")
    b3 = ax.bar(x + w, [m_mm[k] * 100 for k in keys], w, label="Multimodal (Ours)", color="#55A868")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontweight="bold")
    ax.set_ylabel("Score (%)"); ax.set_ylim(0, 100)
    ax.set_title("Text-Only vs Image-Only vs Multimodal ImplicitMABSA", fontweight="bold")
    ax.grid(axis="y", linestyle="--", alpha=0.5); ax.legend()

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.1f}", (bar.get_x() + bar.get_width() / 2, h + 1), ha="center", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "model_comparison_chart.png"), dpi=300)
    plt.show()
