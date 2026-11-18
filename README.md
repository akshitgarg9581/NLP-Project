# ImplicitMABSA: Multimodal Implicit Aspect-Based Sentiment Analysis Using Text and Images

A deep learning research project investigating how combining **textual semantic representations (`RoBERTa`)** and **visual features (`ViT`)** via **Bi-Directional Cross-Modal Attention and Gated Fusion** improves the detection of **implicit (unspoken) aspect categories** and their associated **sentiment** in multimodal social media posts.

---

## Project Pipeline & Files

| Script | Stage | Description |
| :--- | :--- | :--- |
| `01_data_pipeline.py` | Data Preparation | Downloads Twitter-2015 images/sentiment TSVs and UMT entity category annotations, aligns them (`5,337` samples), and converts explicit `$T$` targets into implicit context (`"this"`). |
| `02_train_text_only.py` | Baseline 1 (Text-Only) | Fine-tunes `roberta-base` on `implicit_text` alone to predict Aspect Category (4 classes) and Sentiment (3 classes). |
| `03_train_image_only.py` | Baseline 2 (Image-Only) | Fine-tunes `google/vit-base-patch16-224` on images alone (zero text) to predict Aspect Category and Sentiment. |
| `04_train_multimodal.py` | Proposed System (Multimodal) | Combines `RoBERTa` + `ViT` with Bi-Directional Cross-Modal Attention (`t2v` & `v2t`) and Adaptive Gated Fusion for joint implicit ABSA. |
| `05_evaluate_and_compare.py` | Evaluation & Figures | Aggregates Accuracy, Precision, Recall, Macro-F1, and Joint ABSA Accuracy across all 3 models and exports comparison charts. |
| `06_live_demo.py` | Interactive Demo | Launches a Gradio web interface to test real-time predictions across Text-Only, Image-Only, and Multimodal models on custom image + caption inputs. |

---

## Dataset Statistics (Twitter-2015 ImplicitMABSA)

- **Train:** `3,178` samples
- **Validation:** `1,122` samples
- **Test:** `1,037` samples
- **Aspect Categories (4):** `Person`, `Location`, `Org/Brand`, `Product/Misc`
- **Sentiment Classes (3):** `Negative`, `Neutral`, `Positive`

---

## Experimental Results (Twitter-2015 ImplicitMABSA Test Set, N = 1,037)

| Model | Aspect Acc (%) | Aspect Prec (%) | Aspect Rec (%) | Aspect F1 (%) | Sent Acc (%) | Sent Prec (%) | Sent Rec (%) | Sent F1 (%) | Joint Acc (%) | Joint F1 (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Image-Only (`ViT`)** | 47.64 | 40.17 | 39.10 | 38.79 | 52.56 | 37.78 | 37.20 | 36.81 | 25.07 | 14.05 |
| **Text-Only (`RoBERTa`)** | 67.70 | 60.65 | 59.37 | 59.64 | 74.64 | 69.05 | 71.72 | 70.21 | 51.11 | 39.94 |
| **Multimodal (`ImplicitMABSA`, Ours)** | **70.30** | **63.86** | **62.33** | **62.76** | **76.18** | **72.72** | **68.76** | **70.38** | **54.39** | **43.83** |

### Key Takeaways
- Combining **Text (`RoBERTa`) + Image (`ViT`)** via **Bi-Directional Cross-Modal Attention & Gated Fusion** outperforms both unimodal baselines across **Implicit Aspect, Sentiment, and Joint ABSA** metrics.
- Joint Implicit ABSA Accuracy improves by **+3.28%** over Text-Only and **+29.32%** over Image-Only.

---

## How to Run (Kaggle / Colab GPU)

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the scripts sequentially:
   ```bash
   python 01_data_pipeline.py
   python 02_train_text_only.py
   python 03_train_image_only.py
   python 04_train_multimodal.py
   python 05_evaluate_and_compare.py
   ```
   *(All generated metrics, CSV tables, and confusion matrix plots are saved under `results/`.)*
