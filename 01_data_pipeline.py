import os
import re
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

BASE_DIR = "/kaggle/working" if os.path.exists("/kaggle/working") else "./workspace_data"
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
OUT_DIR = os.path.join(BASE_DIR, "processed_data")
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

ASPECT_MAP = {"PER": 0, "LOC": 1, "ORG": 2, "OTHER": 3}
ASPECT_NAMES = {0: "Person", 1: "Location", 2: "Org/Brand", 3: "Product/Misc"}
SENTIMENT_NAMES = {0: "Negative", 1: "Neutral", 2: "Positive"}


def download_data():
    mabsa_dir = os.path.join(RAW_DIR, "TwitterDataMABSA")
    umt_dir = os.path.join(RAW_DIR, "UMT")
    if not os.path.exists(mabsa_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/CopotronicRifat/TwitterDataMABSA.git", mabsa_dir], check=True)
    if not os.path.exists(umt_dir):
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/jefferyYu/UMT.git", umt_dir], check=True)
    return mabsa_dir, umt_dir


def build_aspect_lookup(umt_dir):
    # Maps image_id -> [(entity_text, category)] from UMT BIO tags
    lookup = {}
    data_dir = os.path.join(umt_dir, "data", "twitter2015")
    for fname in ["train.txt", "valid.txt", "test.txt"]:
        img_name, tokens, cat = None, [], None
        with open(os.path.join(data_dir, fname), encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("IMGID:"):
                    if img_name and tokens:
                        lookup.setdefault(img_name, []).append((" ".join(tokens), cat))
                    img_name = line.split("IMGID:")[1].strip() + ".jpg"
                    tokens, cat = [], None
                elif not line:
                    if img_name and tokens:
                        lookup.setdefault(img_name, []).append((" ".join(tokens), cat))
                        tokens, cat = [], None
                else:
                    parts = line.split("\t")
                    if len(parts) >= 2:
                        tok, tag = parts[0], parts[1]
                        if tag.startswith("B-"):
                            if tokens:
                                lookup.setdefault(img_name, []).append((" ".join(tokens), cat))
                            tokens, cat = [tok], tag[2:]
                        elif tag.startswith("I-") and tokens:
                            tokens.append(tok)
                        elif tokens:
                            lookup.setdefault(img_name, []).append((" ".join(tokens), cat))
                            tokens, cat = [], None
        if img_name and tokens:
            lookup.setdefault(img_name, []).append((" ".join(tokens), cat))
    return lookup


def norm(s):
    return re.sub(r"\s+", "", s.strip().lower())


def clean_text(s):
    return re.sub(r"\s+", " ", re.sub(r"http\S+|www\.\S+", "", s)).strip()


def process_split(tsv_path, img_dir, lookup):
    rows = []
    with open(tsv_path, encoding="utf-8", errors="ignore") as f:
        lines = f.read().splitlines()[1:]

    for line in lines:
        cols = line.split("\t")
        if len(cols) < 5:
            continue
        sent_id, img_name, template, target = int(cols[1]), cols[2].strip(), cols[3].strip(), cols[4].strip()
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            continue

        matched_cat = None
        nt = norm(target)
        for ent_text, ent_cat in lookup.get(img_name, []):
            ne = norm(ent_text)
            if ne == nt or nt in ne or ne in nt:
                matched_cat = ent_cat
                break
        if matched_cat not in ASPECT_MAP:
            continue

        asp_id = ASPECT_MAP[matched_cat]
        rows.append({
            "image_name": img_name,
            "image_path": img_path,
            "target_word": target,
            "explicit_text": clean_text(template.replace("$T$", target)),
            "implicit_text": clean_text(template.replace("$T$", "this")),
            "aspect_id": asp_id,
            "aspect_name": ASPECT_NAMES[asp_id],
            "sentiment_id": sent_id,
            "sentiment_name": SENTIMENT_NAMES[sent_id],
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    mabsa_dir, umt_dir = download_data()
    lookup = build_aspect_lookup(umt_dir)
    tsv_dir = os.path.join(mabsa_dir, "twitter2015")
    img_dir = os.path.join(mabsa_dir, "twitter2015_images")

    train_df = process_split(os.path.join(tsv_dir, "train.tsv"), img_dir, lookup)
    val_df = process_split(os.path.join(tsv_dir, "dev.tsv"), img_dir, lookup)
    test_df = process_split(os.path.join(tsv_dir, "test.tsv"), img_dir, lookup)

    train_df.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print(f"Saved splits -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    print("\nAspect counts:\n", train_df["aspect_name"].value_counts())
    print("\nSentiment counts:\n", train_df["sentiment_name"].value_counts())
