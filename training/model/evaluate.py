"""
evaluate.py

Usage:
    python evaluate.py --data_dir <crops_dir> --label_csv <labels.csv> --dataset_name <name>

Arguments:
    --data_dir       Directory of face crop images
    --label_csv      CSV with columns: filename, label (0=real, 1=fake)
                     Optionally include a 'compression' column (c0/c23/c40) for FF++
    --dataset_name   Name tag for reporting (e.g. ff_plus_plus, celeb_df_v2, dfdc)
    --ff_auc         (Optional) FF++ AUC to compare against for overfitting check
"""

import os
import sys
import argparse
import csv
import numpy as np
import onnxruntime as ort
import cv2
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
import config


def _get_best_providers() -> list:
    """
    Automatically select the best available ONNX Runtime execution providers.
    Priority: CUDA (NVIDIA GPU) > DirectML (Windows GPU) > Azure > CPU
    """
    available = ort.get_available_providers()
    print(f"[INFO] Available ONNX providers: {available}")

    priority = [
        "CUDAExecutionProvider",
        "DirectMLExecutionProvider",
        "AzureExecutionProvider",
        "CPUExecutionProvider",
    ]

    selected = [p for p in priority if p in available]
    print(f"[INFO] Selected providers: {selected}")
    return selected


# -- Preprocessing --------------------------------------------------------------


def preprocess(img_path: str) -> np.ndarray:
    image = cv2.imread(img_path)
    if image is None:
        return None
    image = cv2.resize(
        image, (config.INPUT_SIZE, config.INPUT_SIZE), interpolation=cv2.INTER_CUBIC
    )
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32) / 255.0
    mean = np.array(config.INPUT_MEAN, dtype=np.float32)
    std = np.array(config.INPUT_STD, dtype=np.float32)
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
    image = np.expand_dims(image, axis=0)  # (1, 3, 224, 224)
    return image.astype(np.float32)


# -- Bootstrap -----------------------------------------------------------------


def bootstrap_auc(labels, probs, n_iter=1000, ci=0.95):
    aucs = []
    n = len(labels)
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        sampled_labels = np.array(labels)[idx]
        sampled_probs = np.array(probs)[idx]
        if len(set(sampled_labels)) < 2:
            continue
        aucs.append(roc_auc_score(sampled_labels, sampled_probs))
    lower = np.percentile(aucs, (1 - ci) / 2 * 100)
    upper = np.percentile(aucs, (1 + ci) / 2 * 100)
    return np.mean(aucs), lower, upper


# -- Evaluation ----------------------------------------------------------------


def evaluate(data_dir, label_csv, dataset_name, ff_auc=None):
    # Load ONNX model
    session = ort.InferenceSession(
        config.ONNX_MODEL_PATH,
        providers=_get_best_providers(),
    )
    input_name = session.get_inputs()[0].name

    # Load labels
    samples = []
    with open(label_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            compression = row.get("compression", "unknown")
            samples.append((row["filename"], int(row["label"]), compression))

    print(f"\n[INFO] Evaluating on: {dataset_name} ({len(samples)} samples)")

    all_labels = []
    all_probs = []
    all_compression = []
    video_scores = defaultdict(list)
    video_labels = {}

    for filename, label, compression in samples:
        img_path = os.path.join(data_dir, filename)
        tensor = preprocess(img_path)
        if tensor is None:
            continue

        logits = session.run(None, {input_name: tensor})[0]
        probs = softmax(logits[0])
        p_fake = float(probs[1])

        all_labels.append(label)
        all_probs.append(p_fake)
        all_compression.append(compression)

        # Group by video ID (filename prefix before last underscore + frame index)
        video_id = "_".join(filename.split("_")[:-1])
        video_scores[video_id].append(p_fake)
        video_labels[video_id] = label

    # -- Frame-level metrics ---------------------------------------------------
    preds = [1 if p >= config.CLASSIFICATION_THRESHOLD else 0 for p in all_probs]
    auc, auc_low, auc_high = bootstrap_auc(
        all_labels, all_probs, config.BOOTSTRAP_ITERATIONS
    )
    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, zero_division=0)
    precision = precision_score(all_labels, preds, zero_division=0)
    recall = recall_score(all_labels, preds, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(all_labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    _ = fn + tp  # kept for readability parity with confusion matrix unpacking

    print(f"\n  -- Frame-Level Metrics ({dataset_name}) --")
    print(f"  AUC-ROC : {auc:.4f} (95% CI: {auc_low:.4f} - {auc_high:.4f})")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print(f"  FPR     : {fpr:.4f}")
    print(f"  Precision: {precision:.4f} | Recall: {recall:.4f}")

    # -- Compression-level stratification (FF++ only) --------------------------
    if dataset_name == "ff_plus_plus":
        for level in ["c0", "c23", "c40"]:
            idx = [i for i, c in enumerate(all_compression) if c == level]
            if not idx:
                continue
            lvl_labels = [all_labels[i] for i in idx]
            lvl_probs = [all_probs[i] for i in idx]
            if len(set(lvl_labels)) < 2:
                continue
            lvl_auc = roc_auc_score(lvl_labels, lvl_probs)
            lvl_preds = [1 if p >= config.CLASSIFICATION_THRESHOLD else 0 for p in lvl_probs]
            lvl_acc = accuracy_score(lvl_labels, lvl_preds)
            print(
                f"  [{level}] AUC: {lvl_auc:.4f} | Accuracy: {lvl_acc:.4f} | Samples: {len(idx)}"
            )

    # -- Video-level metrics ---------------------------------------------------
    vid_labels = list(video_labels.values())
    vid_probs = [np.mean(video_scores[vid]) for vid in video_labels]
    vid_preds = [1 if p >= config.CLASSIFICATION_THRESHOLD else 0 for p in vid_probs]
    vid_acc = accuracy_score(vid_labels, vid_preds)
    vid_auc = roc_auc_score(vid_labels, vid_probs) if len(set(vid_labels)) > 1 else 0.0
    print(f"\n  -- Video-Level Metrics ({dataset_name}) --")
    print(f"  Video AUC     : {vid_auc:.4f}")
    print(f"  Video Accuracy: {vid_acc:.4f}")

    # -- Domain overfitting check ----------------------------------------------
    if ff_auc is not None:
        drop = ff_auc - auc
        status = (
            "PASS"
            if drop <= config.OVERFITTING_AUC_DROP_LIMIT
            else "FAIL -- DOMAIN OVERFITTING DETECTED"
        )
        print("\n  -- Overfitting Check --")
        print(
            f"  FF++ AUC: {ff_auc:.4f} | {dataset_name} AUC: {auc:.4f} | Drop: {drop:.4f} -> {status}"
        )

    return auc


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--label_csv", required=True)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--ff_auc", type=float, default=None)
    args = parser.parse_args()

    evaluate(args.data_dir, args.label_csv, args.dataset_name, args.ff_auc)
