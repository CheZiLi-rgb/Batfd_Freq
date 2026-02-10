import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, roc_curve
import argparse
import os
from dataset.fakeav import FakeAVDataModule
from model.batfd_freq_classier import Batfd_Classifier


def compute_eer(labels, probs):
    """计算等误差率 (Equal Error Rate)"""
    fpr, tpr, thresholds = roc_curve(labels, probs, pos_label=1)
    fnr = 1 - tpr
    # 找到 fnr 和 fpr 最接近的点
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer, eer_threshold


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Setting up DataModule from {args.data_root}...")
    dm = FakeAVDataModule(
        root=args.data_root,
        frame_padding=args.frame_padding,
        max_duration=args.max_duration,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        take_test=None
    )

    dm.setup(stage="test")
    test_loader = dm.test_dataloader()

    print(f"Test dataset size: {len(dm.test_dataset)}")

    print(f"Loading model from {args.checkpoint}...")
    model = Batfd_Classifier.load_from_checkpoint(args.checkpoint)
    model.to(device)
    model.eval()

    all_probs = []
    all_labels = []

    print("Starting inference...")
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):

            if len(batch) >= 3:
                video = batch[0]
                audio = batch[1]
                label = batch[2]
            else:
                continue

            video = video.to(device)
            audio = audio.to(device)
            label = label.to(device)

            logits, _, _ = model(video, audio)

            probs = torch.sigmoid(logits)

            if label.dim() > 1:
                label = label.view(-1)
            if probs.dim() > 1:
                probs = probs.view(-1)

            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    valid_indices = ~np.isnan(all_probs)
    all_probs = all_probs[valid_indices]
    all_labels = all_labels[valid_indices]

    if len(all_labels) == 0:
        print("Error: No valid predictions collected.")
        return

    print(f"\nCollected {len(all_labels)} valid predictions.")

    try:
        # AUC
        auc = roc_auc_score(all_labels, all_probs)
        # AP (Average Precision)
        ap = average_precision_score(all_labels, all_probs)
        # Accuracy (阈值 0.5)
        preds = (all_probs > 0.5).astype(int)
        acc = accuracy_score(all_labels, preds)
        # EER
        eer, eer_thresh = compute_eer(all_labels, all_probs)

        print("\n" + "=" * 40)
        print(f" Evaluation Results on {dm.root}")
        print("=" * 40)
        print(f" AUC      : {auc:.4f}")
        print(f" AP       : {ap:.4f}")
        print(f" Accuracy : {acc:.4f}")
        print(f" EER      : {eer:.4f} (Threshold: {eer_thresh:.4f})")
        print("=" * 40)

        with open("eval_results.txt", "w") as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"AUC: {auc:.4f}\n")
            f.write(f"AP: {ap:.4f}\n")
            f.write(f"Acc: {acc:.4f}\n")
            f.write(f"EER: {eer:.4f}\n")

    except ValueError as e:
        print(f"Error calculating metrics: {e}")
        print("Tips: Ensure your test set contains BOTH Real (0) and Fake (1) samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the .ckpt model file')
    parser.add_argument('--data_root', type=str, default="/data1/lic/FakeAVCelev_v1.2",
                        help='Root directory of the dataset')

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--frame_padding', type=int, default=512, help='Number of frames to pad/truncate')
    parser.add_argument('--max_duration', type=int, default=40)

    args = parser.parse_args()

    evaluate(args)