"""Fine-tune an image classifier for current-beam prediction on Scenario 23.

This follows the paper's current-time image-classification task more closely
than the frozen-feature benchmark, while retaining our stricter sequence-level
train/validation/test split. All written artifacts stay under --output-dir.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


NUM_CLASSES = 64
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_rows(project_root: Path) -> tuple[pd.DataFrame, Path]:
    data_root = project_root / "data" / "scenario23_dev"
    scenario = pd.read_csv(data_root / "scenario23.csv")
    eda = pd.read_csv(project_root / "results" / "eda_sensor_data.csv")
    need_s = ["index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_beam_index"]
    need_e = ["index", "seq_index", "split"]
    missing = [f"scenario23.csv:{c}" for c in need_s if c not in scenario]
    missing += [f"eda_sensor_data.csv:{c}" for c in need_e if c not in eda]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    rows = scenario[need_s].merge(eda[need_e], on="index", how="inner", validate="one_to_one")
    rows = rows.sort_values(["seq_index", "index"]).reset_index(drop=True)
    if set(rows["split"].dropna().unique()) != {"train", "val", "test"}:
        raise ValueError("Expected train, val, and test samples.")
    if rows["split"].isna().any():
        raise ValueError("Some rows have no split assignment.")
    cross_split = rows.groupby("seq_index")["split"].nunique()
    if (cross_split != 1).any():
        raise ValueError("A sequence appears in more than one split; refusing a leaky run.")
    beam = rows["unit1_beam_index"].to_numpy(dtype=np.float64)
    if not np.isfinite(beam).all() or beam.min() < 1 or beam.max() > NUM_CLASSES:
        raise ValueError("Expected beam labels in 1..64.")
    rows["previous_beam"] = rows.groupby("seq_index")["unit1_beam_index"].shift(1)
    return rows, data_root


class ImageBeamDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: Path, transform):
        self.rows = rows.reset_index(drop=True)
        self.data_root = data_root
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        image_path = self.data_root / str(row["unit1_rgb"])
        with Image.open(image_path) as image:
            image = self.transform(image.convert("RGB"))
        return image, int(row["unit1_beam_index"]) - 1, int(row["index"])


def metrics(probs: torch.Tensor, labels: torch.Tensor) -> dict:
    probs = probs.detach().cpu()
    labels = labels.detach().cpu().long()
    top = probs.topk(5, dim=1).indices
    pred = top[:, 0]
    row = {"n": int(len(labels))}
    for k in (1, 3, 5):
        row[f"top{k}_accuracy"] = float((top[:, :k] == labels[:, None]).any(1).float().mean())
    support = torch.bincount(labels, minlength=NUM_CLASSES).float()
    predicted = torch.bincount(pred, minlength=NUM_CLASSES).float()
    tp = torch.bincount(labels[pred == labels], minlength=NUM_CLASSES).float()
    f1 = 2 * tp / (support + predicted).clamp_min(1)
    present = support > 0
    row["macro_f1_present_classes"] = float(f1[present].mean())
    row["classes_present"] = int(present.sum())
    return row


def make_probs(model, loader, device):
    model.eval()
    prob_rows, label_rows, index_rows = [], [], []
    with torch.inference_mode():
        for images, labels, indices in loader:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
            else:
                logits = model(images)
            prob_rows.append(torch.softmax(logits.float(), dim=1).cpu())
            label_rows.append(labels)
            index_rows.append(indices)
    return torch.cat(prob_rows), torch.cat(label_rows), torch.cat(index_rows)


def power_regrets(rows: pd.DataFrame, data_root: Path, top_beams: np.ndarray) -> dict:
    values = {1: [], 3: [], 5: []}
    for rel, candidates in zip(rows["unit1_pwr_60ghz"].astype(str), top_beams):
        rel = rel.replace("\\", "/").removeprefix("./")
        power = np.atleast_1d(np.loadtxt(data_root / rel, dtype=np.float64))
        if len(power) < NUM_CLASSES or not np.isfinite(power[:NUM_CLASSES]).all():
            continue
        power = np.maximum(power[:NUM_CLASSES], 1e-15)
        best_db = 10 * np.log10(float(power.max()))
        for k in values:
            selected = float(power[candidates[:k]].max())
            values[k].append(best_db - 10 * np.log10(max(selected, 1e-15)))
    return {f"top{k}_power_regret_db": float(np.mean(v)) if v else float("nan")
            for k, v in values.items()} | {"samples_with_power": len(values[1])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=47)
    args = ap.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = ["best_model.pt", "metrics.csv", "test_predictions.csv", "run_summary.json"]
    if any((out / name).exists() for name in protected):
        raise FileExistsError(f"Output folder already contains a run: {out}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)

    rows, data_root = load_rows(root)
    print("Grouped split rows:", rows.groupby("split").size().to_dict(), flush=True)
    print("Sequence counts:", rows.groupby("split")["seq_index"].nunique().to_dict(), flush=True)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.12, contrast=0.12,
                                                       saturation=0.08, hue=0.02)], p=0.25),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    datasets = {
        "train": ImageBeamDataset(rows[rows.split == "train"], data_root, train_transform),
        "val": ImageBeamDataset(rows[rows.split == "val"], data_root, eval_transform),
        "test": ImageBeamDataset(rows[rows.split == "test"], data_root, eval_transform),
    }
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda")),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda")),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda")),
    }

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.to(device)
    backbone_params = [p for name, p in model.named_parameters() if not name.startswith("fc.")]
    head_params = list(model.fc.parameters())
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr": args.backbone_lr},
        {"params": head_params, "lr": args.head_lr},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    best_top1, best_epoch, stale = -1.0, 0, 0
    history = []
    checkpoint_path = out / "best_model.pt"
    start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_total, seen = 0.0, 0
        for images, labels, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits = model(images)
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.02)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(labels)
            seen += len(labels)

        val_probs, val_y, _ = make_probs(model, loaders["val"], device)
        val_row = metrics(val_probs, val_y)
        scheduler.step(val_row["top1_accuracy"])
        rec = {"epoch": epoch, "train_loss": loss_total / max(seen, 1),
               "val_top1_accuracy": val_row["top1_accuracy"],
               "val_top3_accuracy": val_row["top3_accuracy"],
               "val_top5_accuracy": val_row["top5_accuracy"],
               "val_macro_f1_present_classes": val_row["macro_f1_present_classes"],
               "backbone_lr": optimizer.param_groups[0]["lr"],
               "head_lr": optimizer.param_groups[1]["lr"]}
        history.append(rec)
        print(f"Epoch {epoch:02d}/{args.epochs} loss={rec['train_loss']:.4f} "
              f"val_top1={rec['val_top1_accuracy']:.4f} top3={rec['val_top3_accuracy']:.4f} "
              f"top5={rec['val_top5_accuracy']:.4f}", flush=True)
        if rec["val_top1_accuracy"] > best_top1 + 1e-6:
            best_top1, best_epoch, stale = rec["val_top1_accuracy"], epoch, 0
            torch.save({"model": model.state_dict(), "best_epoch": epoch,
                        "num_classes": NUM_CLASSES, "architecture": "torchvision_resnet18"}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping; best validation epoch {best_epoch}.", flush=True)
                break

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    test_probs, test_y, test_ids = make_probs(model, loaders["test"], device)
    test_row = {"model": "fine_tuned_resnet18_current_beam", "split": "test", **metrics(test_probs, test_y)}
    test_top0 = test_probs.topk(5, dim=1).indices.numpy()
    test_rows = rows[rows.split == "test"].reset_index(drop=True)
    test_row.update(power_regrets(test_rows, data_root, test_top0))
    test_row["best_epoch"] = best_epoch
    test_row["training_seconds"] = time.perf_counter() - start

    has_prev = test_rows["previous_beam"].notna().to_numpy()
    pred1 = test_probs.argmax(1).numpy() + 1
    test_true = test_y.numpy() + 1
    if has_prev.any():
        test_row["top1_accuracy_on_persistence_subset"] = float(np.mean(pred1[has_prev] == test_true[has_prev]))
        persistence = test_rows.loc[has_prev, "previous_beam"].to_numpy(dtype=np.int64)
        test_row["persistence_top1_same_samples"] = float(np.mean(persistence == test_true[has_prev]))
        test_row["persistence_comparable_n"] = int(has_prev.sum())

    top5 = test_top0 + 1
    test_predictions = pd.DataFrame({
        "index": test_ids.numpy(),
        "seq_index": test_rows["seq_index"].to_numpy(dtype=np.int64),
        "true_beam": test_true,
        "predicted_beam": pred1,
        "top3_beams": [" ".join(map(str, row[:3])) for row in top5],
        "top5_beams": [" ".join(map(str, row[:5])) for row in top5],
        "previous_beam": test_rows["previous_beam"].to_numpy(),
    })
    test_predictions.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame([test_row]).to_csv(out / "metrics.csv", index=False)
    summary = {
        "task": "current-beam classification from one synchronized RGB image at time t",
        "architecture": "ImageNet-pretrained ResNet-18 fine-tuned end-to-end",
        "split": "existing sequence-grouped split; no sequence crosses splits",
        "split_rows": {str(k): int(v) for k, v in rows.groupby("split").size().items()},
        "split_sequences": {str(k): int(v) for k, v in rows.groupby("split")["seq_index"].nunique().items()},
        "checkpoint_selection": "validation Top-1 accuracy",
        "test_metrics": test_row,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TEST:", json.dumps(test_row, indent=2), flush=True)
    print(f"Saved fine-tuning run under: {out}", flush=True)


if __name__ == "__main__":
    main()
