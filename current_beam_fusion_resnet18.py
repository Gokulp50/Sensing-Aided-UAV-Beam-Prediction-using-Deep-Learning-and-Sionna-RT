"""End-to-end current-beam prediction with camera + sensor fusion.

Uses an ImageNet-pretrained ResNet-18 fine-tuned on Scenario 23 images, an
11-feature sensor encoder, and a learned per-sample reliability gate. The
existing sequence-grouped split is enforced. Model selection uses validation
Top-1; all artifacts are written only under --output-dir.
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

from current_beam_benchmark import (
    FEATURES,
    add_power_metrics,
    load_frame,
    metric_row,
    normalize_sensors,
)


NUM_CLASSES = 64
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ImageSensorDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: Path, sensor_values: np.ndarray,
                 global_indices: np.ndarray, transform):
        self.rows = rows.reset_index(drop=True)
        self.data_root = data_root
        self.sensor_values = sensor_values[global_indices]
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        with Image.open(self.data_root / str(row["unit1_rgb"])) as image:
            image = self.transform(image.convert("RGB"))
        return (image, torch.from_numpy(self.sensor_values[i].copy()),
                int(row["unit1_beam_index"]) - 1, int(row["index"]))


class GatedResNet18(nn.Module):
    def __init__(self, dropout: float = 0.25):
        super().__init__()
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        image_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.visual_encoder = nn.Sequential(
            nn.Linear(image_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(len(FEATURES), 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(dropout), nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, images, sensors):
        image_features = self.visual_encoder(self.backbone(images))
        sensor_features = self.sensor_encoder(sensors)
        weights = torch.softmax(self.gate(torch.cat([image_features, sensor_features], dim=1)), dim=1)
        fused = weights[:, 0:1] * image_features + weights[:, 1:2] * sensor_features
        return self.classifier(fused), weights


def predict(model, loader, device):
    model.eval()
    probabilities, labels, indices, gates = [], [], [], []
    with torch.inference_mode():
        for images, sensors, y, idx in loader:
            images = images.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, gate = model(images, sensors)
            probabilities.append(torch.softmax(logits.float(), dim=1).cpu())
            labels.append(y.cpu())
            indices.append(idx.cpu())
            gates.append(gate.cpu())
    return torch.cat(probabilities), torch.cat(labels), torch.cat(indices), torch.cat(gates)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=53)
    ap.add_argument("--augmentation-profile", choices=("standard", "illumination_robust"),
                    default="standard",
                    help="Use stronger random brightness/contrast augmentation for the robust profile.")
    ap.add_argument("--skip-test", action="store_true",
                    help="Train/select by validation only; evaluate test after the candidate is frozen.")
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

    rows, data_root = load_frame(root)
    rows["previous_beam"] = rows.groupby("seq_index")["unit1_beam_index"].shift(1)
    sensor_values, scaler = normalize_sensors(rows)
    (out / "sensor_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")
    print("Grouped split rows:", rows.groupby("split").size().to_dict(), flush=True)
    print("Sequence counts:", rows.groupby("split")["seq_index"].nunique().to_dict(), flush=True)

    if args.augmentation_profile == "illumination_robust":
        # Training-only photometric variation includes substantial darkening,
        # matching the failure mode found in the robustness diagnostic.
        train_photometric = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.65, contrast=0.35,
                                   saturation=0.20, hue=0.02)
        ], p=0.80)
    else:
        train_photometric = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.12, contrast=0.12,
                                   saturation=0.08, hue=0.02)
        ], p=0.25)
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)), train_photometric,
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
    ])
    row_indices = np.arange(len(rows))
    loaders = {}
    for split in ("train", "val", "test"):
        mask = rows["split"].to_numpy() == split
        ds = ImageSensorDataset(rows.loc[mask], data_root, sensor_values, row_indices[mask],
                                train_transform if split == "train" else eval_transform)
        loaders[split] = DataLoader(ds, batch_size=args.batch_size, shuffle=(split == "train"),
                                    num_workers=0, pin_memory=(device.type == "cuda"))

    model = GatedResNet18().to(device)
    backbone_params = list(model.backbone.parameters())
    head_params = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
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
        for images, sensors, labels, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).long()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits, _ = model(images, sensors)
                loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.02)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach()) * len(labels)
            seen += len(labels)

        val_probs, val_y, _, _ = predict(model, loaders["val"], device)
        val_metric = metric_row(val_probs, val_y, "gated_multimodal", "val")
        scheduler.step(val_metric["top1_accuracy"])
        rec = {"epoch": epoch, "train_loss": loss_total / max(seen, 1),
               "val_top1_accuracy": val_metric["top1_accuracy"],
               "val_top3_accuracy": val_metric["top3_accuracy"],
               "val_top5_accuracy": val_metric["top5_accuracy"],
               "val_macro_f1_present_classes": val_metric["macro_f1_present_classes"],
               "backbone_lr": optimizer.param_groups[0]["lr"],
               "head_lr": optimizer.param_groups[1]["lr"]}
        history.append(rec)
        print(f"Epoch {epoch:02d}/{args.epochs} loss={rec['train_loss']:.4f} "
              f"val_top1={rec['val_top1_accuracy']:.4f} top3={rec['val_top3_accuracy']:.4f} "
              f"top5={rec['val_top5_accuracy']:.4f}", flush=True)
        if rec["val_top1_accuracy"] > best_top1 + 1e-6:
            best_top1, best_epoch, stale = rec["val_top1_accuracy"], epoch, 0
            torch.save({"model": model.state_dict(), "best_epoch": epoch,
                        "architecture": "gated_resnet18_sensor", "features": FEATURES}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping; best validation epoch {best_epoch}.", flush=True)
                break

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    if args.skip_test:
        summary = {
            "task": "current-beam classification from synchronized RGB and 11 sensor features at time t",
            "architecture": "ImageNet-pretrained ResNet-18 fine-tuned end-to-end + gated sensor encoder",
            "augmentation_profile": args.augmentation_profile,
            "split": "existing sequence-grouped split; no sequence crosses splits",
            "split_rows": {str(k): int(v) for k, v in rows.groupby("split").size().items()},
            "split_sequences": {str(k): int(v) for k, v in rows.groupby("split")["seq_index"].nunique().items()},
            "checkpoint_selection": "validation Top-1 only; test evaluation intentionally skipped",
            "best_epoch": best_epoch,
            "best_validation_top1": best_top1,
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        }
        (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved validation-selected checkpoint without test evaluation: {out}", flush=True)
        return
    test_probs, test_y, test_ids, test_gate = predict(model, loaders["test"], device)
    test_row = {"model": "fine_tuned_resnet18_gated_sensor_current_beam", "split": "test",
                **metric_row(test_probs, test_y, "fine_tuned_resnet18_gated_sensor_current_beam", "test")}
    top0 = test_probs.topk(5, dim=1).indices.numpy()
    test_rows = rows[rows.split == "test"].reset_index(drop=True)
    test_row = add_power_metrics(test_row, test_rows, data_root, top0)
    test_row["best_epoch"] = best_epoch
    test_row["training_seconds"] = time.perf_counter() - start
    test_row["mean_visual_gate"] = float(test_gate[:, 0].mean())
    test_row["mean_sensor_gate"] = float(test_gate[:, 1].mean())
    has_prev = test_rows["previous_beam"].notna().to_numpy()
    predicted = test_probs.argmax(1).numpy() + 1
    truth = test_y.numpy() + 1
    if has_prev.any():
        persistence = test_rows.loc[has_prev, "previous_beam"].to_numpy(dtype=np.int64)
        test_row["top1_accuracy_on_persistence_subset"] = float(np.mean(predicted[has_prev] == truth[has_prev]))
        test_row["persistence_top1_same_samples"] = float(np.mean(persistence == truth[has_prev]))
        test_row["persistence_comparable_n"] = int(has_prev.sum())

    top1 = predicted
    top5 = top0 + 1
    predictions = pd.DataFrame({
        "index": test_ids.numpy(), "seq_index": test_rows["seq_index"].to_numpy(dtype=np.int64),
        "true_beam": truth, "predicted_beam": top1,
        "top3_beams": [" ".join(map(str, r[:3])) for r in top5],
        "top5_beams": [" ".join(map(str, r[:5])) for r in top5],
        "previous_beam": test_rows["previous_beam"].to_numpy(),
        "visual_gate": test_gate[:, 0].numpy(), "sensor_gate": test_gate[:, 1].numpy(),
    })
    predictions.to_csv(out / "test_predictions.csv", index=False)
    pd.DataFrame([test_row]).to_csv(out / "metrics.csv", index=False)
    summary = {
        "task": "current-beam classification from synchronized RGB and 11 sensor features at time t",
        "architecture": "ImageNet-pretrained ResNet-18 fine-tuned end-to-end + gated sensor encoder",
        "augmentation_profile": args.augmentation_profile,
        "split": "existing sequence-grouped split; no sequence crosses splits",
        "split_rows": {str(k): int(v) for k, v in rows.groupby("split").size().items()},
        "split_sequences": {str(k): int(v) for k, v in rows.groupby("split")["seq_index"].nunique().items()},
        "checkpoint_selection": "validation Top-1 accuracy",
        "test_metrics": test_row,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TEST:", json.dumps(test_row, indent=2), flush=True)
    print(f"Saved multimodal current-beam run under: {out}", flush=True)


if __name__ == "__main__":
    main()
