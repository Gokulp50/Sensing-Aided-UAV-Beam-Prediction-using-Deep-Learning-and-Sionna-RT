"""Evaluate a trained current-beam model under camera/sensor degradation.

This is a read-only evaluation of the existing held-out test split. It loads
the checkpoint and training sensor scaler, then saves a separate robustness
report beneath --output-dir.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from current_beam_benchmark import FEATURES, NUM_CLASSES, load_frame, normalize_sensors

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class GatedResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        image_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.visual_encoder = nn.Sequential(
            nn.Linear(image_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25)
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(len(FEATURES), 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.25)
        )
        self.gate = nn.Sequential(
            nn.Linear(512, 128), nn.GELU(), nn.Dropout(0.25), nn.Linear(128, 2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 256), nn.GELU(), nn.Dropout(0.25), nn.Linear(256, NUM_CLASSES)
        )

    def forward(self, images, sensors):
        image_features = self.visual_encoder(self.backbone(images))
        sensor_features = self.sensor_encoder(sensors)
        weights = torch.softmax(self.gate(torch.cat([image_features, sensor_features], dim=1)), dim=1)
        fused = weights[:, 0:1] * image_features + weights[:, 1:2] * sensor_features
        return self.classifier(fused), weights


class RobustnessDataset(Dataset):
    def __init__(self, rows, data_root, sensors, scaler, condition):
        self.rows = rows.reset_index(drop=True)
        self.data_root = data_root
        self.sensors = sensors.astype(np.float32, copy=True)
        self.scaler = scaler
        self.condition = condition
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)
        ])
        self.sensor_values = self._perturb_sensors()

    def _perturb_sensors(self):
        x = self.sensors
        if self.condition in ("gps_noise_5m", "gps_noise_15m"):
            rng = np.random.default_rng(20261102)
            sigma_m = 5.0 if self.condition == "gps_noise_5m" else 15.0
            lat = self.rows["latitude"].to_numpy(dtype=np.float64)
            # Convert metre-scale position error to latitude/longitude degrees.
            x[:, 0] += rng.normal(0, sigma_m / 111_320.0, len(x)) / self.scaler["std"][0]
            lon_m_per_degree = 111_320.0 * np.maximum(np.cos(np.deg2rad(lat)), 0.1)
            x[:, 1] += rng.normal(0, sigma_m / lon_m_per_degree, len(x)) / self.scaler["std"][1]
            return np.clip(x, -8, 8)
        if self.condition == "gps_missing":
            x[:, [0, 1]] = 0.0
        elif self.condition == "motion_missing":
            cols = [FEATURES.index(c) for c in ("speed", "x_speed", "y_speed", "z_speed", "pitch", "roll")]
            x[:, cols] = 0.0
        elif self.condition == "all_sensors_missing":
            x[:] = 0.0
        return x

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows.iloc[i]
        with Image.open(self.data_root / str(row["unit1_rgb"])) as image:
            image = image.convert("RGB")
            if self.condition in ("blur_r1_5", "blur_lowlight"):
                image = image.filter(ImageFilter.GaussianBlur(radius=1.5))
            if self.condition in ("lowlight_0_5", "blur_lowlight"):
                image = ImageEnhance.Brightness(image).enhance(0.5)
            image = self.transform(image)
        return image, torch.from_numpy(self.sensor_values[i]), int(row["unit1_beam_index"]) - 1


def evaluate(model, loader, device):
    model.eval()
    counts = np.zeros(3, dtype=np.int64)
    total = 0
    class_true = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_pred = np.zeros(NUM_CLASSES, dtype=np.int64)
    class_tp = np.zeros(NUM_CLASSES, dtype=np.int64)
    with torch.inference_mode():
        for images, sensors, labels in loader:
            images = images.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16,
                                enabled=(device.type == "cuda")):
                logits, _ = model(images, sensors)
            top = logits.float().topk(5, dim=1).indices
            total += len(labels)
            for i, k in enumerate((1, 3, 5)):
                counts[i] += int((top[:, :k] == labels[:, None]).any(1).sum().item())
            pred = top[:, 0]
            class_true += torch.bincount(labels, minlength=NUM_CLASSES).cpu().numpy()
            class_pred += torch.bincount(pred, minlength=NUM_CLASSES).cpu().numpy()
            class_tp += torch.bincount(labels[pred == labels], minlength=NUM_CLASSES).cpu().numpy()
    f1 = 2 * class_tp / np.maximum(class_true + class_pred, 1)
    present = class_true > 0
    return {"samples": total,
            "top1_accuracy": counts[0] / total,
            "top3_accuracy": counts[1] / total,
            "top5_accuracy": counts[2] / total,
            "macro_f1_present_classes": float(f1[present].mean()) if present.any() else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--scaler-json", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test",
                    help="Use validation while comparing candidate models; reserve test for final reporting.")
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    out = args.output_dir.resolve()
    protected = ["robustness_metrics.csv", "robustness_summary.json"]
    if any((out / name).exists() for name in protected):
        raise FileExistsError(f"Output folder already contains a robustness report: {out}")
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
    rows, data_root = load_frame(args.project_root.resolve())
    sensors, _ = normalize_sensors(rows)
    scaler = json.loads(args.scaler_json.read_text(encoding="utf-8"))
    eval_rows = rows.loc[rows["split"] == args.split].reset_index(drop=True)
    eval_sensors = sensors[rows["split"].to_numpy() == args.split]

    model = GatedResNet18().to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("architecture") != "gated_resnet18_sensor":
        raise ValueError(f"Unexpected checkpoint architecture: {checkpoint.get('architecture')}")
    model.load_state_dict(checkpoint["model"])

    conditions = ["clean", "gps_noise_5m", "gps_noise_15m", "gps_missing",
                  "motion_missing", "all_sensors_missing", "blur_r1_5",
                  "lowlight_0_5", "blur_lowlight"]
    records = []
    for condition in conditions:
        ds = RobustnessDataset(eval_rows, data_root, eval_sensors, scaler, condition)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
                            pin_memory=(device.type == "cuda"))
        row = {"condition": condition, **evaluate(model, loader, device)}
        records.append(row)
        print(f"{condition:20s} top1={row['top1_accuracy']:.4f} "
              f"top3={row['top3_accuracy']:.4f} top5={row['top5_accuracy']:.4f}", flush=True)

    clean = next(r for r in records if r["condition"] == "clean")
    for row in records:
        for metric in ("top1_accuracy", "top3_accuracy", "top5_accuracy"):
            row[f"{metric}_delta_pp_vs_clean"] = 100.0 * (row[metric] - clean[metric])
    pd.DataFrame(records).to_csv(out / "robustness_metrics.csv", index=False)
    summary = {
        "task": "robustness of current-beam camera/sensor fusion under controlled input degradation",
        "checkpoint": str(args.checkpoint.resolve()),
        "split": f"existing sequence-grouped {args.split} split",
        "samples": len(eval_rows),
        "conditions": conditions,
        "notes": [
            "GPS noise is independent Gaussian position error with the stated metre standard deviation.",
            "Missing standardized sensor values are set to zero, equivalent to training-mean imputation.",
            "Blur uses PIL Gaussian blur radius 1.5; low-light uses brightness factor 0.5.",
            "Degraded test inputs are diagnostic only; no retraining or threshold tuning is performed."
        ],
        "results": records,
    }
    (out / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved report under: {out}", flush=True)


if __name__ == "__main__":
    main()
