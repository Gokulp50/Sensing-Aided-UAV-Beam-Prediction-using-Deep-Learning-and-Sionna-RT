"""Grouped current-beam benchmark for DeepSense 6G Scenario 23.

This is a separate task from next-beam forecasting. It predicts the recorded
beam at time t from the camera embedding and sensors synchronized at time t.
It compares visual-only, sensor-only, and learned gated fusion, plus the
previous-beam persistence baseline where a prior sample exists in-sequence.

All run artifacts are written beneath --output-dir. Existing project files and
the supplied feature cache are read-only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


FEATURES = [
    "latitude", "longitude", "speed", "altitude", "distance", "height",
    "x_speed", "y_speed", "z_speed", "pitch", "roll",
]
NUM_CLASSES = 64
IMAGE_DIM = 2048
MODES = ("visual_only", "sensor_only", "gated_multimodal")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_frame(project_root: Path) -> tuple[pd.DataFrame, Path]:
    data_root = project_root / "data" / "scenario23_dev"
    scenario_path = data_root / "scenario23.csv"
    eda_path = project_root / "results" / "eda_sensor_data.csv"
    scenario = pd.read_csv(scenario_path)
    eda = pd.read_csv(eda_path)
    s_cols = ["index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_beam_index"]
    e_cols = ["index", "seq_index", "split", *FEATURES]
    missing_s = [c for c in s_cols if c not in scenario]
    missing_e = [c for c in e_cols if c not in eda]
    if missing_s or missing_e:
        raise ValueError(f"Missing columns: scenario={missing_s}, sensor/split={missing_e}")
    frame = scenario[s_cols].merge(
        eda[e_cols], on="index", how="inner", validate="one_to_one"
    ).sort_values(["seq_index", "index"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("The scenario/sensor join produced no samples.")
    if frame["split"].isna().any() or set(frame["split"].unique()) != {"train", "val", "test"}:
        raise ValueError("Expected non-empty train, val, and test split labels.")
    seq_split_counts = frame.groupby("seq_index")["split"].nunique()
    if (seq_split_counts != 1).any():
        bad = seq_split_counts[seq_split_counts != 1].index.tolist()
        raise ValueError(f"Sequence leakage detected; sequences cross splits: {bad[:10]}")
    beam = frame["unit1_beam_index"].to_numpy(dtype=np.float64)
    if not np.isfinite(beam).all() or beam.min() < 1 or beam.max() > NUM_CLASSES:
        raise ValueError("Beam labels must be finite integers in 1..64.")
    if not np.allclose(beam, np.round(beam)):
        raise ValueError("Beam labels contain non-integer values.")
    return frame, data_root


def load_features(frame: pd.DataFrame, cache_dir: Path) -> np.ndarray:
    feats_path = cache_dir / "resnet50_frame_features.npy"
    ids_path = cache_dir / "resnet50_frame_indices.npy"
    if not feats_path.exists() or not ids_path.exists():
        raise FileNotFoundError(
            "Expected cached ResNet-50 frame features and indices in "
            f"{cache_dir}. Run the existing future model once to create the cache."
        )
    ids = np.load(ids_path, allow_pickle=False)
    feats = np.load(feats_path, mmap_mode="r", allow_pickle=False)
    if feats.ndim != 2 or feats.shape[1] != IMAGE_DIM or len(feats) != len(ids):
        raise ValueError(f"Unexpected feature cache shape: {feats.shape}; expected [N,{IMAGE_DIM}].")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Cached feature indices are not unique.")
    cache_lookup = {int(v): i for i, v in enumerate(ids)}
    frame_ids = frame["index"].to_numpy(dtype=np.int64)
    missing = [int(v) for v in frame_ids if int(v) not in cache_lookup]
    if missing:
        raise ValueError(f"Feature cache is missing {len(missing)} scenario rows; first IDs: {missing[:5]}")
    return np.asarray(feats[[cache_lookup[int(v)] for v in frame_ids]], dtype=np.float32)


def normalize_sensors(frame: pd.DataFrame) -> tuple[np.ndarray, dict]:
    train = frame.loc[frame["split"] == "train", FEATURES].to_numpy(dtype=np.float64)
    median = np.nanmedian(train, axis=0)
    median[~np.isfinite(median)] = 0.0
    train = np.where(np.isfinite(train), train, median)
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std < 1e-8] = 1.0
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, median)
    scaled = np.clip((values - mean) / std, -8, 8).astype(np.float32)
    stats = {"features": FEATURES, "mean": mean.tolist(), "std": std.tolist(),
             "median_imputation": median.tolist()}
    return scaled, stats


class CurrentBeamNet(nn.Module):
    def __init__(self, mode: str, dropout: float = 0.25):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"Unknown modality mode: {mode}")
        self.mode = mode
        self.visual_encoder = nn.Sequential(
            nn.Linear(IMAGE_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout)
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

    def forward(self, visual: torch.Tensor, sensors: torch.Tensor):
        v = self.visual_encoder(visual)
        s = self.sensor_encoder(sensors)
        if self.mode == "visual_only":
            weights = visual.new_tensor([1.0, 0.0]).expand(visual.size(0), 2)
        elif self.mode == "sensor_only":
            weights = visual.new_tensor([0.0, 1.0]).expand(visual.size(0), 2)
        else:
            weights = torch.softmax(self.gate(torch.cat([v, s], dim=1)), dim=1)
        fused = weights[:, 0:1] * v + weights[:, 1:2] * s
        return self.classifier(fused), weights


def metric_row(probs: torch.Tensor, y: torch.Tensor, mode: str, split: str) -> dict:
    probs = probs.detach().cpu()
    y = y.detach().cpu().long()
    top = probs.topk(5, dim=1).indices
    pred = top[:, 0]
    result = {"model": mode, "split": split, "n": int(len(y))}
    for k in (1, 3, 5):
        result[f"top{k}_accuracy"] = float((top[:, :k] == y[:, None]).any(1).float().mean())
    actual = torch.bincount(y, minlength=NUM_CLASSES).float()
    predicted = torch.bincount(pred, minlength=NUM_CLASSES).float()
    tp = torch.bincount(y[pred == y], minlength=NUM_CLASSES).float()
    f1 = (2 * tp / (actual + predicted).clamp_min(1))
    present = actual > 0
    result["macro_f1_present_classes"] = float(f1[present].mean())
    result["classes_present"] = int(present.sum())
    return result


def read_power(data_root: Path, rel_path: str) -> np.ndarray:
    rel = str(rel_path).replace("\\", "/").removeprefix("./")
    return np.atleast_1d(np.loadtxt(data_root / rel, dtype=np.float64))


def add_power_metrics(row: dict, frame_rows: pd.DataFrame, data_root: Path,
                      topk: np.ndarray) -> dict:
    regrets = {1: [], 3: [], 5: []}
    for rel, beams in zip(frame_rows["unit1_pwr_60ghz"].astype(str), topk):
        power = read_power(data_root, rel)
        if not np.isfinite(power).all() or len(power) < NUM_CLASSES:
            continue
        power = np.maximum(power[:NUM_CLASSES], 1e-15)
        best_db = 10 * np.log10(float(power.max()))
        for k in regrets:
            chosen_power = float(power[beams[:k]].max())
            regrets[k].append(best_db - 10 * np.log10(max(chosen_power, 1e-15)))
    for k, vals in regrets.items():
        row[f"top{k}_power_regret_db"] = float(np.mean(vals)) if vals else float("nan")
    row["samples_with_power"] = len(regrets[1])
    return row


def make_loader(visual: np.ndarray, sensors: np.ndarray, y: np.ndarray,
                mask: np.ndarray, batch_size: int, shuffle: bool, device: torch.device):
    ds = TensorDataset(torch.from_numpy(visual[mask]), torch.from_numpy(sensors[mask]),
                       torch.from_numpy(y[mask].astype(np.int64)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=(device.type == "cuda"))


def predict(model, loader, device):
    model.eval()
    probs, ys, gates = [], [], []
    with torch.inference_mode():
        for v, s, y in loader:
            logits, weights = model(v.to(device, non_blocking=True), s.to(device, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1).cpu())
            ys.append(y.cpu())
            gates.append(weights.cpu())
    return torch.cat(probs), torch.cat(ys), torch.cat(gates)


def train_mode(mode: str, frame: pd.DataFrame, visual: np.ndarray, sensors: np.ndarray,
               args, device: torch.device, out: Path) -> tuple[dict, dict, pd.DataFrame, list[dict]]:
    split = frame["split"].to_numpy()
    train_mask, val_mask, test_mask = split == "train", split == "val", split == "test"
    y = frame["unit1_beam_index"].to_numpy(dtype=np.int64) - 1
    train_loader = make_loader(visual, sensors, y, train_mask, args.batch_size, True, device)
    val_loader = make_loader(visual, sensors, y, val_mask, args.batch_size, False, device)
    test_loader = make_loader(visual, sensors, y, test_mask, args.batch_size, False, device)

    model = CurrentBeamNet(mode).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    best_val, best_epoch, stale = -1.0, 0, 0
    history = []
    checkpoint_path = out / f"best_model_{mode}.pt"
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, seen = 0.0, 0
        for v, s, target in train_loader:
            v = v.to(device, non_blocking=True)
            s = s.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(v, s)
            loss = F.cross_entropy(logits, target, label_smoothing=0.02)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(target)
            seen += len(target)

        val_probs, val_y, _ = predict(model, val_loader, device)
        val_metric = metric_row(val_probs, val_y, mode, "val")
        scheduler.step(val_metric["top1_accuracy"])
        rec = {"model": mode, "epoch": epoch, "train_loss": total_loss / max(seen, 1),
               "val_top1_accuracy": val_metric["top1_accuracy"],
               "val_top3_accuracy": val_metric["top3_accuracy"],
               "val_top5_accuracy": val_metric["top5_accuracy"],
               "val_macro_f1_present_classes": val_metric["macro_f1_present_classes"],
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(rec)
        print(f"{mode} epoch {epoch:02d}/{args.epochs} loss={rec['train_loss']:.4f} "
              f"val_top1={rec['val_top1_accuracy']:.4f} "
              f"top3={rec['val_top3_accuracy']:.4f} top5={rec['val_top5_accuracy']:.4f}", flush=True)
        if rec["val_top1_accuracy"] > best_val + 1e-6:
            best_val, best_epoch, stale = rec["val_top1_accuracy"], epoch, 0
            torch.save({"model": model.state_dict(), "mode": mode, "epoch": epoch,
                        "features": FEATURES, "num_classes": NUM_CLASSES}, checkpoint_path)
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping {mode}: best validation epoch {best_epoch}.", flush=True)
                break

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model"])
    val_probs, val_y, val_gate = predict(model, val_loader, device)
    test_probs, test_y, test_gate = predict(model, test_loader, device)
    val_row = metric_row(val_probs, val_y, mode, "val")
    test_row = metric_row(test_probs, test_y, mode, "test")
    test_top = test_probs.topk(5, dim=1).indices.numpy()
    test_frame = frame.loc[test_mask].reset_index(drop=True)
    test_row = add_power_metrics(test_row, test_frame, args.data_root, test_top)
    has_previous = test_frame["previous_beam"].notna().to_numpy()
    if has_previous.any():
        same_rows = np.flatnonzero(has_previous)
        test_row["top1_accuracy_on_persistence_subset"] = float(
            np.mean(test_probs.argmax(1).numpy()[same_rows] ==
                    (test_frame.loc[has_previous, "unit1_beam_index"].to_numpy(dtype=np.int64) - 1))
        )
    else:
        test_row["top1_accuracy_on_persistence_subset"] = float("nan")
    test_row["persistence_comparable_n"] = int(has_previous.sum())
    test_row["best_epoch"] = best_epoch
    test_row["training_seconds"] = time.perf_counter() - start_time
    test_row["mean_visual_gate"] = float(test_gate[:, 0].mean())
    test_row["mean_sensor_gate"] = float(test_gate[:, 1].mean())

    val_idx = np.flatnonzero(val_mask)
    test_idx = np.flatnonzero(test_mask)
    top5_val = val_probs.topk(5, dim=1).indices.numpy() + 1
    top5_test = test_top + 1
    detail_rows = []
    for split_name, inds, probabilities, top5, gates in (
        ("val", val_idx, val_probs, top5_val, val_gate),
        ("test", test_idx, test_probs, top5_test, test_gate),
    ):
        top1 = probabilities.argmax(1).numpy() + 1
        for local_i, global_i in enumerate(inds):
            r = frame.iloc[global_i]
            detail_rows.append({
                "model": mode, "split": split_name, "index": int(r["index"]),
                "seq_index": int(r["seq_index"]), "true_beam": int(y[global_i] + 1),
                "predicted_beam": int(top1[local_i]),
                "top3_beams": " ".join(map(str, top5[local_i, :3])),
                "top5_beams": " ".join(map(str, top5[local_i, :5])),
                "visual_gate": float(gates[local_i, 0]),
                "sensor_gate": float(gates[local_i, 1]),
            })
    return val_row, test_row, pd.DataFrame(detail_rows), history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--feature-cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--learning-rate", type=float, default=7e-4)
    ap.add_argument("--seed", type=int, default=43)
    args = ap.parse_args()
    args.project_root = args.project_root.resolve()
    args.feature_cache_dir = args.feature_cache_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.data_root = args.project_root / "data" / "scenario23_dev"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected = ["metrics.csv", "validation_metrics.csv", "predictions_by_split.csv", "run_summary.json"]
    if any((args.output_dir / name).exists() for name in expected):
        raise FileExistsError(f"Output directory already has benchmark artifacts: {args.output_dir}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)

    frame, data_root = load_frame(args.project_root)
    frame["previous_beam"] = frame.groupby("seq_index")["unit1_beam_index"].shift(1)
    print("Grouped split rows:", frame.groupby("split").size().to_dict(), flush=True)
    print("Sequence counts:", frame.groupby("split")["seq_index"].nunique().to_dict(), flush=True)
    visual = load_features(frame, args.feature_cache_dir)
    sensors, scaler = normalize_sensors(frame)
    (args.output_dir / "sensor_scaler.json").write_text(json.dumps(scaler, indent=2), encoding="utf-8")

    val_rows, test_rows, pred_frames, histories = [], [], [], []
    for mode in MODES:
        seed_everything(args.seed)
        val_row, test_row, pred_frame, history = train_mode(
            mode, frame, visual, sensors, args, device, args.output_dir
        )
        val_rows.append(val_row)
        test_rows.append(test_row)
        pred_frames.append(pred_frame)
        histories.extend(history)

    # Persistence is the previous recorded beam within each sequence. First
    # samples have no in-sequence predecessor and are excluded from this row.
    frame = frame.copy()
    frame["previous_beam"] = frame.groupby("seq_index")["unit1_beam_index"].shift(1)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    has_previous = test_frame["previous_beam"].notna().to_numpy()
    persistence_pred = test_frame.loc[has_previous, "previous_beam"].to_numpy(dtype=np.int64)
    persistence_true = test_frame.loc[has_previous, "unit1_beam_index"].to_numpy(dtype=np.int64)
    persistence_row = {"model": "persistence_previous_beam", "split": "test",
                       "n": int(has_previous.sum()), "samples_with_power": 0}
    for k in (1, 3, 5):
        persistence_row[f"top{k}_accuracy"] = float(np.mean(persistence_pred == persistence_true)) if k == 1 else float("nan")
    persistence_row["macro_f1_present_classes"] = float("nan")
    persistence_row["classes_present"] = int(test_frame.loc[has_previous, "unit1_beam_index"].nunique())
    persistence_row["top1_power_regret_db"] = float("nan")
    persistence_row["top3_power_regret_db"] = float("nan")
    persistence_row["top5_power_regret_db"] = float("nan")
    test_rows.append(persistence_row)
    pd.DataFrame(test_rows).to_csv(args.output_dir / "metrics.csv", index=False)
    pd.DataFrame(val_rows).to_csv(args.output_dir / "validation_metrics.csv", index=False)
    pd.concat(pred_frames, ignore_index=True).to_csv(args.output_dir / "predictions_by_split.csv", index=False)
    pd.DataFrame(histories).to_csv(args.output_dir / "training_history.csv", index=False)
    summary = {
        "task": "current-beam classification: predict beam at time t from camera embedding and sensors at time t",
        "split": "existing sequence-grouped train/validation/test assignment",
        "split_rows": {str(k): int(v) for k, v in frame.groupby("split").size().items()},
        "split_sequences": {str(k): int(v) for k, v in frame.groupby("split")["seq_index"].nunique().items()},
        "window": 1,
        "modalities": list(MODES),
        "checkpoint_selection": "validation Top-1 accuracy",
        "test_evaluation": "used only after each model was selected on validation; no test-driven tuning",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "persistence_test_samples": int(has_previous.sum()),
        "test_metrics": test_rows,
        "caution": "Scores from current-beam classification are not directly comparable to next-beam forecasting scores.",
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("TEST METRICS:", flush=True)
    print(pd.DataFrame(test_rows).to_string(index=False), flush=True)
    print(f"Saved current-beam benchmark under: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
