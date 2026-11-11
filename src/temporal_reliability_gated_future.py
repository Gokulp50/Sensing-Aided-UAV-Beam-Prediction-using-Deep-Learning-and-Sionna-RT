"""Train a sequence-safe, next-beam, camera+sensor forecasting model.

Uses a frozen ImageNet ResNet-50 feature cache, a learned per-timestep
visual/sensor reliability gate, a GRU, and a persistence-aware change head.
All files written by this script stay under --output-dir.
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
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


FEATURES = [
    "latitude", "longitude", "speed", "altitude", "distance", "height",
    "x_speed", "y_speed", "z_speed", "pitch", "roll",
]
WINDOW = 8
N_CLASSES = 64
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_joined(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    data_root = project_root / "data" / "scenario23_dev"
    scenario = pd.read_csv(data_root / "scenario23.csv")
    eda = pd.read_csv(project_root / "results" / "eda_sensor_data.csv")
    need_s = ["index", "unit1_rgb", "unit1_pwr_60ghz", "unit1_beam_index"]
    need_e = ["index", "split", "seq_index", *FEATURES]
    missing = [f"scenario23.csv:{c}" for c in need_s if c not in scenario]
    missing += [f"eda_sensor_data.csv:{c}" for c in need_e if c not in eda]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    frame = scenario[need_s].merge(eda[need_e], on="index", how="inner", validate="one_to_one")
    frame = frame.sort_values(["seq_index", "index"]).reset_index(drop=True)
    split_counts = frame.groupby("seq_index")["split"].nunique()
    if (split_counts > 1).any():
        raise ValueError("A sequence appears in more than one split; refusing a leaky run.")
    expected = {"train", "val", "test"}
    if set(frame["split"].dropna().unique()) != expected:
        raise ValueError(f"Expected split values {expected}; found {set(frame['split'].dropna().unique())}")
    beams = frame["unit1_beam_index"].to_numpy()
    if not np.isfinite(beams).all() or beams.min() < 1 or beams.max() > N_CLASSES:
        raise ValueError("Expected 1-based beam indices in the range 1..64.")
    return frame, data_root


def fit_sensor_scaler(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = frame.loc[frame["split"] == "train", FEATURES].to_numpy(np.float64)
    med = np.nanmedian(train, axis=0)
    med[~np.isfinite(med)] = 0.0
    train = np.where(np.isfinite(train), train, med)
    mean = train.mean(0)
    std = train.std(0)
    std[std < 1e-8] = 1.0
    values = frame[FEATURES].to_numpy(np.float64)
    values = np.where(np.isfinite(values), values, med)
    scaled = np.clip((values - mean) / std, -8, 8).astype(np.float32)
    return scaled, mean.astype(np.float32), std.astype(np.float32), med.astype(np.float32)


class FrameDataset(Dataset):
    def __init__(self, rows: pd.DataFrame, data_root: Path, tfm):
        self.rows = rows.reset_index(drop=True)
        self.data_root = data_root
        self.tfm = tfm

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i: int):
        row = self.rows.iloc[i]
        path = self.data_root / str(row["unit1_rgb"])
        with Image.open(path) as im:
            image = self.tfm(im.convert("RGB"))
        return image, int(row["index"])


def get_image_features(frame: pd.DataFrame, data_root: Path, output_dir: Path,
                       device: torch.device, image_batch: int, rebuild: bool) -> tuple[np.ndarray, dict[int, int]]:
    cache_path = output_dir / "resnet50_frame_features.npy"
    ids_path = output_dir / "resnet50_frame_indices.npy"
    ids = frame["index"].to_numpy(np.int64)
    if not rebuild and cache_path.exists() and ids_path.exists():
        cached_ids = np.load(ids_path, allow_pickle=False)
        if np.array_equal(cached_ids, ids):
            print(f"Reusing cached image features: {cache_path}", flush=True)
            return np.load(cache_path, mmap_mode="r"), {int(v): i for i,v in enumerate(ids)}
        print("Existing image cache is from a different sample order; rebuilding it.", flush=True)

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    ds = FrameDataset(frame[["index", "unit1_rgb"]], data_root, tfm)
    loader = DataLoader(ds, batch_size=image_batch, shuffle=False, num_workers=0,
                        pin_memory=(device.type == "cuda"))
    weights = models.ResNet50_Weights.DEFAULT
    backbone = models.resnet50(weights=weights)
    backbone.fc = nn.Identity()
    backbone.eval().to(device)
    result = np.empty((len(ds), 2048), dtype=np.float32)
    start = time.perf_counter()
    cursor = 0
    with torch.inference_mode():
        for batch_no, (images, _) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = backbone(images)
            else:
                feats = backbone(images)
            b = feats.shape[0]
            result[cursor:cursor+b] = feats.float().cpu().numpy()
            cursor += b
            if batch_no % 50 == 0 or cursor == len(ds):
                print(f"Frame features: {cursor}/{len(ds)}", flush=True)
    np.save(cache_path, result, allow_pickle=False)
    np.save(ids_path, ids, allow_pickle=False)
    print(f"Feature extraction took {(time.perf_counter()-start)/60:.1f} min", flush=True)
    del backbone
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.load(cache_path, mmap_mode="r"), {int(v): i for i,v in enumerate(ids)}


def make_windows(frame: pd.DataFrame, scaled_sensors: np.ndarray, image_lookup: dict[int, int]):
    sensor_by_id = {int(i): scaled_sensors[pos] for pos,i in enumerate(frame["index"].to_numpy())}
    rows_by_id = {int(r["index"]): r for _,r in frame.iterrows()}
    splits = {s: [] for s in ("train", "val", "test")}
    for seq, group in frame.groupby("seq_index", sort=True):
        group = group.sort_values("index")
        ids = group["index"].to_numpy(np.int64)
        beams = group["unit1_beam_index"].to_numpy(np.int64)
        split = str(group["split"].iloc[0])
        for start in range(0, len(group)-WINDOW):
            in_ids = ids[start:start+WINDOW]
            target_id = int(ids[start+WINDOW])
            if any(int(i) not in image_lookup for i in in_ids):
                continue
            target_row = rows_by_id[target_id]
            splits[split].append({
                "frames": [image_lookup[int(i)] for i in in_ids],
                "sensor": np.stack([sensor_by_id[int(i)] for i in in_ids]),
                "target": int(beams[start+WINDOW])-1,
                "previous": int(beams[start+WINDOW-1])-1,
                "index": target_id,
                "seq": int(seq),
                "power_path": str(target_row["unit1_pwr_60ghz"]),
                "speed": float(target_row["speed"]),
                "altitude": float(target_row["altitude"]),
            })
    return splits


class WindowDataset(Dataset):
    def __init__(self, records, image_features):
        self.records = records
        self.image_features = image_features

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        vf = np.asarray(self.image_features[r["frames"]], dtype=np.float32).copy()
        return (torch.from_numpy(vf), torch.from_numpy(r["sensor"].copy()),
                r["target"], r["previous"], i)


class ReliabilityGatedGRU(nn.Module):
    def __init__(self, sensor_dim=11, image_dim=2048, hidden=192, dropout=0.25,
                 modality_dropout=0.15):
        super().__init__()
        self.modality_dropout = modality_dropout
        self.visual_encoder = nn.Sequential(
            nn.Linear(image_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Sequential(nn.Linear(hidden*2, hidden//2), nn.GELU(), nn.Linear(hidden//2, 2))
        self.temporal = nn.GRU(hidden, hidden, num_layers=1, batch_first=True)
        self.temporal_dropout = nn.Dropout(dropout)
        self.beam_head = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, N_CLASSES))
        self.change_head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden//2, 1))

    def forward(self, visual, sensors, force_modality=None):
        v = self.visual_encoder(visual)
        s = self.sensor_encoder(sensors)
        if force_modality is None:
            gate_logits = self.gate(torch.cat([v, s], dim=-1))
            if self.training and self.modality_dropout > 0:
                keep = torch.rand((visual.size(0), 1, 2), device=visual.device) >= self.modality_dropout
                both_missing = ~keep.any(dim=-1, keepdim=True)
                if both_missing.any():
                    keep[both_missing.expand_as(keep)] = False
                    choose_visual = torch.rand((visual.size(0), 1, 1), device=visual.device) < 0.5
                    keep[..., 0:1] |= both_missing & choose_visual
                    keep[..., 1:2] |= both_missing & ~choose_visual
                gate_logits = gate_logits.masked_fill(~keep, -1e4)
            weights = torch.softmax(gate_logits, dim=-1)
        elif force_modality in ("visual", "sensor", "equal"):
            fixed = {"visual": (1.0, 0.0), "sensor": (0.0, 1.0), "equal": (0.5, 0.5)}[force_modality]
            weights = visual.new_tensor(fixed).view(1, 1, 2).expand(visual.size(0), visual.size(1), 2)
        else:
            raise ValueError("force_modality must be None, 'visual', 'sensor', or 'equal'")
        fused = weights[..., 0:1] * v + weights[..., 1:2] * s
        sequence, _ = self.temporal(fused)
        last = self.temporal_dropout(sequence[:, -1])
        return self.beam_head(last), self.change_head(last).squeeze(-1), weights[:, -1]


def make_final_prob(class_logits, change_logits, previous, temperature=1.0, bias=0.0):
    change = torch.sigmoid(change_logits / temperature + bias)
    conditional = torch.softmax(class_logits, dim=-1)
    rows = torch.arange(class_logits.shape[0], device=class_logits.device)
    conditional = conditional.clone()
    conditional[rows, previous] = 0.0
    conditional = conditional / conditional.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs = conditional * change[:, None]
    probs[rows, previous] = 1.0 - change
    return probs


def tune_change_calibration(class_logits, change_logits, y, previous):
    best = None
    for temp in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for bias in (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5):
            probs = make_final_prob(class_logits, change_logits, previous, temp, bias)
            pred = probs.argmax(dim=1)
            top1 = (pred == y).float().mean().item()
            nll = F.nll_loss(torch.log(probs.clamp_min(1e-12)), y).item()
            candidate = (top1, -nll, -abs(temp-1), -abs(bias), temp, bias)
            if best is None or candidate > best:
                best = candidate
    return best[-2], best[-1]


def evaluate(model, loader, device, *, calibrate=False):
    model.eval()
    all_class, all_change, all_y, all_prev, all_idx = [], [], [], [], []
    with torch.inference_mode():
        for visual, sensors, y, prev, idx in loader:
            visual = visual.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True)
            logits, change_logits, _ = model(visual, sensors)
            all_class.append(logits.cpu())
            all_change.append(change_logits.cpu())
            all_y.append(y)
            all_prev.append(prev)
            all_idx.append(idx)
    cls = torch.cat(all_class)
    chg = torch.cat(all_change)
    y = torch.cat(all_y).long()
    prev = torch.cat(all_prev).long()
    idx = torch.cat(all_idx).long()
    temp, bias = tune_change_calibration(cls, chg, y, prev) if calibrate else (1.0, 0.0)
    probs = make_final_prob(cls, chg, prev, temp, bias)
    return {"class_logits": cls, "change_logits": chg, "y": y, "previous": prev,
            "idx": idx, "probs": probs, "temperature": temp, "bias": bias}


def classification_metrics(prob: torch.Tensor, y: torch.Tensor) -> dict:
    top = prob.topk(5, dim=1).indices
    pred = top[:, 0]
    result = {f"top{k}_accuracy": float((top[:, :k] == y[:, None]).any(1).float().mean())
              for k in (1,3,5)}
    tp = torch.bincount(y[pred == y], minlength=N_CLASSES).float()
    predicted = torch.bincount(pred, minlength=N_CLASSES).float()
    actual = torch.bincount(y, minlength=N_CLASSES).float()
    f1 = 2 * tp / (predicted + actual).clamp_min(1)
    labels_present = (actual + predicted) > 0
    result["macro_f1"] = float(f1[labels_present].mean())
    return result


def mean_power_regret(data_root: Path, records, predicted_zero_based: np.ndarray) -> tuple[float, int]:
    regrets = []
    for r, beam0 in zip(records, predicted_zero_based):
        rel = r["power_path"].replace("\\", "/").removeprefix("./")
        power = np.atleast_1d(np.loadtxt(data_root / rel, dtype=np.float64))
        if beam0 < 0 or beam0 >= len(power):
            continue
        power = np.maximum(power, 1e-15)
        regrets.append(10 * np.log10(float(power.max()) / float(power[int(beam0)])))
    return (float(np.mean(regrets)) if regrets else float("nan"), len(regrets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--image-batch-size", type=int, default=24)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--seed", type=int, default=23)
    ap.add_argument("--rebuild-image-cache", action="store_true")
    args = ap.parse_args()
    root, out = args.project_root.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any((out / f).exists() for f in ("best_model.pt", "metrics.csv", "test_predictions.csv")):
        raise FileExistsError(f"Output folder already contains a run. Choose a new --output-dir: {out}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)
        torch.backends.cudnn.benchmark = True

    frame, data_root = load_joined(root)
    scaled, mean, std, median = fit_sensor_scaler(frame)
    (out / "sensor_scaler.json").write_text(json.dumps({"features": FEATURES,
        "mean": mean.tolist(), "std": std.tolist(), "median_imputation": median.tolist()}, indent=2), encoding="utf-8")
    print("Sequence split samples:", frame.groupby("split").size().to_dict(), flush=True)

    image_features, lookup = get_image_features(frame, data_root, out, device,
                                                args.image_batch_size, args.rebuild_image_cache)
    windows = make_windows(frame, scaled, lookup)
    for name, rows in windows.items():
        if not rows:
            raise ValueError(f"No {name} windows created")
    print("Future windows:", {k:len(v) for k,v in windows.items()}, flush=True)
    datasets = {k: WindowDataset(v, image_features) for k,v in windows.items()}
    loaders = {
        "train": DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True, num_workers=0,
                            pin_memory=(device.type=="cuda")),
        "val": DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False, num_workers=0,
                          pin_memory=(device.type=="cuda")),
        "test": DataLoader(datasets["test"], batch_size=args.batch_size, shuffle=False, num_workers=0,
                           pin_memory=(device.type=="cuda")),
    }

    model = ReliabilityGatedGRU().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)
    best_top1, best_epoch, stale = -1.0, 0, 0
    best_calibration = (1.0, 0.0)
    training_start = time.perf_counter()
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        loss_sum, seen = 0.0, 0
        for visual, sensors, y, prev, _ in loaders["train"]:
            visual = visual.to(device, non_blocking=True)
            sensors = sensors.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()
            prev = prev.to(device, non_blocking=True).long()
            changed = y != prev
            optimizer.zero_grad(set_to_none=True)
            logits, change_logits, _ = model(visual, sensors)
            change_loss = F.binary_cross_entropy_with_logits(change_logits, changed.float())
            if changed.any():
                beam_loss = F.cross_entropy(logits[changed], y[changed])
            else:
                beam_loss = logits.sum() * 0.0
            loss = change_loss + beam_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
            seen += len(y)

        val = evaluate(model, loaders["val"], device, calibrate=True)
        val_metrics = classification_metrics(val["probs"], val["y"])
        scheduler.step(val_metrics["top1_accuracy"])
        epoch_loss = loss_sum / max(seen, 1)
        row = {"epoch": epoch, "train_loss": epoch_loss,
               "val_top1": val_metrics["top1_accuracy"], "val_top3": val_metrics["top3_accuracy"],
               "val_top5": val_metrics["top5_accuracy"], "val_macro_f1": val_metrics["macro_f1"],
               "change_temperature": val["temperature"], "change_bias": val["bias"],
               "learning_rate": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"Epoch {epoch:02d}/{args.epochs} loss={epoch_loss:.4f} val_top1={row['val_top1']:.4f} "
              f"top3={row['val_top3']:.4f} top5={row['val_top5']:.4f} macroF1={row['val_macro_f1']:.4f}", flush=True)
        if row["val_top1"] > best_top1 + 1e-6:
            best_top1, best_epoch, stale = row["val_top1"], epoch, 0
            best_calibration = (val["temperature"], val["bias"])
            torch.save({"model": model.state_dict(), "epoch": epoch,
                        "temperature": best_calibration[0], "bias": best_calibration[1],
                        "sensor_mean": torch.from_numpy(mean), "sensor_std": torch.from_numpy(std),
                        "sensor_median": torch.from_numpy(median),
                        "features": FEATURES, "window": WINDOW}, out / "best_model.pt")
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}; best epoch {best_epoch}.", flush=True)
                break

    pd.DataFrame(history).to_csv(out / "training_history.csv", index=False)
    checkpoint = torch.load(out / "best_model.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    temp, bias = float(checkpoint["temperature"]), float(checkpoint["bias"])
    test = evaluate(model, loaders["test"], device, calibrate=False)
    probs = make_final_prob(test["class_logits"], test["change_logits"], test["previous"], temp, bias)
    test_metrics = classification_metrics(probs, test["y"])
    pred = probs.argmax(1).numpy()
    y = test["y"].numpy()
    prev = test["previous"].numpy()
    persist_top1 = float(np.mean(prev == y))
    model_regret, n_power = mean_power_regret(data_root, windows["test"], pred)
    persist_regret, _ = mean_power_regret(data_root, windows["test"], prev)
    test_metrics.update({"split":"test", "model":"reliability_gated_camera_sensor_next_beam",
                         "n":len(y), "persistence_top1_same_windows":persist_top1,
                         "mean_power_regret_db":model_regret,
                         "persistence_mean_power_regret_db":persist_regret,
                         "samples_with_power":n_power, "best_epoch":best_epoch,
                         "training_seconds":time.perf_counter()-training_start})
    pd.DataFrame([test_metrics]).to_csv(out / "metrics.csv", index=False)

    train_speed_median = float(frame.loc[frame["split"] == "train", "speed"].median())
    train_altitude_median = float(frame.loc[frame["split"] == "train", "altitude"].median())
    test_speed = np.asarray([r["speed"] for r in windows["test"]], dtype=np.float64)
    test_altitude = np.asarray([r["altitude"] for r in windows["test"]], dtype=np.float64)
    condition_rows = []
    for factor, vals, threshold in (
        ("speed", test_speed, train_speed_median),
        ("altitude", test_altitude, train_altitude_median),
    ):
        for label, mask in (("low_or_equal", vals <= threshold), ("high", vals > threshold)):
            if mask.any():
                condition_rows.append({
                    "factor": factor, "condition": label, "train_median_threshold": threshold,
                    "n": int(mask.sum()), "model_top1": float(np.mean(pred[mask] == y[mask])),
                    "persistence_top1": float(np.mean(prev[mask] == y[mask])),
                })
    pd.DataFrame(condition_rows).to_csv(out / "condition_metrics.csv", index=False)

    top = probs.topk(5, dim=1).indices.numpy() + 1
    gates = []
    # A small, separate pass returns the learned fusion weights for the same test windows.
    model.eval()
    with torch.inference_mode():
        for visual, sensors, _, _, _ in loaders["test"]:
            _, _, gate = model(visual.to(device), sensors.to(device))
            gates.extend(gate.cpu().numpy().tolist())
    pred_rows=[]
    for i,r in enumerate(windows["test"]):
        pred_rows.append({"seq_index":r["seq"], "target_index":r["index"],
            "previous_beam":r["previous"]+1, "true_next_beam":r["target"]+1,
            "predicted_next_beam":int(pred[i])+1, "p_change":float(torch.sigmoid(test["change_logits"][i]/temp+bias)),
            "top3_beams":" ".join(map(str,top[i,:3])), "top5_beams":" ".join(map(str,top[i,:5])),
            "visual_gate":float(gates[i][0]), "sensor_gate":float(gates[i][1]),
            "speed":r["speed"], "altitude":r["altitude"]})
    pd.DataFrame(pred_rows).to_csv(out / "test_predictions.csv", index=False)

    summary={"model":"frozen ResNet-50 + 11-sensor encoder + learned reliability gate + GRU + beam-change prior",
        "target":"next logged beam (t+1) after 8 prior samples", "split":"existing sequence-level split",
        "window_counts":{k:len(v) for k,v in windows.items()}, "best_epoch":best_epoch,
        "condition_thresholds_from_train":{"speed_median":train_speed_median,"altitude_median":train_altitude_median},
        "change_calibration":{"temperature":temp,"bias":bias}, "device":str(device),
        "gpu":torch.cuda.get_device_name(0) if device.type=="cuda" else None,
        "test_metrics":test_metrics,
        "note":"Test set evaluated once after validation model selection; persistence evaluated on the same future windows."}
    (out/"run_summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("TEST:", json.dumps(test_metrics, indent=2), flush=True)
    print(f"Saved run outputs under: {out}", flush=True)


if __name__ == "__main__":
    main()
