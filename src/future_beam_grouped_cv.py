"""Five-fold sequence-grouped CV for future-beam prediction.

Each sequence is held out once. Every fold gets a fresh model initialized from
the same architecture defaults (no checkpoint trained on held-out sequences),
fold-specific sensor scaling, and validation-only checkpoint/calibration
selection. Frozen visual features are read from an existing cache.
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
from torch.utils.data import DataLoader

import temporal_reliability_gated_future as base
import temporal_power_aware_future as power_base


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_existing_cache(frame: pd.DataFrame, cache_dir: Path):
    features_path = cache_dir / "resnet50_frame_features.npy"
    indices_path = cache_dir / "resnet50_frame_indices.npy"
    if not features_path.exists() or not indices_path.exists():
        raise FileNotFoundError(f"Missing existing visual feature cache in {cache_dir}")
    ids = frame["index"].to_numpy(dtype=np.int64)
    cached_ids = np.load(indices_path, allow_pickle=False)
    if not np.array_equal(ids, cached_ids):
        raise ValueError("Cached frame order does not match Scenario 23 rows; refusing misaligned features.")
    features = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if features.shape != (len(frame), 2048):
        raise ValueError(f"Unexpected feature shape {features.shape}; expected ({len(frame)}, 2048).")
    return features, {int(v): i for i, v in enumerate(ids)}


def preload_power_targets(frame: pd.DataFrame, data_root: Path) -> dict[int, np.ndarray]:
    """Read each sample's measured beam-power vector once for all folds."""
    result = {}
    for row in frame[["index", "unit1_pwr_60ghz"]].itertuples(index=False):
        rel = str(row.unit1_pwr_60ghz).replace("\\", "/").removeprefix("./")
        power = np.maximum(np.atleast_1d(np.loadtxt(data_root / rel, dtype=np.float64)), 1e-15)
        if len(power) != base.N_CLASSES or not np.isfinite(power).all():
            raise ValueError(f"Expected {base.N_CLASSES} finite beam powers at {rel}.")
        power_db = 10.0 * np.log10(power)
        result[int(row.index)] = (power_db - power_db.max()).astype(np.float32)
    return result


def make_power_windows(frame, scaled_sensors, image_lookup, power_by_index):
    windows = base.make_windows(frame, scaled_sensors, image_lookup)
    for split_rows in windows.values():
        for row in split_rows:
            row["power_db"] = power_by_index[row["index"]]
    return windows


def get_loaders(windows, features, batch_size, device):
    datasets = {k: power_base.PowerWindowDataset(v, features) for k, v in windows.items()}
    return {
        "train": DataLoader(datasets["train"], batch_size=batch_size, shuffle=True,
                            num_workers=0, pin_memory=(device.type == "cuda")),
        "val": DataLoader(datasets["val"], batch_size=max(batch_size, 128), shuffle=False,
                          num_workers=0, pin_memory=(device.type == "cuda")),
        "test": DataLoader(datasets["test"], batch_size=max(batch_size, 128), shuffle=False,
                            num_workers=0, pin_memory=(device.type == "cuda")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--feature-cache-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--objective", choices=("power", "top1"), default="power",
                    help="Select checkpoints by validation Top-5 power regret or validation Top-1 accuracy.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=83)
    ap.add_argument("--power-temperature", type=float, default=0.75)
    ap.add_argument("--power-listwise-weight", type=float, default=0.6)
    ap.add_argument("--expected-regret-weight", type=float, default=0.05)
    ap.add_argument("--changed-label-weight", type=float, default=0.25)
    args = ap.parse_args()
    if args.folds < 3:
        raise ValueError("Use at least 3 folds.")
    root = args.project_root.resolve()
    cache_dir = args.feature_cache_dir.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = ["fold_metrics.csv", "out_of_fold_predictions.csv", "cv_summary.json"]
    if any((out / name).exists() for name in protected):
        raise FileExistsError(f"Output folder already contains a CV run: {out}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)

    frame, data_root = base.load_joined(root)
    features, lookup = load_existing_cache(frame, cache_dir)
    power_by_index = preload_power_targets(frame, data_root)
    groups = np.sort(frame["seq_index"].unique())
    if len(groups) < args.folds * 2:
        raise ValueError(f"Need at least {args.folds * 2} sequences; found {len(groups)}.")
    shuffled = np.random.default_rng(args.seed).permutation(groups)
    test_group_folds = np.array_split(shuffled, args.folds)
    print(f"Sequences: {len(groups)}; folds: {args.folds}; rows: {len(frame)}", flush=True)

    fold_metrics = []
    prediction_rows = []
    history_rows = []
    pooled_probs, pooled_y, pooled_prev, pooled_power = [], [], [], []
    started_all = time.perf_counter()
    for fold_idx, test_groups in enumerate(test_group_folds, start=1):
        test_set = set(test_groups.tolist())
        remaining = np.array([g for g in groups if g not in test_set])
        fold_rng = np.random.default_rng(args.seed + fold_idx)
        val_count = max(2, int(round(0.15 * len(remaining))))
        val_groups = fold_rng.choice(remaining, size=val_count, replace=False)
        val_set = set(val_groups.tolist())
        train_groups = np.array([g for g in remaining if g not in val_set])

        fold_frame = frame.copy()
        fold_frame["split"] = np.select(
            [fold_frame["seq_index"].isin(test_groups), fold_frame["seq_index"].isin(val_groups)],
            ["test", "val"], default="train"
        )
        seq_split_counts = fold_frame.groupby("seq_index")["split"].nunique()
        if (seq_split_counts != 1).any():
            raise RuntimeError(f"Sequence overlap detected in fold {fold_idx}.")
        scaled, sensor_mean, sensor_std, sensor_median = base.fit_sensor_scaler(fold_frame)
        windows = make_power_windows(fold_frame, scaled, lookup, power_by_index)
        if any(not windows[s] for s in ("train", "val", "test")):
            raise ValueError(f"Fold {fold_idx} produced an empty data split: "
                             f"{{k:len(v) for k,v in windows.items()}}")
        loaders = get_loaders(windows, features, args.batch_size, device)
        seed_everything(args.seed + fold_idx)
        model = base.ReliabilityGatedGRU().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-3)
        selection_mode = "min" if args.objective == "power" else "max"
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=selection_mode, factor=0.5, patience=2
        )
        best_selection, best_epoch, stale, best_state, best_calibration = (
            (float("inf") if args.objective == "power" else -float("inf")), 0, 0, None, (1.0, 0.0)
        )
        best_val_power_regret = float("inf")
        fold_started = time.perf_counter()
        print(f"\nFOLD {fold_idx}/{args.folds}: train={len(train_groups)}, "
              f"val={len(val_groups)}, test={len(test_groups)} sequences; "
              f"windows={{{', '.join(f'{k}:{len(v)}' for k,v in windows.items())}}}", flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for visual, sensors, y, previous, _, power_db in loaders["train"]:
                visual = visual.to(device, non_blocking=True)
                sensors = sensors.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True).long()
                previous = previous.to(device, non_blocking=True).long()
                power_db = power_db.to(device, non_blocking=True, dtype=torch.float32)
                changed = y != previous
                optimizer.zero_grad(set_to_none=True)
                logits, change_logits, _ = model(visual, sensors)
                change_loss = F.binary_cross_entropy_with_logits(change_logits, changed.float())
                if args.objective == "power":
                    list_loss, expected_regret, conditional_logits = power_base.soft_power_loss(
                        logits, previous, power_db, args.power_temperature
                    )
                    hard_loss = (F.cross_entropy(conditional_logits[changed], y[changed])
                                 if changed.any() else logits.sum() * 0.0)
                    loss = (change_loss + args.power_listwise_weight * list_loss
                            + args.changed_label_weight * hard_loss
                            + args.expected_regret_weight * expected_regret)
                else:
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

            val_data = power_base.collect(model, loaders["val"], device)
            if args.objective == "power":
                temperature, bias, _ = power_base.tune_calibration_for_top5(val_data)
            else:
                temperature, bias = base.tune_change_calibration(
                    val_data["cls"], val_data["change"], val_data["y"], val_data["prev"]
                )
            val_probs = base.make_final_prob(val_data["cls"], val_data["change"],
                                             val_data["prev"], temperature, bias)
            val_class = base.classification_metrics(val_probs, val_data["y"])
            val_regret, _ = power_base.topk_power_metrics(val_probs, val_data["power_db"], 5)
            selection_value = val_regret if args.objective == "power" else val_class["top1_accuracy"]
            scheduler.step(selection_value)
            record = {"fold": fold_idx, "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                      "val_top1": val_class["top1_accuracy"], "val_top3": val_class["top3_accuracy"],
                      "val_top5": val_class["top5_accuracy"], "val_top5_power_regret_db": val_regret,
                      "selection_objective": args.objective, "selection_value": selection_value,
                      "calibration_temperature": temperature, "calibration_bias": bias}
            history_rows.append(record)
            print(f"fold {fold_idx} epoch {epoch:02d}/{args.epochs} loss={record['train_loss']:.4f} "
                  f"val_{args.objective}="
                  f"{selection_value:.5f}{' dB' if args.objective == 'power' else ''} "
                  f"val_top5_regret={val_regret:.5f} dB top1={record['val_top1']:.4f} "
                  f"top5={record['val_top5']:.4f}", flush=True)
            improved = (selection_value < best_selection - 1e-7 if args.objective == "power"
                        else selection_value > best_selection + 1e-7)
            if improved:
                best_selection, best_epoch, stale = selection_value, epoch, 0
                best_val_power_regret = val_regret
                best_calibration = (temperature, bias)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if best_state is None:
            raise RuntimeError(f"No best model selected in fold {fold_idx}.")
        model.load_state_dict(best_state)
        test_data = power_base.collect(model, loaders["test"], device)
        temperature, bias = best_calibration
        probs = base.make_final_prob(test_data["cls"], test_data["change"],
                                     test_data["prev"], temperature, bias)
        class_metrics = base.classification_metrics(probs, test_data["y"])
        row = {"fold": fold_idx, "n": int(len(test_data["y"])), "best_epoch": best_epoch,
               "test_top1": class_metrics["top1_accuracy"], "test_top3": class_metrics["top3_accuracy"],
               "test_top5": class_metrics["top5_accuracy"], "test_macro_f1": class_metrics["macro_f1"],
               "test_top5_power_regret_db": power_base.topk_power_metrics(probs, test_data["power_db"], 5)[0],
               "test_top5_within_0_01db": power_base.topk_power_metrics(probs, test_data["power_db"], 5)[1],
               "test_top3_power_regret_db": power_base.topk_power_metrics(probs, test_data["power_db"], 3)[0],
               "test_top1_power_regret_db": power_base.topk_power_metrics(probs, test_data["power_db"], 1)[0],
               "validation_selection": args.objective,
               "validation_selection_value": best_selection,
               "val_best_top5_power_regret_db": best_val_power_regret,
               "test_persistence_top1": float((test_data["prev"] == test_data["y"]).double().mean()),
               "test_persistence_top1_power_regret_db": float(
                   (test_data["power_db"].max(1).values -
                    test_data["power_db"].gather(1, test_data["prev"].view(-1, 1)).squeeze(1)).clamp_min(0).mean()
               ),
               "test_persistence_within_0_01db": float(
                   ((test_data["power_db"].max(1).values -
                     test_data["power_db"].gather(1, test_data["prev"].view(-1, 1)).squeeze(1)).clamp_min(0) <= 0.01).double().mean()
               ),
               "test_sequences": " ".join(map(str, sorted(test_groups.tolist()))),
               "mean_visual_gate": float("nan"), "mean_sensor_gate": float("nan"),
               "seconds": time.perf_counter() - fold_started}

        # Collect the selected model's gate weights for the held-out fold.
        model.eval()
        gates = []
        with torch.inference_mode():
            for visual, sensors, _, _, _, _ in loaders["test"]:
                _, _, gate = model(visual.to(device), sensors.to(device))
                gates.append(gate.cpu())
        gate_array = torch.cat(gates)
        row["mean_visual_gate"] = float(gate_array[:, 0].mean())
        row["mean_sensor_gate"] = float(gate_array[:, 1].mean())
        fold_metrics.append(row)

        y = test_data["y"].numpy()
        previous = test_data["prev"].numpy()
        predicted = probs.argmax(1).numpy()
        top = probs.topk(5, dim=1).indices.numpy()
        records = windows["test"]
        for i, rec in enumerate(records):
            prediction_rows.append({
                "fold": fold_idx, "seq_index": rec["seq"], "target_index": rec["index"],
                "previous_beam": int(previous[i]) + 1, "true_next_beam": int(y[i]) + 1,
                "predicted_next_beam": int(predicted[i]) + 1,
                "top3_beams": " ".join(str(int(v) + 1) for v in top[i, :3]),
                "top5_beams": " ".join(str(int(v) + 1) for v in top[i, :5]),
                "p_change": float(torch.sigmoid(test_data["change"][i] / temperature + bias)),
                "visual_gate": float(gate_array[i, 0]), "sensor_gate": float(gate_array[i, 1]),
            })
        pooled_probs.append(probs.cpu())
        pooled_y.append(test_data["y"].cpu())
        pooled_prev.append(test_data["prev"].cpu())
        pooled_power.append(test_data["power_db"].cpu())
        print(f"FOLD {fold_idx} TEST: top1={row['test_top1']:.4f} top3={row['test_top3']:.4f} "
              f"top5={row['test_top5']:.4f} top5_regret={row['test_top5_power_regret_db']:.5f} dB "
              f"persistence_top1={row['test_persistence_top1']:.4f} "
              f"({row['seconds']:.1f}s)", flush=True)
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    probs = torch.cat(pooled_probs)
    y = torch.cat(pooled_y).long()
    previous = torch.cat(pooled_prev).long()
    power_db = torch.cat(pooled_power).double()
    pooled_classes = base.classification_metrics(probs, y)
    persistence_regret = (power_db.max(1).values - power_db.gather(1, previous.view(-1, 1)).squeeze(1)).clamp_min(0)
    pooled = {
        "n_future_windows": int(len(y)),
        "top1_accuracy": pooled_classes["top1_accuracy"],
        "top3_accuracy": pooled_classes["top3_accuracy"],
        "top5_accuracy": pooled_classes["top5_accuracy"],
        "macro_f1": pooled_classes["macro_f1"],
        "persistence_top1_accuracy": float((previous == y).double().mean()),
        "persistence_top1_power_regret_db": float(persistence_regret.mean()),
        "persistence_within_0_01db_fraction": float((persistence_regret <= 0.01).double().mean()),
    }
    for k in (1, 3, 5):
        regret, within = power_base.topk_power_metrics(probs, power_db, k)
        pooled[f"top{k}_power_regret_db"] = regret
        pooled[f"top{k}_within_0_01db_fraction"] = within
    fold_df = pd.DataFrame(fold_metrics)
    for col in ("test_top1", "test_top3", "test_top5", "test_macro_f1", "test_top1_power_regret_db",
                "test_top3_power_regret_db", "test_top5_power_regret_db", "test_persistence_top1"):
        vals = fold_df[col].to_numpy(dtype=np.float64)
        pooled[f"fold_{col}_mean"] = float(vals.mean())
        pooled[f"fold_{col}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(history_rows).to_csv(out / "training_history.csv", index=False)
    loss_configuration = ({"type": "power-aware listwise + expected regret + change loss",
                          "listwise_weight": args.power_listwise_weight,
                          "expected_regret_weight": args.expected_regret_weight,
                          "changed_label_weight": args.changed_label_weight,
                          "temperature_db": args.power_temperature}
                         if args.objective == "power" else
                         {"type": "change BCE + beam cross-entropy on changed samples"})
    summary = {
        "task": "predict next logged beam from eight previous synchronized image/sensor observations",
        "method": f"reliability-gated ResNet-50 feature + sensor GRU; {args.objective}-selected objective",
        "validation_selection": "minimum validation Top-5 power regret" if args.objective == "power"
                                else "maximum validation Top-1 accuracy",
        "cv": f"{args.folds}-fold sequence-grouped CV; each sequence held out exactly once",
        "sensor_scaling": "fold-specific; fit on training sequences only",
        "feature_cache": str(cache_dir),
        "loss_configuration": loss_configuration,
        "pooled_out_of_fold_metrics": pooled,
        "per_fold_metrics": fold_metrics,
        "total_training_seconds": time.perf_counter() - started_all,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "caveat": "This CV follows earlier model development on the same dataset; report as grouped CV after exploratory iteration, not as a pristine external test.",
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("POOLED OUT-OF-FOLD:", json.dumps(pooled, indent=2), flush=True)
    print(f"Saved grouped future-beam CV outputs under: {out}", flush=True)


if __name__ == "__main__":
    main()
