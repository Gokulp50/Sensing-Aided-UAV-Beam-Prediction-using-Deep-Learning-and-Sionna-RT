"""Five-fold sequence-grouped CV for the current-beam gated ResNet-18 model.

Every sequence is held out exactly once for testing. Within each outer fold,
training sequences are split again into train/validation groups; validation
Top-1 selects the checkpoint. Sensor normalization is fit on that fold's train
groups only. The script writes reports under --output-dir and does not alter
the project or any prior run.
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
from torch.utils.data import DataLoader
from torchvision import transforms

from current_beam_benchmark import FEATURES, load_frame, metric_row, normalize_sensors, add_power_metrics
from current_beam_fusion_resnet18 import GatedResNet18, ImageSensorDataset, predict, MEAN, STD


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loader_for(rows: pd.DataFrame, sensor_values: np.ndarray, split: str,
               data_root: Path, transform, batch_size: int, device):
    indices = np.flatnonzero(rows["split"].to_numpy() == split)
    ds = ImageSensorDataset(rows.iloc[indices], data_root, sensor_values, indices, transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=(split == "train"), num_workers=0,
                      pin_memory=(device.type == "cuda"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--backbone-lr", type=float, default=5e-5)
    ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--seed", type=int, default=71)
    ap.add_argument("--augmentation-profile", choices=("standard", "illumination_robust"),
                    default="standard",
                    help="Use the same stronger brightness/contrast augmentation as the robust candidate.")
    args = ap.parse_args()
    if args.folds < 3:
        raise ValueError("Use at least 3 folds for grouped cross-validation.")
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protected = ["fold_metrics.csv", "out_of_fold_predictions.csv", "cv_summary.json"]
    if any((out / f).exists() for f in protected):
        raise FileExistsError(f"Output folder already contains a CV run: {out}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}; "
              f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB", flush=True)

    rows, data_root = load_frame(root)
    rows["previous_beam"] = rows.groupby("seq_index")["unit1_beam_index"].shift(1)
    groups = np.sort(rows["seq_index"].unique())
    if len(groups) < args.folds * 2:
        raise ValueError(f"Need at least {args.folds * 2} sequences for {args.folds}-fold CV with validation groups; found {len(groups)}.")
    rng = np.random.default_rng(args.seed)
    shuffled_groups = rng.permutation(groups)
    test_group_folds = np.array_split(shuffled_groups, args.folds)

    if args.augmentation_profile == "illumination_robust":
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

    fold_rows = []
    all_predictions = []
    all_history = []
    total_start = time.perf_counter()
    for fold_idx, test_groups in enumerate(test_group_folds, start=1):
        remaining = np.array([g for g in groups if g not in set(test_groups)])
        fold_rng = np.random.default_rng(args.seed + fold_idx)
        val_count = max(2, int(round(len(remaining) * 0.15)))
        val_groups = fold_rng.choice(remaining, size=val_count, replace=False)
        train_groups = np.array([g for g in remaining if g not in set(val_groups)])

        fold = rows.copy()
        fold["split"] = np.select(
            [fold["seq_index"].isin(test_groups), fold["seq_index"].isin(val_groups)],
            ["test", "val"], default="train"
        )
        # Refuse any accidental overlap between fold partitions.
        group_splits = fold.groupby("seq_index")["split"].nunique()
        if (group_splits != 1).any():
            raise RuntimeError(f"Fold {fold_idx} has a sequence crossing split boundaries.")
        sensors, sensor_stats = normalize_sensors(fold)
        loaders = {
            split: loader_for(fold, sensors, split, data_root,
                              train_transform if split == "train" else eval_transform,
                              args.batch_size, device)
            for split in ("train", "val", "test")
        }
        seed_everything(args.seed + fold_idx)
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
        best_weights = None
        history = []
        fold_start = time.perf_counter()
        print(f"\nFOLD {fold_idx}/{args.folds}: train={len(train_groups)} sequences, "
              f"val={len(val_groups)}, test={len(test_groups)}", flush=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            loss_sum, seen = 0.0, 0
            for images, sensors_batch, labels, _ in loaders["train"]:
                images = images.to(device, non_blocking=True)
                sensors_batch = sensors_batch.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True).long()
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16,
                                    enabled=(device.type == "cuda")):
                    logits, _ = model(images, sensors_batch)
                    loss = nn.functional.cross_entropy(logits, labels, label_smoothing=0.02)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
                loss_sum += float(loss.detach()) * len(labels)
                seen += len(labels)

            val_probs, val_y, _, _ = predict(model, loaders["val"], device)
            val_metrics = metric_row(val_probs, val_y, "gated_resnet18_sensor", "val")
            scheduler.step(val_metrics["top1_accuracy"])
            record = {"fold": fold_idx, "epoch": epoch, "train_loss": loss_sum / max(seen, 1),
                      "val_top1_accuracy": val_metrics["top1_accuracy"],
                      "val_top3_accuracy": val_metrics["top3_accuracy"],
                      "val_top5_accuracy": val_metrics["top5_accuracy"]}
            history.append(record)
            print(f"fold {fold_idx} epoch {epoch:02d}/{args.epochs} "
                  f"loss={record['train_loss']:.4f} val_top1={record['val_top1_accuracy']:.4f} "
                  f"top3={record['val_top3_accuracy']:.4f} top5={record['val_top5_accuracy']:.4f}", flush=True)
            if record["val_top1_accuracy"] > best_top1 + 1e-6:
                best_top1, best_epoch, stale = record["val_top1_accuracy"], epoch, 0
                best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
                if stale >= args.patience:
                    break

        if best_weights is None:
            raise RuntimeError(f"No model checkpoint selected in fold {fold_idx}.")
        model.load_state_dict(best_weights)
        test_probs, test_y, test_ids, gates = predict(model, loaders["test"], device)
        test_metric = metric_row(test_probs, test_y, "gated_resnet18_sensor", "test")
        test_rows = fold.loc[fold["split"] == "test"].reset_index(drop=True)
        top0 = test_probs.topk(5, dim=1).indices.numpy()
        test_metric = add_power_metrics(test_metric, test_rows, data_root, top0)
        test_metric["fold"] = fold_idx
        test_metric["best_epoch"] = best_epoch
        test_metric["test_sequences"] = " ".join(map(str, sorted(test_groups.tolist())))
        test_metric["train_sensor_mean"] = json.dumps(sensor_stats["mean"])
        test_metric["mean_visual_gate"] = float(gates[:, 0].mean())
        test_metric["mean_sensor_gate"] = float(gates[:, 1].mean())

        true = test_y.numpy() + 1
        pred = test_probs.argmax(1).numpy() + 1
        top5 = top0 + 1
        prev = test_rows["previous_beam"].to_numpy()
        previous_valid = ~pd.isna(prev)
        persist = np.where(previous_valid, prev, 0).astype(np.int64)
        test_metric["persistence_comparable_n"] = int(previous_valid.sum())
        test_metric["persistence_top1_accuracy"] = (
            float(np.mean(persist[previous_valid] == true[previous_valid])) if previous_valid.any() else float("nan")
        )
        test_metric["model_top1_accuracy_on_persistence_subset"] = (
            float(np.mean(pred[previous_valid] == true[previous_valid])) if previous_valid.any() else float("nan")
        )
        fold_rows.append(test_metric)
        for j, idx in enumerate(test_ids.numpy()):
            all_predictions.append({
                "fold": fold_idx, "index": int(idx), "seq_index": int(test_rows.iloc[j]["seq_index"]),
                "true_beam": int(true[j]), "predicted_beam": int(pred[j]),
                "previous_beam": int(persist[j]) if previous_valid[j] else "",
                "top3_beams": " ".join(map(str, top5[j, :3])),
                "top5_beams": " ".join(map(str, top5[j, :5])),
                "visual_gate": float(gates[j, 0]), "sensor_gate": float(gates[j, 1]),
            })
        for rec in history:
            all_history.append(rec)
        print(f"FOLD {fold_idx} TEST: top1={test_metric['top1_accuracy']:.4f} "
              f"top3={test_metric['top3_accuracy']:.4f} top5={test_metric['top5_accuracy']:.4f} "
              f"power_regret={test_metric['top1_power_regret_db']:.4f} dB "
              f"({time.perf_counter()-fold_start:.1f}s)", flush=True)
        del model, optimizer, scheduler, scaler
        if device.type == "cuda":
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(all_predictions)
    # Pooled out-of-fold performance weights folds by their number of samples.
    pooled = {}
    for name, predicate in (("all_oof", np.ones(len(pred_df), dtype=bool)),
                            ("oof_with_previous", pred_df["previous_beam"].astype(str).ne("").to_numpy())):
        subset = pred_df.loc[predicate]
        pooled[name] = {
            "n": int(len(subset)),
            "top1_accuracy": float(np.mean(subset["predicted_beam"].to_numpy() == subset["true_beam"].to_numpy())) if len(subset) else float("nan"),
            "top3_accuracy": float(np.mean([int(y) in [int(x) for x in row.split()[:3]]
                                             for y, row in zip(subset["true_beam"], subset["top3_beams"])])) if len(subset) else float("nan"),
            "top5_accuracy": float(np.mean([int(y) in [int(x) for x in row.split()[:5]]
                                             for y, row in zip(subset["true_beam"], subset["top5_beams"])])) if len(subset) else float("nan"),
        }
    for split_name in ("top1_accuracy", "top3_accuracy", "top5_accuracy", "macro_f1_present_classes",
                       "top1_power_regret_db", "top3_power_regret_db", "top5_power_regret_db"):
        vals = fold_df[split_name].dropna().to_numpy(dtype=np.float64)
        pooled[f"fold_{split_name}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        pooled[f"fold_{split_name}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

    fold_df.to_csv(out / "fold_metrics.csv", index=False)
    pred_df.to_csv(out / "out_of_fold_predictions.csv", index=False)
    pd.DataFrame(all_history).to_csv(out / "training_history.csv", index=False)
    summary = {
        "task": "current-beam classification from synchronized RGB + 11 sensors at time t",
        "method": "ImageNet-pretrained ResNet-18 fine-tuned end-to-end + learned reliability gate",
        "augmentation_profile": args.augmentation_profile,
        "cv": f"{args.folds}-fold grouped by sequence; each sequence tested once",
        "validation": "validation groups drawn from outer-training sequences; sensor scaler fit on inner training groups only",
        "sequence_count": int(len(groups)), "folds": int(args.folds), "pooled_metrics": pooled,
        "per_fold_metrics": fold_rows,
        "total_training_seconds": time.perf_counter() - total_start,
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "caveat": "This CV follows earlier model development on the same dataset; disclose it as grouped cross-validation after exploratory iteration, not as a pristine external test.",
    }
    (out / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("POOLED OUT-OF-FOLD:", json.dumps(pooled, indent=2), flush=True)
    print(f"Saved grouped CV outputs under: {out}", flush=True)


if __name__ == "__main__":
    main()
