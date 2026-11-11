"""Sequence-safe sensor-only t+1 beam experiment.

This is an early measurable baseline for the revised methodology, not the
proposed camera+sensor reliability-gated GRU. It trains only from the existing
EDA sensor table and writes results to an isolated output folder.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd


FEATURES = [
    "latitude", "longitude", "speed", "altitude", "distance", "height",
    "x_speed", "y_speed", "z_speed", "pitch", "roll",
]
WINDOW = 8
N_CLASSES = 64


def make_windows(df: pd.DataFrame):
    records = {s: {"x": [], "y": [], "prev": [], "target_index": [], "seq": []}
               for s in ("train", "val", "test")}
    for seq, g in df.sort_values(["seq_index", "index"]).groupby("seq_index", sort=True):
        g = g.reset_index(drop=True)
        split_values = g["split"].dropna().unique()
        if len(split_values) != 1:
            raise ValueError(f"Sequence {seq} spans multiple splits: {split_values}")
        split = split_values[0]
        if split not in records:
            continue
        values = g[FEATURES].to_numpy(dtype=np.float64)
        beams = g["unit1_beam_index"].to_numpy(dtype=np.int64)
        indexes = g["index"].to_numpy(dtype=np.int64)
        for start in range(0, len(g) - WINDOW):
            past = values[start:start + WINDOW]
            # Levels plus frame-to-frame changes; target is exactly t+1.
            feat = np.concatenate([past.reshape(-1), np.diff(past, axis=0).reshape(-1)])
            records[split]["x"].append(feat)
            records[split]["prev"].append(beams[start + WINDOW - 1])
            records[split]["y"].append(beams[start + WINDOW])
            records[split]["target_index"].append(indexes[start + WINDOW])
            records[split]["seq"].append(seq)
    out = {}
    for split, d in records.items():
        out[split] = {k: np.asarray(v) for k, v in d.items()}
        if not len(out[split]["y"]):
            raise ValueError(f"No {split} future windows; check split names and sequence lengths")
    return out


def standardize(train_x, arrays):
    med = np.nanmedian(train_x, axis=0)
    med[~np.isfinite(med)] = 0.0
    train_x = np.where(np.isfinite(train_x), train_x, med)
    mean = train_x.mean(0)
    std = train_x.std(0)
    std[std < 1e-8] = 1.0
    result = {}
    for k, x in arrays.items():
        x = np.where(np.isfinite(x), x, med)
        result[k] = np.clip((x - mean) / std, -8, 8).astype(np.float32)
    return result, mean, std, med


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def adam_fit(x, y, n_out, *, epochs, lr, weight_decay, seed):
    rng = np.random.default_rng(seed)
    x = np.c_[x, np.ones(len(x), dtype=np.float32)]
    w = rng.normal(0, 0.01, (x.shape[1], n_out)).astype(np.float32)
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    beta1, beta2 = 0.9, 0.999
    y = y.astype(np.int64)
    for step in range(1, epochs + 1):
        p = softmax(x @ w)
        p[np.arange(len(y)), y] -= 1.0
        grad = (x.T @ p) / len(y)
        grad[:-1] += weight_decay * w[:-1]
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad * grad
        mh = m / (1 - beta1 ** step)
        vh = v / (1 - beta2 ** step)
        w -= lr * mh / (np.sqrt(vh) + 1e-8)
    return w


def probabilities(x, w):
    return softmax(np.c_[x, np.ones(len(x), dtype=np.float32)] @ w)


def calibrated_change(p, temperature, bias):
    eps = 1e-6
    logits = np.log(np.clip(p, eps, 1-eps) / np.clip(1-p, eps, 1-eps))
    z = np.clip(temperature * logits + bias, -30, 30)
    return 1 / (1 + np.exp(-z))


def persistence_aware(class_p, change_p, previous):
    q = class_p.copy()
    rows = np.arange(len(q))
    prev0 = previous - 1
    q[rows, prev0] = 0
    denom = q.sum(axis=1, keepdims=True)
    q = np.divide(q, denom, out=np.full_like(q, 1 / (N_CLASSES - 1)), where=denom > 0)
    q *= change_p[:, None]
    q[rows, prev0] = 1 - change_p
    return q


def score(prob, y):
    top = np.argsort(prob, axis=1)[:, ::-1]
    pred = top[:, 0] + 1
    y0 = y - 1
    topk = {f"top{k}_accuracy": float(np.mean(np.any(top[:, :k] == y0[:, None], axis=1)))
            for k in (1, 3, 5)}
    labels = np.union1d(np.unique(y0), np.unique(top[:, 0]))
    f1s = []
    for c in labels:
        tp = np.sum((top[:, 0] == c) & (y0 == c))
        fp = np.sum((top[:, 0] == c) & (y0 != c))
        fn = np.sum((top[:, 0] != c) & (y0 == c))
        f1s.append(2*tp / max(2*tp + fp + fn, 1))
    return {**topk, "macro_f1": float(np.mean(f1s)), "n": int(len(y)), "top": top + 1}


def power_regret(project_root, scenario, target_indexes, predicted):
    data_root = project_root / "data" / "scenario23_dev"
    by_index = scenario.set_index("index")
    regrets = []
    for idx, beam in zip(target_indexes, predicted):
        rel = str(by_index.loc[int(idx), "unit1_pwr_60ghz"]).replace("\\", "/")
        p = np.atleast_1d(np.loadtxt(data_root / rel.removeprefix("./"), dtype=np.float64))
        if beam < 1 or beam > len(p):
            continue
        positive = np.maximum(p, 1e-15)
        best = float(np.max(positive))
        selected = float(positive[beam - 1])
        regrets.append(10 * np.log10(best / selected))
    return float(np.mean(regrets)) if regrets else float("nan"), len(regrets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=65)
    args = parser.parse_args()
    root = args.project_root.resolve()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    eda = pd.read_csv(root / "results" / "eda_sensor_data.csv")
    scenario = pd.read_csv(root / "data" / "scenario23_dev" / "scenario23.csv")
    required = ["index", "seq_index", "split", "unit1_beam_index", *FEATURES]
    missing = [c for c in required if c not in eda.columns]
    if missing:
        raise ValueError(f"Missing columns in eda_sensor_data.csv: {missing}")
    windows = make_windows(eda)
    xs, mu, sd, med = standardize(windows["train"]["x"], {s: d["x"] for s,d in windows.items()})
    ys = {s: d["y"].astype(int) - 1 for s,d in windows.items()}
    if min(np.min(y) for y in ys.values()) < 0 or max(np.max(y) for y in ys.values()) >= N_CLASSES:
        raise ValueError("Beam indices must be 1..64")

    print(f"Windows: train={len(ys['train'])}, val={len(ys['val'])}, test={len(ys['test'])}")
    t0 = time.perf_counter()
    multiclass = adam_fit(xs["train"], ys["train"], N_CLASSES, epochs=args.epochs,
                          lr=0.035, weight_decay=0.004, seed=23)
    y_change = (windows["train"]["y"] != windows["train"]["prev"]).astype(int)
    binary = adam_fit(xs["train"], y_change, 2, epochs=args.epochs,
                      lr=0.025, weight_decay=0.01, seed=24)
    training_seconds = time.perf_counter() - t0

    val_class = probabilities(xs["val"], multiclass)
    val_change = probabilities(xs["val"], binary)[:, 1]
    val_prev = windows["val"]["prev"].astype(int)
    best = None
    for temperature in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
        for bias in (-1.25, -0.75, -0.4, 0.0, 0.4, 0.75, 1.25):
            q = persistence_aware(val_class, calibrated_change(val_change, temperature, bias), val_prev)
            acc = np.mean(np.argmax(q, axis=1) + 1 == windows["val"]["y"])
            loss = -np.mean(np.log(np.maximum(q[np.arange(len(q)), ys["val"]], 1e-12)))
            candidate = (acc, -loss, -abs(temperature - 1), -abs(bias), temperature, bias)
            if best is None or candidate > best:
                best = candidate
    temperature, bias = best[-2], best[-1]

    metrics = []
    prediction_rows = []
    for split in ("val", "test"):
        raw = probabilities(xs[split], multiclass)
        chg = calibrated_change(probabilities(xs[split], binary)[:, 1], temperature, bias)
        fused = persistence_aware(raw, chg, windows[split]["prev"].astype(int))
        res = score(fused, windows[split]["y"].astype(int))
        pdata = windows[split]
        persistence = pdata["prev"].astype(int)
        persistence_top1 = float(np.mean(persistence == pdata["y"]))
        pred = res["top"][:, 0]
        regret, n_power = power_regret(root, scenario, pdata["target_index"], pred)
        p_regret, _ = power_regret(root, scenario, pdata["target_index"], persistence)
        metrics.append({
            "split": split, "model": "sensor_only_next_beam_persistence_aware",
            "top1_accuracy": res["top1_accuracy"], "top3_accuracy": res["top3_accuracy"],
            "top5_accuracy": res["top5_accuracy"], "macro_f1": res["macro_f1"],
            "mean_power_regret_db": regret, "samples_with_power": n_power,
            "training_seconds": training_seconds if split == "test" else np.nan,
            "persistence_top1_same_windows": persistence_top1,
            "persistence_mean_power_regret_db": p_regret,
        })
        if split == "test":
            for i in range(len(pred)):
                prediction_rows.append({
                    "seq_index": int(pdata["seq"][i]), "target_index": int(pdata["target_index"][i]),
                    "previous_beam": int(persistence[i]), "true_next_beam": int(pdata["y"][i]),
                    "predicted_next_beam": int(pred[i]), "p_change": float(chg[i]),
                    "top3_beams": " ".join(map(str, res["top"][i, :3])),
                    "top5_beams": " ".join(map(str, res["top"][i, :5])),
                })
        print(f"{split}: top1={res['top1_accuracy']:.4f}, top3={res['top3_accuracy']:.4f}, "
              f"top5={res['top5_accuracy']:.4f}, persistence_top1={persistence_top1:.4f}, "
              f"power_regret={regret:.4f} dB")

    pd.DataFrame(metrics).to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(out / "test_predictions.csv", index=False)
    np.savez_compressed(out / "sensor_linear_model.npz", multiclass=multiclass, change=binary,
                        mean=mu, std=sd, median=med, temperature=temperature, bias=bias)
    summary = {
        "experiment": "sensor-only future-beam pilot",
        "target": "beam at t+1 after 8 within-sequence logged samples",
        "features": FEATURES,
        "split_policy": "existing sequence-level train/val/test split; scaler fit on train only",
        "train_windows": int(len(ys["train"])), "val_windows": int(len(ys["val"])),
        "test_windows": int(len(ys["test"])), "epochs": int(args.epochs),
        "training_seconds": training_seconds,
        "validation_tuned_change_temperature": float(temperature),
        "validation_tuned_change_bias": float(bias),
        "scope_note": "This is a CPU sensor-only pilot, not the proposed camera+sensor gated-GRU model.",
    }
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved pilot outputs to {out}")


if __name__ == "__main__":
    main()
