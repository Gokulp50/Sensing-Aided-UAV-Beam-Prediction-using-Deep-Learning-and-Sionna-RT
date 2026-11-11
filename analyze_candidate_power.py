"""Measure the best measured beam within each model shortlist on a saved run."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    data_root = args.project_root.resolve() / "data" / "scenario23_dev"
    run_dir = args.run_dir.resolve()
    pred = pd.read_csv(run_dir / "test_predictions.csv")
    scenario = pd.read_csv(data_root / "scenario23.csv").set_index("index")
    records=[]
    for row in pred.itertuples(index=False):
        rel = str(scenario.loc[int(row.target_index), "unit1_pwr_60ghz"]).replace("\\", "/").removeprefix("./")
        power = np.atleast_1d(np.loadtxt(data_root / rel, dtype=np.float64))
        power = np.maximum(power, 1e-15)
        best_power = float(power.max())
        candidates = {
            "persistence_1": [int(row.previous_beam)],
            "model_top1": [int(row.predicted_next_beam)],
            "model_top3": list(map(int, str(row.top3_beams).split())),
            "model_top5": list(map(int, str(row.top5_beams).split())),
        }
        record = {"seq_index": int(row.seq_index), "target_index": int(row.target_index),
                  "true_next_beam": int(row.true_next_beam)}
        for name, beams in candidates.items():
            valid = [b for b in beams if 1 <= b <= len(power)]
            if not valid:
                record[f"{name}_regret_db"] = np.nan
                record[f"{name}_selected_beam"] = np.nan
                continue
            selected = max(valid, key=lambda b: power[b-1])
            record[f"{name}_regret_db"] = float(10*np.log10(best_power / float(power[selected-1])))
            record[f"{name}_selected_beam"] = selected
        records.append(record)
    per_sample = pd.DataFrame(records)
    per_sample.to_csv(run_dir / "candidate_power_predictions.csv", index=False)
    rows=[]
    candidates = [
        ("persistence_1", 1), ("model_top1", 1), ("model_top3", 3), ("model_top5", 5)
    ]
    for name, count in candidates:
        r = per_sample[f"{name}_regret_db"].dropna().to_numpy()
        rows.append({"candidate_policy": name, "beams_scanned": count,
                     "fraction_of_64_beams_scanned": count/64,
                     "mean_regret_db": float(np.mean(r)),
                     "median_regret_db": float(np.median(r)),
                     "zero_regret_fraction": float(np.mean(r < 1e-10)),
                     "evaluated_samples": int(len(r))})
    pd.DataFrame(rows).to_csv(run_dir / "candidate_power_metrics.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"Saved shortlist power analysis to {run_dir}")


if __name__ == "__main__":
    main()
