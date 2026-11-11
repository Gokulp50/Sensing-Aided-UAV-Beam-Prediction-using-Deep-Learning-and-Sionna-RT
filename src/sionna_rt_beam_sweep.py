"""Run a small DFT-codebook beam sweep on a Sionna RT link."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def run(output_dir: Path, num_beams: int) -> None:
    import sionna.rt
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    frequency_hz = 28e9
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=num_beams,
        pattern="tr38901",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )

    tx = Transmitter(name="sweep_tx", position=[8.5, 21.0, 27.0])
    rx = Receiver(name="sweep_rx", position=[45.0, 90.0, 1.5])
    scene.add(tx)
    scene.add(rx)
    tx.look_at(rx)

    paths = PathSolver(deterministic=True)(
        scene=scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        seed=23,
    )
    cir, delays = paths.cir(normalize_delays=False, out_type="numpy")
    cir = np.asarray(cir)
    delays = np.asarray(delays)

    # Expected CIR layout is [rx, rx_ant, tx, tx_ant, path, time]. Sum the
    # coherent path coefficients to form one narrowband channel per TX element.
    h_elements = np.sum(cir[0, 0, 0, :, :, 0], axis=1)
    if h_elements.shape != (num_beams,):
        raise RuntimeError(
            f"Unexpected per-element channel shape {h_elements.shape}; "
            f"expected {(num_beams,)}"
        )

    element = np.arange(num_beams, dtype=np.float64)
    rows: list[dict[str, float | int]] = []
    powers = []
    for beam_id in range(num_beams):
        # Demonstration-only, uniformly spaced DFT codebook across spatial frequency.
        weights = np.exp(-2j * np.pi * element * beam_id / num_beams) / np.sqrt(num_beams)
        power = float(np.abs(np.dot(h_elements, weights)) ** 2)
        powers.append(power)

    max_power = max(powers)
    order = np.argsort(np.asarray(powers))[::-1]
    ranks = np.empty(num_beams, dtype=int)
    ranks[order] = np.arange(1, num_beams + 1)
    for beam_id, power in enumerate(powers):
        relative_db = 10.0 * np.log10(max(power, 1e-30) / max(max_power, 1e-30))
        rows.append({
            "beam_id": beam_id,
            "rank": int(ranks[beam_id]),
            "received_power_linear": power,
            "relative_power_db": float(relative_db),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sionna_rt_beam_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "success",
        "scene": "Sionna integrated Munich scene (software demonstration only)",
        "frequency_hz": frequency_hz,
        "array": f"1x{num_beams} transmit ULA; single isotropic receive antenna",
        "codebook": f"{num_beams}-candidate DFT codebook; illustrative only",
        "transmitter_position_m": [8.5, 21.0, 27.0],
        "receiver_position_m": [45.0, 90.0, 1.5],
        "path_count": int(cir.shape[-2]),
        "path_delays_shape": list(delays.shape),
        "best_beam_id": int(order[0]),
        "top3_beam_ids": [int(i) for i in order[: min(3, num_beams)]],
        "note": (
            "Synthetic codebook and Munich scene are not aligned to DeepSense "
            "Scenario 23. These are simulated relative link powers, not measured "
            "beam accuracy or project results."
        ),
    }
    json_path = output_dir / "sionna_rt_beam_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved beam scores: {csv_path}")
    print(f"Saved summary: {json_path}")
    print("Relative simulated beam powers (dB):")
    for row in sorted(rows, key=lambda item: int(item["rank"])):
        print(f"  beam {row['beam_id']:>2}: rank {row['rank']}, {row['relative_power_db']:.2f} dB")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sionna_rt_beam_sweep_01"))
    parser.add_argument("--num-beams", type=int, default=8)
    args = parser.parse_args()
    if args.num_beams < 2:
        parser.error("--num-beams must be at least 2")
    run(args.output_dir, args.num_beams)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
