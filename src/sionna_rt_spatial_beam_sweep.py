"""Evaluate the illustrative DFT beam ranking at multiple receiver locations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def run(
    output_dir: Path,
    num_beams: int,
    num_positions: int,
    x_start: float,
    x_end: float,
    y: float,
    z: float,
) -> None:
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

    tx = Transmitter(name="spatial_sweep_tx", position=[8.5, 21.0, 27.0])
    scene.add(tx)
    x_positions = np.linspace(x_start, x_end, num_positions)
    rx_positions = [[float(x), y, z] for x in x_positions]
    receivers = []
    for i, position in enumerate(rx_positions):
        rx = Receiver(name=f"spatial_sweep_rx_{i:02d}", position=position)
        scene.add(rx)
        receivers.append(rx)

    # Keep array orientation fixed toward the middle of the sampled receiver line.
    tx.look_at([float((x_start + x_end) / 2), y, z])
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
    if cir.shape[0] != num_positions:
        raise RuntimeError(
            f"Unexpected receiver dimension {cir.shape[0]}; expected {num_positions}"
        )

    element = np.arange(num_beams, dtype=np.float64)
    weight_matrix = np.stack(
        [
            np.exp(-2j * np.pi * element * beam_id / num_beams)
            / np.sqrt(num_beams)
            for beam_id in range(num_beams)
        ],
        axis=1,
    )
    rows: list[dict[str, float | int]] = []
    position_summaries = []

    for rx_index, position in enumerate(rx_positions):
        # CIR dimensions: [rx, rx_ant, tx, tx_ant, path, time].
        h_elements = np.sum(cir[rx_index, 0, 0, :, :, 0], axis=1)
        if h_elements.shape != (num_beams,):
            raise RuntimeError(
                f"Unexpected channel shape {h_elements.shape}; expected {(num_beams,)}"
            )
        beam_amplitudes = h_elements @ weight_matrix
        powers = np.abs(beam_amplitudes) ** 2
        order = np.argsort(powers)[::-1]
        ranks = np.empty(num_beams, dtype=int)
        ranks[order] = np.arange(1, num_beams + 1)
        max_power = float(np.max(powers))
        for beam_id, power_value in enumerate(powers):
            power = float(power_value)
            relative_db = 10.0 * np.log10(max(power, 1e-30) / max(max_power, 1e-30))
            rows.append(
                {
                    "receiver_index": rx_index,
                    "receiver_x_m": position[0],
                    "receiver_y_m": position[1],
                    "receiver_z_m": position[2],
                    "beam_id": beam_id,
                    "rank": int(ranks[beam_id]),
                    "received_power_linear": power,
                    "relative_power_db": float(relative_db),
                }
            )
        position_summaries.append(
            {
                "receiver_index": rx_index,
                "receiver_position_m": position,
                "best_beam_id": int(order[0]),
                "top3_beam_ids": [int(i) for i in order[: min(3, num_beams)]],
                "best_to_second_gap_db": float(
                    10.0
                    * np.log10(
                        max(float(powers[order[0]]), 1e-30)
                        / max(float(powers[order[1]]), 1e-30)
                    )
                ),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sionna_rt_spatial_beam_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "success",
        "scene": "Sionna integrated Munich scene (software demonstration only)",
        "frequency_hz": frequency_hz,
        "array": f"1x{num_beams} transmit ULA; single isotropic receive antenna",
        "codebook": f"{num_beams}-candidate illustrative DFT codebook",
        "transmitter_position_m": [8.5, 21.0, 27.0],
        "fixed_transmitter_orientation": "toward midpoint of receiver line",
        "receiver_positions_m": rx_positions,
        "path_count_dimension": int(cir.shape[-2]),
        "path_delays_shape": list(delays.shape),
        "position_results": position_summaries,
        "note": (
            "Synthetic Munich-scene spatial demonstration only. The codebook, "
            "array assumptions, and coordinates are not aligned to DeepSense "
            "Scenario 23; results are not measured beam accuracy or project results."
        ),
    }
    json_path = output_dir / "sionna_rt_spatial_beam_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved spatial beam scores: {csv_path}")
    print(f"Saved summary: {json_path}")
    print("Best simulated beam by receiver position:")
    for result in position_summaries:
        print(
            f"  rx {result['receiver_index']:>2} at x={result['receiver_position_m'][0]:.2f} m: "
            f"beam {result['best_beam_id']} "
            f"(gap {result['best_to_second_gap_db']:.2f} dB)"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("sionna_rt_spatial_sweep_01"))
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--num-positions", type=int, default=7)
    parser.add_argument("--x-start", type=float, default=42.0)
    parser.add_argument("--x-end", type=float, default=48.0)
    parser.add_argument("--y", type=float, default=90.0)
    parser.add_argument("--z", type=float, default=1.5)
    args = parser.parse_args()
    if args.num_beams < 2:
        parser.error("--num-beams must be at least 2")
    if args.num_positions < 2:
        parser.error("--num-positions must be at least 2")
    run(
        args.output_dir,
        args.num_beams,
        args.num_positions,
        args.x_start,
        args.x_end,
        args.y,
        args.z,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
