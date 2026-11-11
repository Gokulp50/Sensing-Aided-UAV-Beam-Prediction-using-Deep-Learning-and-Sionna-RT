"""Run a Scenario-23-informed (but not geographically/codebook-aligned) sweep.

The project uses 64 zero-based beam classes. This pilot puts a 16-element,
half-wavelength ULA at the base-station-side receiver and uses 64 oversampled
DFT receive weights. The actual measured array geometry and codebook are not
available here, so outputs are assumption-study results, not project metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def run(
    output_dir: Path,
    num_positions: int,
    x_start: float,
    x_end: float,
    y: float,
    z: float,
    tx_position: tuple[float, float, float],
) -> None:
    import sionna.rt
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    # The Munich scene has several ITU material tables that do not cover
    # 60 GHz (e.g., brick ends at 40 GHz). Freeze the scene-loaded material
    # properties before setting the carrier, so Sionna does not call those
    # unsupported frequency-update tables. This is a software/geometry pilot;
    # material attenuation is not calibrated for the nominal 60 GHz band.
    frequency_hz = 60e9
    num_elements = 16
    num_beams = 64
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    frozen_materials = []
    for material in scene.radio_materials.values():
        if material.frequency_update_callback is not None:
            material.frequency_update_callback = None
            frozen_materials.append(material.name)
    scene.frequency = frequency_hz
    # Scenario-23-informed placement: array is at the stationary base-station
    # receiver; the drone-side transmitter is modeled as a single isotropic port.
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        pattern="iso",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=num_elements,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )

    tx = Transmitter(name="scenario23_assumed_drone_tx", position=list(tx_position))
    scene.add(tx)
    x_positions = np.linspace(x_start, x_end, num_positions)
    rx_positions = [[float(x), y, z] for x in x_positions]
    receivers = []
    for i, position in enumerate(rx_positions):
        rx = Receiver(name=f"scenario23_assumed_bs_rx_{i:02d}", position=position)
        scene.add(rx)
        rx.look_at(tx)
        receivers.append(rx)

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
    if cir.shape[0] != num_positions or cir.shape[1] != num_elements:
        raise RuntimeError(
            "Unexpected CIR dimensions; expected receiver and 16-element array "
            f"dimensions {(num_positions, num_elements)}, got {cir.shape[:2]}"
        )

    # Half-wavelength ULA steering vectors sampled at 4x the 16-element DFT
    # resolution. IDs are zero-based to match this project's 64-class target.
    element = np.arange(num_elements, dtype=np.float64)
    spatial_frequencies = -1.0 + 2.0 * np.arange(num_beams) / num_beams
    weights = np.stack(
        [np.exp(1j * np.pi * element * u) / np.sqrt(num_elements)
         for u in spatial_frequencies],
        axis=0,
    )

    rows: list[dict[str, float | int]] = []
    position_summaries = []
    for rx_index, position in enumerate(rx_positions):
        # CIR layout: [rx, rx_ant, tx, tx_ant, path, time]. Sum coherent paths
        # to make a narrowband channel per receive element, then scan weights.
        h_elements = np.sum(cir[rx_index, :, 0, 0, :, 0], axis=1)
        if h_elements.shape != (num_elements,):
            raise RuntimeError(
                f"Unexpected per-element channel {h_elements.shape}; "
                f"expected {(num_elements,)}"
            )
        powers = np.abs(np.conjugate(weights) @ h_elements) ** 2
        order = np.argsort(powers)[::-1]
        ranks = np.empty(num_beams, dtype=int)
        ranks[order] = np.arange(1, num_beams + 1)
        max_power = float(np.max(powers))
        for beam_id, power_value in enumerate(powers):
            power = float(power_value)
            relative_db = 10.0 * np.log10(
                max(power, 1e-30) / max(max_power, 1e-30)
            )
            rows.append({
                "receiver_index": rx_index,
                "receiver_x_m": position[0],
                "receiver_y_m": position[1],
                "receiver_z_m": position[2],
                "beam_id_zero_based": beam_id,
                "spatial_frequency_assumed": float(spatial_frequencies[beam_id]),
                "rank": int(ranks[beam_id]),
                "received_power_linear": power,
                "relative_power_db": float(relative_db),
            })
        position_summaries.append({
            "receiver_index": rx_index,
            "receiver_position_m": position,
            "best_beam_id_zero_based": int(order[0]),
            "top3_beam_ids_zero_based": [int(i) for i in order[:3]],
            "best_to_second_gap_db": float(
                10.0 * np.log10(
                    max(float(powers[order[0]]), 1e-30)
                    / max(float(powers[order[1]]), 1e-30)
                )
            ),
        })

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "sionna_scenario23_assumption_beam_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "status": "success",
        "study_type": "Scenario-23-informed assumptions; not aligned/validated",
        "scene": "Sionna integrated Munich scene; not geographically aligned",
        "frequency_hz": frequency_hz,
        "nominal_project_frequency_hz": 60e9,
        "frequency_note": "Carrier set to nominal 60 GHz. Built-in Munich scene material frequency callbacks were disabled because several ITU material tables (including brick) do not cover 60 GHz; material permittivity/conductivity remain at their scene-load values and are not calibrated for 60 GHz.",
        "materials_with_frozen_frequency_properties": frozen_materials,
        "tx_array": "single isotropic port, representing quasi-omni drone TX",
        "rx_array": "16-element ULA at stationary base-station-side receiver",
        "array_spacing_assumption": "0.5 wavelengths; must be verified against hardware",
        "codebook": "64 zero-based oversampled DFT receive weights; illustrative, not measured codebook",
        "tx_position_m": list(tx_position),
        "rx_positions_m": rx_positions,
        "path_count_dimension": int(cir.shape[-2]),
        "path_delays_shape": list(delays.shape),
        "position_results": position_summaries,
        "project_label_note": (
            "Local search results indicate 64 output classes and zero-based power argmax. "
            "Numeric IDs here are generated by an assumed codebook and do not represent "
            "measured Scenario 23 beams."
        ),
        "limitations": [
            "Not geographically aligned to Scenario 23; Munich geometry is retained.",
            "Actual array element locations, spacing, orientation, and element pattern are unverified.",
            "Actual measured 64-beam phase/codebook weights are unavailable.",
            "Do not report these outputs as measured accuracy or mix them into ML training.",
        ],
    }
    json_path = output_dir / "sionna_scenario23_assumption_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved beam scores: {csv_path}")
    print(f"Saved summary: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-positions", type=int, default=7)
    parser.add_argument("--x-start", type=float, default=42.0)
    parser.add_argument("--x-end", type=float, default=48.0)
    parser.add_argument("--y", type=float, default=90.0)
    parser.add_argument("--z", type=float, default=1.5)
    parser.add_argument("--tx-x", type=float, default=8.5)
    parser.add_argument("--tx-y", type=float, default=21.0)
    parser.add_argument("--tx-z", type=float, default=27.0)
    args = parser.parse_args()
    if args.num_positions < 2:
        parser.error("--num-positions must be at least 2")
    run(
        args.output_dir,
        args.num_positions,
        args.x_start,
        args.x_end,
        args.y,
        args.z,
        (args.tx_x, args.tx_y, args.tx_z),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
