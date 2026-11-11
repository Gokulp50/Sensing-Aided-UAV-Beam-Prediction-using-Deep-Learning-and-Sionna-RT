"""Rank beams using the supplied 64-beam angular-pattern table and Sionna paths.

This is a simulation-only pattern lookup. It treats the CSV values as relative
power gains in a one-dimensional azimuth cut and sums path powers incoherently.
The angle reference, elevation response, measured antenna mounting, scene, and
coordinates are not calibrated to Scenario 23.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def load_codebook(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    by_beam: dict[int, list[tuple[float, float]]] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"beam_id_zero_based", "angle_degrees", "pattern_value"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Codebook CSV must contain columns {sorted(required)}")
        for row in reader:
            beam = int(row["beam_id_zero_based"])
            angle = float(row["angle_degrees"])
            value = float(row["pattern_value"])
            if not np.isfinite(angle) or not np.isfinite(value) or value < 0:
                raise ValueError(f"Invalid pattern row: {row}")
            by_beam.setdefault(beam, []).append((angle, value))

    expected = set(range(64))
    if set(by_beam) != expected:
        raise ValueError(
            f"Expected zero-based beam IDs 0..63; found {sorted(by_beam)}"
        )
    angle_rows = []
    value_rows = []
    for beam in range(64):
        pairs = sorted(by_beam[beam])
        angles = np.asarray([p[0] for p in pairs], dtype=np.float64)
        values = np.asarray([p[1] for p in pairs], dtype=np.float64)
        if len(angles) < 2 or np.any(np.diff(angles) <= 0):
            raise ValueError(f"Beam {beam} needs unique ascending angle samples")
        if beam and not np.allclose(angles, angle_rows[0], atol=1e-5):
            raise ValueError("All beams must use the same angle grid")
        angle_rows.append(angles)
        value_rows.append(values)
    return np.arange(64, dtype=np.int64), angle_rows[0], np.stack(value_rows)


def extract_path_vector(value: np.ndarray, receiver_index: int) -> np.ndarray:
    """Extract one receiver's path vector from either Sionna angle layout."""
    value = np.asarray(value)
    if value.ndim == 3:  # [rx, tx, path] with synthetic arrays
        return np.asarray(value[receiver_index, 0, :]).reshape(-1)
    if value.ndim == 5:  # [rx, rx_ant, tx, tx_ant, path]
        return np.asarray(value[receiver_index, 0, 0, 0, :]).reshape(-1)
    raise RuntimeError(f"Unsupported Sionna path-angle shape: {value.shape}")


def run(args: argparse.Namespace) -> None:
    import sionna.rt
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    beam_ids, pattern_angles, pattern_values = load_codebook(args.codebook)
    frequency_hz = 60e9
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)

    # The built-in Munich materials do not all define ITU properties at 60 GHz.
    # Freeze their loaded parameters so setting the nominal carrier can proceed.
    frozen_materials = []
    for material in scene.radio_materials.values():
        if material.frequency_update_callback is not None:
            material.frequency_update_callback = None
            frozen_materials.append(material.name)
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )
    scene.rx_array = PlanarArray(
        num_rows=1, num_cols=1, pattern="iso", polarization="V"
    )

    tx = Transmitter(name="testbed4_assumed_drone_tx", position=args.tx_position)
    scene.add(tx)
    x_positions = np.linspace(args.x_start, args.x_end, args.num_positions)
    rx_positions = [[float(x), args.y, args.z] for x in x_positions]
    for i, position in enumerate(rx_positions):
        scene.add(Receiver(name=f"testbed4_assumed_bs_rx_{i:02d}", position=position))

    paths = PathSolver(deterministic=True)(
        scene=scene,
        max_depth=args.max_depth,
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
    # Mitsuba/Dr.Jit tensors expose NumPy conversion as a .numpy() method in
    # the installed Sionna RT environment.
    phi_r = np.asarray(paths.phi_r.numpy())

    if cir.shape[0] != args.num_positions:
        raise RuntimeError(f"CIR receiver dimension mismatch: {cir.shape}")
    angle_layout = phi_r.shape
    rows: list[dict[str, float | int]] = []
    position_summaries = []
    for rx_index, position in enumerate(rx_positions):
        angles_rad = extract_path_vector(phi_r, rx_index)
        # With a single-antenna synthetic receiver, select one complex
        # coefficient per propagation path at time sample zero.
        h_paths = np.asarray(cir[rx_index, 0, 0, 0, :, 0]).reshape(-1)
        if len(angles_rad) != len(h_paths):
            raise RuntimeError(
                f"Path angle/coefficient count mismatch: {len(angles_rad)} vs {len(h_paths)}"
            )
        valid = np.isfinite(angles_rad) & np.isfinite(h_paths) & (np.abs(h_paths) > 0)
        angles_rad = angles_rad[valid]
        path_power = np.abs(h_paths[valid]) ** 2
        if not len(path_power):
            raise RuntimeError(f"No valid propagation paths for receiver {rx_index}")

        # Sionna's phi_r is the arrival azimuth. The codebook file is a single
        # angular cut, so fold azimuth into 0..180 and expose orientation as an
        # explicit user-controlled offset. Values outside the measured grid are
        # clamped by np.interp to its measured endpoint.
        phi_deg = np.degrees(angles_rad)
        cut_angle = (phi_deg + args.angle_offset_deg) % 360.0
        cut_angle = np.where(cut_angle > 180.0, 360.0 - cut_angle, cut_angle)
        # A sign flip before folding is mathematically erased by the fold:
        # fold(-phi) == fold(phi). Apply direction reversal inside the measured
        # 0..180-degree cut instead, as a reflection about its 90-degree center.
        if args.angle_sign < 0:
            cut_angle = 180.0 - cut_angle
        gains = np.stack(
            [np.interp(cut_angle, pattern_angles, row) for row in pattern_values],
            axis=0,
        )
        powers = gains @ path_power
        order = np.argsort(powers)[::-1]
        ranks = np.empty(64, dtype=np.int64)
        ranks[order] = np.arange(1, 65)
        best = max(float(powers[order[0]]), 1e-30)
        for beam_id, power in enumerate(powers):
            rows.append({
                "receiver_index": rx_index,
                "receiver_x_m": position[0],
                "receiver_y_m": position[1],
                "receiver_z_m": position[2],
                "beam_id_zero_based": int(beam_ids[beam_id]),
                "rank": int(ranks[beam_id]),
                "pattern_weighted_power_relative": float(power),
                "relative_power_db": float(10.0 * np.log10(max(float(power), 1e-30) / best)),
            })
        position_summaries.append({
            "receiver_index": rx_index,
            "receiver_position_m": position,
            "valid_path_count": int(len(path_power)),
            "best_beam_id_zero_based": int(order[0]),
            "top3_beam_ids_zero_based": [int(i) for i in order[:3]],
            "best_to_second_gap_db": float(
                10.0 * np.log10(
                    max(float(powers[order[0]]), 1e-30)
                    / max(float(powers[order[1]]), 1e-30)
                )
            ),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "sionna_testbed4_pattern_beam_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "success",
        "study_type": "Sionna simulation with supplied Testbed 4 angular-pattern table; assumption study only",
        "scene": "Sionna integrated Munich scene; not Scenario 23 geometry",
        "frequency_hz": frequency_hz,
        "codebook_csv": str(args.codebook.resolve()),
        "codebook_beams": 64,
        "angle_samples_per_beam": int(len(pattern_angles)),
        "codebook_pattern_interpretation": "relative power gain in one angular cut, used as a lookup; not complex array weights",
        "angle_mapping": {
            "source": "Sionna paths.phi_r arrival azimuth",
            "angle_sign": args.angle_sign,
            "negative_sign_behavior": "reflect folded measured-cut angle as 180 degrees minus angle",
            "angle_offset_deg": args.angle_offset_deg,
            "folded_to_measured_cut_deg": [float(pattern_angles[0]), float(pattern_angles[-1])],
            "out_of_range_behavior": "clamp to measured angular endpoints",
        },
        "power_estimator": "sum of per-path Sionna coefficient powers times interpolated measured pattern values; paths combined noncoherently",
        "array_model": "single isotropic ports; measured patterns applied as external receive-beam lookup",
        "transmitter_position_m": args.tx_position,
        "receiver_positions_m": rx_positions,
        "materials_with_frozen_frequency_properties": frozen_materials,
        "path_angle_tensor_shape": list(angle_layout),
        "path_delays_shape": list(delays.shape),
        "position_results": position_summaries,
        "limitations": [
            "Munich scene and coordinates are not aligned to Scenario 23.",
            "The angular cut's boresight, azimuth direction, and mounting orientation have not been calibrated; use --angle-offset-deg only for sensitivity analysis until verified.",
            "Elevation response is omitted; this is a one-dimensional angular-pattern lookup.",
            "The pattern samples are treated as relative power gains and path powers are combined noncoherently.",
            "Do not report these beam rankings as measured accuracy or use them as Scenario 23 labels without confirming codebook provenance and beam ordering.",
        ],
    }
    json_path = args.output_dir / "sionna_testbed4_pattern_sweep_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved beam scores: {csv_path}")
    print(f"Saved summary: {json_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codebook", type=Path, default=Path(__file__).with_name("sionna_measured_codebook_testbed4.csv"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-positions", type=int, default=7)
    parser.add_argument("--x-start", type=float, default=42.0)
    parser.add_argument("--x-end", type=float, default=48.0)
    parser.add_argument("--y", type=float, default=90.0)
    parser.add_argument("--z", type=float, default=1.5)
    parser.add_argument("--tx-position", type=float, nargs=3, default=[8.5, 21.0, 27.0], metavar=("X", "Y", "Z"))
    parser.add_argument("--angle-offset-deg", type=float, default=0.0)
    parser.add_argument("--angle-sign", type=float, choices=(-1.0, 1.0), default=1.0)
    parser.add_argument("--max-depth", type=int, default=1)
    args = parser.parse_args()
    if args.num_positions < 2:
        parser.error("--num-positions must be at least 2")
    if args.max_depth < 0:
        parser.error("--max-depth must be nonnegative")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
