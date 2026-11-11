"""Run a small Sionna RT propagation smoke check in an isolated environment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def run_simulation(output_file: Path) -> None:
    import mitsuba as mi
    import numpy as np
    import sionna.rt
    from sionna.rt import PathSolver, PlanarArray, Receiver, Transmitter, load_scene

    # This integrated scene and endpoint placement follow NVIDIA's RT tutorial.
    scene = load_scene(sionna.rt.scene.munich, merge_shapes=True)
    frequency_hz = 28e9
    scene.frequency = frequency_hz
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V",
    )
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross",
    )

    tx = Transmitter(name="pilot_tx", position=[8.5, 21.0, 27.0])
    rx = Receiver(name="pilot_rx", position=[45.0, 90.0, 1.5])
    scene.add(tx)
    scene.add(rx)
    tx.look_at(rx)

    solver = PathSolver(deterministic=True)
    paths = solver(
        scene=scene,
        max_depth=1,
        los=True,
        specular_reflection=True,
        diffuse_reflection=False,
        refraction=False,
        synthetic_array=True,
        seed=23,
    )
    coefficients, delays = paths.cir(normalize_delays=False, out_type="numpy")
    coefficients = np.asarray(coefficients)
    delays = np.asarray(delays)

    # CIR layout: [rx, rx_ant, tx, tx_ant, path, time].
    path_count = int(coefficients.shape[-2])
    result = {
        "status": "success",
        "scene": "Sionna integrated Munich scene (software demonstration only)",
        # Keep the Python scalar used for configuration. Mitsuba may expose
        # scene.frequency as a Dr.Jit scalar, which cannot be cast with float().
        "frequency_hz": frequency_hz,
        "mitsuba_variant": mi.variant(),
        "channel_coefficients_shape": list(coefficients.shape),
        "path_delays_shape": list(delays.shape),
        "path_count": path_count,
        "cir_coefficient_energy_diagnostic": float(np.sum(np.abs(coefficients) ** 2)),
        "note": "Not geographically aligned to DeepSense Scenario 23; not a project accuracy result.",
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_file}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", dest="python_exe", help="Python executable in the isolated Sionna environment")
    parser.add_argument("--output-dir", type=Path, default=Path("sionna_rt_pilot_run_01"))
    args = parser.parse_args()

    output_file = args.output_dir / "sionna_rt_smoke_result.json"
    if args.python_exe:
        script = Path(__file__).resolve()
        command = [args.python_exe, str(script), "--output-dir", str(args.output_dir)]
        return subprocess.run(command, check=False).returncode

    try:
        run_simulation(output_file)
    except Exception as exc:
        print(f"Sionna RT smoke check failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
