# Sionna RT pilot for UAV beam prediction

## Why this pilot

Sionna RT is a differentiable radio-propagation ray tracer. It can compute propagation paths, channel impulse/frequency responses, and radio maps for a configured 3D scene. This is a simulation extension to the DeepSense 6G Scenario 23 study; it does not directly improve or validate the real-data model.

The pilot is deliberately staged:

1. Confirm that Sionna RT imports and can compute a small channel in its built-in Munich scene.
2. Save a reproducible result record and check GPU backend selection.
3. Only after the smoke run works, decide whether to obtain/build 3D geometry aligned to the Scenario 23 flight area and match its antenna/codebook assumptions.
4. Generate simulated beam rankings only after the scene and beam codebook are defensible. Keep those results separate from the real-data test and grouped-CV results.
5. Consider synthetic-data augmentation only if the simulated and recorded labels have compatible definitions and the real-data evaluation remains sequence-grouped and untouched during tuning.

The built-in Munich scene is only a software demonstration. It is not geographically aligned to Scenario 23 and cannot support a claim about Scenario 23 performance.

## Isolated setup on Windows

Keep Sionna in a separate virtual environment so the existing PyTorch 2.12/CUDA project environment is left alone. Mitsuba 3 supports Windows; its current docs list Python 3.10+ and NVIDIA driver 535+ for GPU compute. Sionna PHY/SYS have separate requirements, but this pilot only installs the standalone RT package.

**Setup status:** Sionna RT 2.2.0, Mitsuba 3.9.1, and Dr.Jit 1.5.0 installed successfully in `.venv-sionna` with Python 3.13.14. The smoke check completed successfully: it selected `cuda_ad_mono_polarized`, traced four paths at 28 GHz, and saved its JSON diagnostic. The console also printed `jitc_llvm_init(): LLVM API initialization failed ..`; because the run completed and produced channel coefficients, this warning did not prevent this smoke check. Watch for a fatal backend error on subsequent runs.

In PowerShell, check the Python version that works in your current project terminal. It must be Python 3.10 or newer:

```powershell
python --version
```

If the version is 3.10 or newer, create the isolated environment in this workspace and install the standalone package:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
python -m venv $PilotEnv
& "$PilotEnv\Scripts\python.exe" -m pip install --upgrade pip
& "$PilotEnv\Scripts\python.exe" -m pip install sionna-rt
```

Do not change the Python environment used for the existing project runs. The RTX 3050 can be used for a small pilot, though scene size and ray counts should stay modest because it has 6 GB VRAM.

## First run

Run the supplied smoke-check script directly with the isolated interpreter:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
& "$PilotEnv\Scripts\python.exe" "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_smoke_check.py" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_pilot_run_01"
```

The script loads the built-in Munich scene, places one transmitter and one receiver using coordinates from NVIDIA's introductory tutorial, computes first-order propagation paths and a channel impulse response, then saves `sionna_rt_smoke_result.json`. The JSON reports the Mitsuba variant, channel dimensions, path count, and a simple channel-coefficient energy diagnostic. This is a software smoke check, not a beam-selection score.

If the output says the Mitsuba variant uses `cuda`, the GPU backend was selected. A CPU backend still demonstrates the simulator works, but we should investigate the graphics driver/backend before scaling the pilot. Paste the full error if the run fails; do not reinstall or modify the existing project environment.

## Next pilot: illustrative DFT beam sweep

The smoke check establishes that a small channel can be computed. The next script forms an illustrative 8-candidate DFT codebook on a transmit ULA, computes each candidate's simulated narrowband received power, and saves the relative beam ranking. It uses the same built-in Munich scene and tutorial transmitter/receiver locations. The antenna/codebook are demonstration assumptions, not the Scenario 23 beam codebook; these scores are not real-data accuracy or a claim of improved project performance.

**Observed result (2026-11-02):** the deterministic run traced four paths. Beam 0 ranked first (0.00 dB relative), followed by beam 1 (-19.27 dB) and beam 5 (-19.54 dB). The remaining candidates ranged from -23.48 to -35.29 dB. This is one simulated link at one receiver location, so it only confirms the sweep and ranking calculation execute; it does not show spatial coverage or prediction accuracy. The `jitc_llvm_init()` warning appeared again without preventing the run from saving its outputs.

Run it in the same isolated environment:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
& "$PilotEnv\Scripts\python.exe" "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_beam_sweep.py" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_beam_sweep_01" `
  --num-beams 8
```

Expected files are `sionna_rt_beam_scores.csv` and `sionna_rt_beam_sweep_summary.json` in the output folder. Share the terminal output if the run fails; if it succeeds, share the summary and the printed relative powers so we can interpret the result before attempting any Scenario 23 alignment.

## Next research step

The spatial extension also completed (2026-11-02). Across seven receivers from x=42 m to x=48 m at fixed y=90 m and z=1.5 m, beam 0 ranked first at every point. The runner-up varied among beams 1, 2, 5, and 7; the best-to-second power gap ranged from 12.76 to 22.12 dB. This only indicates a stable top candidate along this short synthetic line under the chosen fixed array/codebook assumptions. It does not establish that beam 0 maps to any Scenario 23 label.

The next Sionna-to-project task is an alignment audit, not another accuracy claim. Confirm from the Scenario 23 data/code and project preprocessing: (1) what each logged beam ID represents, (2) the measured beam codebook and antenna geometry/orientation, (3) carrier frequency, and (4) the base-station position and coordinate frame relative to the UAV GPS. The built-in Munich scene is not a substitute for those details. Until the values and coordinate transform are defensible, keep all Sionna outputs as a separate simulator demonstration and do not mix synthetic samples into training.

### Alignment facts established from the Scenario 23 paper and repository

The Scenario 23 testbed description reports a 60 GHz link, a 16-element phased array at stationary Unit 1 (the receiver/base-station side), and a quasi-omnidirectional antenna on the drone transmitter. It records received power across a 64-beam oversampled codebook. The paper's beam-prediction experiments reduce that 64-entry power vector to 32 entries by keeping every other beam, then use the maximum-power entry as the label. The original repository supports both the original 64-beam label and the downsampled 32-beam label. These are different target definitions and their IDs must not be mixed. ([Scenario 23 paper](https://par.nsf.gov/servlets/purl/10404319); [repository README](https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction/blob/main/README.md); [label-generation code](https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction/blob/main/downsample_pwr_gen_train_test_data.py))

**Local-project audit from the user's recursive search (2026-11-02):** training scripts `06_train_sensor_baseline.py`, `11_train_multimodal_baseline.py`, and `12_train_temporal_multimodal.py` define 64 classes. The label column is `unit1_beam_index`; `04_test_dataset.py` describes the power argmax as zero-based. The current project is therefore using the original 64-label branch with IDs intended to be 0–63, rather than the paper's downsampled 32-class target. The local command output is the evidence for the project convention; a direct min/max check of the CSV is still a useful final sanity check.

This narrows the next simulation design: configure the radio link at 60 GHz and put the 16-element array on the stationary base-station side. It does **not** yet define the array spacing, element pattern, orientation, measured beam weights, or exact local base-station coordinates. The existing 28 GHz, 1x8 transmit-ULA DFT sweeps therefore remain software demonstrations; their `beam 0` cannot be mapped to any Scenario 23 beam ID.

The local target convention is now identified as 64 classes with zero-based labels. The separate script `sionna_rt_scenario23_assumption_sweep.py` uses nominal 60 GHz, a 16-element base-station-side receive ULA, half-wavelength spacing, and an illustrative 64-candidate oversampled DFT receive codebook. Several built-in Munich-scene ITU material tables do not cover 60 GHz (including brick, which ends at 40 GHz), so the script disables their frequency callbacks and retains the scene-load material constants. This makes it a software/geometry pilot only; material losses are not calibrated for 60 GHz. The geometry, orientation, codebook, and coordinates also remain assumptions, so its beam IDs cannot be compared against measured Scenario 23 labels.

Run this separate pilot using the isolated environment. It will write a 64-beam score CSV and a JSON summary into a new output folder:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
& "$PilotEnv\Scripts\python.exe" "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_scenario23_assumption_sweep.py" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_scenario23_assumption_sweep_03"
```

The run completed. Beam 32 ranked first at all seven assumed receiver positions; beams 31 and 33 were the runners-up, about 0.90 dB behind. The output CSV and JSON are in `sionna_scenario23_assumption_sweep_03`. This consistency checks the small synthetic setup only. Do not interpret the printed `beam_id_zero_based` values as actual Scenario 23 beam labels; the Munich scene, codebook, and coordinates are not aligned, and its material losses are not calibrated at 60 GHz.

## What is needed to connect Sionna to Scenario 23

- A 3D scene covering the real flight area, with a coordinate transform from the dataset's GPS coordinates into local metric coordinates.
- Base-station location/height, carrier frequency, antenna array geometry/pattern, and the actual measured beam-codebook definition.
- UAV position/orientation over time aligned to the Scenario 23 timestamps.
- A documented comparison between Sionna's simulated strongest-beam labels and the dataset's measured beam/power labels. Any mismatch must be reported rather than silently mapped.

Known from the paper and local-code search: 60 GHz, 16 phased-array elements on stationary Unit 1, drone-side quasi-omni transmission, 64 measured beam powers, and local use of 64 zero-based labels; the cited paper workflow also reports a 32-class every-other-beam reduction. Still to confirm: actual codebook weights, array element geometry/orientation, and coordinate transform.

Until these are available and checked, Sionna outputs stay a separate simulator demonstration. No synthetic samples are to be added to the frozen real-data evaluation.

### Testbed 4 measured-pattern lookup (2026-11-02)

The user identified the downloaded beam-codebook bundle as **Testbed 4** and asked that we use it. The archive contains 64 files, `built_in_beam_pattern_0.mat` through `built_in_beam_pattern_63.mat`, each with 211 angular samples. The arrays have been converted into `sionna_measured_codebook_testbed4.csv` with zero-based beam IDs, angle in degrees, and the stored pattern value. The associated hardware page screenshot describes Testbed 4 as a 60 GHz, 16-element ULA with a 64-beam codebook.

These files provide measured angular-pattern samples, not complex weights for 16 individual antenna elements. The new Sionna script therefore uses each pattern as a relative-power lookup over simulated arrival azimuths; it does not install them as Sionna array weights. It folds azimuth into the supplied 0–180-degree cut, interpolates the pattern value for each path, and sums path powers noncoherently. The angle offset and sign are exposed as command-line options because the codebook cut's mounting and orientation convention have not yet been calibrated. The elevation response is also omitted.

Run the separate sweep in the isolated Sionna environment:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
& "$PilotEnv\Scripts\python.exe" "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_testbed4_pattern_sweep.py" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_testbed4_pattern_sweep_01"
```

It should save `sionna_testbed4_pattern_beam_scores.csv` and `sionna_testbed4_pattern_sweep_summary.json`. If needed, angle-convention sensitivity can be explored by passing `--angle-offset-deg` and `--angle-sign`; those are assumptions until verified against the antenna mounting. The scene remains Munich, its coordinates do not match Scenario 23, and its material properties are not calibrated at 60 GHz. Keep the results separate from measured accuracy and do not feed them into model training. Before interpreting a best-beam ID as a Scenario 23 prediction, check the codebook's angle reference and beam ordering against the receiver and the recorded power-vector labels.

**Observed Testbed 4 pattern-sweep result (2026-11-02):** the user's run completed and saved outputs to `sionna_testbed4_pattern_sweep_01`. Across receiver x positions 42–48 m, the leading IDs were 12, 12, 10, 10, 9, 9, 9. The top-two margins were 0.1445, 0.1021, 0.0236, 0.0517, 0.0860, 0.3515, and 0.4366 dB. These are near ties, so the leading ID is sensitive to the uncalibrated angular mapping and must not be described as a reliable Scenario 23 beam prediction. The console's `jitc_llvm_init()` line was non-fatal for this run: the process completed and wrote the summary and score CSV.

**Next: test angular-convention sensitivity.** The zero-offset, positive-sign run above is the reference. Run the following two cases into new folders and compare the top-beam IDs and top-two gaps in each JSON/CSV. This is a sensitivity check, not a search for whichever mapping gives the preferred result:

```powershell
$PilotEnv = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\.venv-sionna"
$Script = "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_rt_testbed4_pattern_sweep.py"
& "$PilotEnv\Scripts\python.exe" $Script `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_testbed4_angle_plus30" `
  --angle-offset-deg 30
& "$PilotEnv\Scripts\python.exe" $Script `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\sionna_testbed4_angle_minus30" `
  --angle-offset-deg -30
```

After those, check sign sensitivity with `--angle-sign -1` in a third new output directory, retaining the default zero-degree offset. If the rankings or margins shift materially across these plausible conventions, the pilot demonstrates that orientation calibration is required; it does not identify the physically correct convention. Use hardware mounting metadata or a known receiver-angle reference to choose that convention.

**Observed minus-30-degree case (2026-11-02):** the user's run completed and saved to `sionna_testbed4_angle_minus30`. The leading IDs across x=42–48 m were 8, 4, 4, 4, 4, 8, 8; top-two gaps were 0.0025–0.1366 dB. Compared with the zero-offset IDs (12, 12, 10, 10, 9, 9, 9), the ranking changes substantially and remains almost tied. This confirms the simulated ranking is highly sensitive to the uncalibrated angular convention. It does not determine which offset is physically correct.

**Observed plus-30-degree case (2026-11-02):** the user's run completed and saved to `sionna_testbed4_angle_plus30`; the JSON confirms `angle_offset_deg: 30.0`. The leading IDs across x=42–48 m were 33, 31, 31, 31, 30, 30, 30, with top-two gaps of 0.0087–0.4598 dB. Thus the three tested offsets produce different leading IDs (0°: 12/12/10/10/9/9/9; −30°: 8/4/4/4/4/8/8; +30°: 33/31/31/31/30/30/30), and every case has very small top-two margins. The rankings are not robust to angle convention and cannot be used as calibrated beam predictions.

**Corrected sign-flip result (2026-11-02):** the rerun in `sionna_testbed4_angle_sign_minus_v2` includes `negative_sign_behavior`, confirming it used the corrected implementation. Its leading IDs across x=42–48 m are 49, 49, 50, 52, 52, 53, 53; top-two gaps range from 0.0135 to 0.1789 dB. Compared with the zero-offset positive-sign IDs (12, 12, 10, 10, 9, 9, 9), the ranking changes substantially, while the top candidates remain near-tied. This confirms sensitivity to the assumed direction mapping; it does not establish which sign is physically correct. The sign-sensitivity run is complete. No further sign/offset sweep is needed; calibration requires physical mounting or coordinate-frame information.

**Codebook provenance check:** the screenshot of the DeepSense data page identifies Testbed 4 as a 60 GHz, 16-element ULA with a 64-beam codebook ([DeepSense data-collection page](https://deepsense6g.net/data-collection)). However, the downloaded bundle's own `plot_codebook_beams.m` header calls it “DeepSense 6G - Testbed 1 Beam Patterns.” This conflicts with the user's download-page identification. Until the source/archive association is verified, refer to the local pattern table as the *user-supplied 64-beam pattern bundle* and do not claim its beam ordering is confirmed as Testbed 4 or map these IDs to Scenario 23 labels. The plotted angular cut spans 0–180° in the README, with a 4° offset applied in the MATLAB plotting script; this still does not establish the hardware mounting convention.

Source references: [DeepSense data-collection page](https://deepsense6g.net/data-collection); [Scenario 23 paper](https://par.nsf.gov/servlets/purl/10404319).

## References

- [Sionna RT introduction and tutorial](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html)
- [Sionna package installation and version requirements](https://github.com/NVlabs/sionna)
- [Mitsuba 3 installation and system requirements](https://mitsuba.readthedocs.io/en/latest/)
