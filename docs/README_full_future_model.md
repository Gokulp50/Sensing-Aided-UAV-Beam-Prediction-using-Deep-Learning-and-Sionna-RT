# Full next-beam multimodal run

The separate script `temporal_reliability_gated_future.py` implements the proposed camera-plus-sensor model. It uses eight past RGB frames and all 11 synchronized telemetry features to predict the next logged beam. The 11,387 images are encoded once with a frozen ImageNet ResNet-50; cached embeddings make later reruns faster. A learned gate combines image and sensor features per frame, modality dropout improves tolerance to a weak input stream, and a GRU models the eight-frame history. A change head mixes the previous-beam prior with the learned distribution over alternative beams.

The script retains the existing sequence-level split, fits sensor normalization on train rows only, selects/calibrates on validation, and evaluates the test set once against persistence on identical future windows. It records top-1/3/5, macro-F1, mean received-power regret, speed/altitude subgroup accuracy, predictions, and gate weights. It does not change the project source or its existing results/checkpoints.

## Run from the PBL VS Code terminal

Use the Python environment you just confirmed (`torch 2.12.0+cu126`, `torchvision 0.27.0+cu126`, CUDA available):

```powershell
python "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\temporal_reliability_gated_future.py" `
  --project-root "C:\Users\gokul\OneDrive\Documents\Documents\Electronics\7th sem\ANTENNA 5G\PBL\UAV_BeamPrediction" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_multimodal_run_01"
```

The first run prints frame-feature progress, then training loss and validation metrics each epoch. The local pretrained ResNet-50 weights are already cached on your machine, so the script should not need to download them. If you interrupt after frame-feature extraction completes, the cache files remain in the run folder; rerun with that folder to reuse them. Once a completed run has `best_model.pt` or `metrics.csv`, choose a new output folder for another run.

After the first run, select a shortlist size using validation received-power regret (rather than tuning from test):

```powershell
python "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\evaluate_validation_shortlists.py" --project-root "C:\Users\gokul\OneDrive\Documents\Documents\Electronics\7th sem\ANTENNA 5G\PBL\UAV_BeamPrediction" --run-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_multimodal_run_01"
```

To check whether the camera stream adds value relative to sensors alone, run the validation-only modality ablation (no retraining):

```powershell
python "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\evaluate_validation_modalities.py" --project-root "C:\Users\gokul\OneDrive\Documents\Documents\Electronics\7th sem\ANTENNA 5G\PBL\UAV_BeamPrediction" --run-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_multimodal_run_01"
```

If that confirms both streams help, the next optional model run warm-starts from the current checkpoint and trains the temporal/fusion heads with a received-power-aware ranking loss. It selects the epoch and change calibration using validation top-5 power regret. The ResNet-50 stays frozen and uses the existing feature cache.

```powershell
python "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\temporal_power_aware_future.py" --project-root "C:\Users\gokul\OneDrive\Documents\Documents\Electronics\7th sem\ANTENNA 5G\PBL\UAV_BeamPrediction" --feature-cache-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_multimodal_run_01" --init-checkpoint "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_multimodal_run_01\best_model.pt" --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\future_poweraware_run_01"
```

Expected files in the output folder: `metrics.csv`, `condition_metrics.csv`, `test_predictions.csv`, `training_history.csv`, `run_summary.json`, `sensor_scaler.json`, `best_model.pt`, plus the cached ResNet-50 feature files. The validation commands add their corresponding metrics CSVs without retraining.

**Important:** the full multimodal run has not been executed in this Codex session because its controlled shell cannot launch the Windows Store Python interpreter. The script was syntax-checked here; run it in the VS Code terminal with your confirmed interpreter, then share the final `TEST` summary or `metrics.csv` so I can analyze and refine the model based on validation/test evidence.
