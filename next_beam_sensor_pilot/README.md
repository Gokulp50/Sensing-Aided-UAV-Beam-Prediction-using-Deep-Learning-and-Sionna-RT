# Next-beam sensor pilot

This pilot is an early measurement for the revised methodology. It uses eight past sensor rows to predict the next logged beam, keeps whole sequences in the existing train/validation/test split, and fits feature scaling on training windows only. It uses all 11 sensor features and their frame-to-frame changes. The classifier is a small CPU multinomial linear model plus a beam-change head; it does not use camera images or the proposed reliability-gated GRU.

## Test result

There are 1,583 future test windows. The sensor model reached 68.54% top-1, 78.84% top-3, 85.41% top-5, and 0.48083 macro-F1. Persistence on the exact same windows reached 68.60% top-1 and 0.48093 macro-F1. Mean received-power regret was 0.05513 dB for the sensor model and 0.05468 dB for persistence.

The pilot therefore does not establish an improvement over persistence. The top-3/top-5 values are the model's ranked beam candidates; the one-label persistence rule has no directly equivalent ranked list. The older persistence output used a neighboring-beam tolerance for its top-3/top-5 figures, so those values should not be compared directly.

## Reproduce

From PowerShell, using an interpreter with NumPy and pandas:

```powershell
python "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\next_beam_sensor_experiment.py" `
  --project-root "C:\Users\gokul\OneDrive\Documents\Documents\Electronics\7th sem\ANTENNA 5G\PBL\UAV_BeamPrediction" `
  --output-dir "C:\Users\gokul\Documents\Codex\2026-11-01\referenced-chatgpt-conversation-this-is-an\outputs\next_beam_sensor_pilot"
```

The source project is read-only for this run. The script writes only into the selected output directory.

## Files

- `metrics.csv` — validation and test metrics, including persistence comparisons.
- `test_predictions.csv` — next-beam predictions and top candidate beams for each test window.
- `sensor_linear_model.npz` — fitted pilot model and train-only scaling values.
- `run_summary.json` — split sizes and experiment settings.
