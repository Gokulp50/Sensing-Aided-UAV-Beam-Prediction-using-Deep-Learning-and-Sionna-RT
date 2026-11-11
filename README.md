# Sensing-Aided UAV Beam Prediction

This repository contains the final project code, evaluation scripts, and results for predicting mmWave beams in DeepSense 6G Scenario 23 using multimodal sensing (camera + telemetry) and Sionna RT.

## 📂 Repository Structure
- `src/`: Core Python scripts for model training, evaluation, and Sionna RT sweeps.
- `results/`: Output folders containing `metrics.csv`, predictions, and Sionna JSON summaries.
- `docs/`: Detailed reports, guides, and project summaries.
- `data/`: Codebook data (Note: The raw DeepSense dataset is not included here due to size; see below).

## 🚀 Key Contributions
1. **Sequence-Grouped Cross-Validation:** Replaced standard random splits to prevent temporal data leakage.
2. **Illumination Robustness:** Training-only brightness/contrast augmentation fixed a catastrophic failure (Top-1 jumped from 25.69% to 72.64% in low-light).
3. **Top-5 Scan Efficiency:** Demonstrated that a 5-beam shortlist reduces measured power regret by 76% compared to persistence (0.0165 dB vs 0.0690 dB).
4. **Sionna RT Pilot:** Highlighted the sensitivity of simulated beam rankings to uncalibrated angle conventions.

## 📊 Results Summary
| Task | Method | Top-1 Acc | Top-5 Power Regret |
|---|---|---|---|
| Current-Beam | Gated ResNet-18 (Illum. Aug) | 70.04% | N/A |
| Next-Beam | Persistence Baseline | 69.19% | 0.0690 dB |
| Next-Beam | Power-Aware GRU (Top-5 Scan) | 68.42% | **0.0165 dB** |

## 📖 Documentation
Please refer to the `docs/` folder for complete details:
- [Final Project Report](docs/FINAL_PROJECT_REPORT.md)
- [Sionna RT Pilot Guide](docs/SIONNA_PILOT_GUIDE.md)
- [Results Summary](docs/RESULTS_SUMMARY.md)

## 🛠️ Setup & Reproduction
To run these scripts, you need the DeepSense 6G Scenario 23 dataset and the following environment:
- Python 3.13
- PyTorch 2.12 (with CUDA)
- Sionna RT 2.2.0 (isolated environment recommended)

*Note: The `.venv-sionna` folder and raw datasets are excluded from this repository via `.gitignore`.*
