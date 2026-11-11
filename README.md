# Sensing-Aided UAV Beam Prediction using Deep Learning and Ray Tracing

**Course Project / Final Report**  
**Dataset:** DeepSense 6G Scenario 23  
**Core Focus:** Multimodal Current-Beam Prediction, Temporal Next-Beam Prediction, and Sionna RT Simulation.

---

## 📖 Executive Summary
Millimeter-wave (mmWave) links require precise beam alignment to overcome severe path loss. For highly dynamic UAVs (drones), traditional exhaustive beam sweeping introduces unacceptable latency. This project investigates whether synchronized camera images and UAV telemetry can predict the optimal beam, reducing sweeping overhead. 

We propose a **Multimodal Gated ResNet-18** for current-beam prediction and a **Temporal GRU** for next-beam prediction. We directly address critical gaps in prior literature by introducing strict sequence-grouped validation, illumination robustness training, and a practical Top-K scanning policy.

---

## 🚀 Key Enhancements vs. Prior Work (Team20 & Sensing-Aided 2025)

| Feature | Prior Work | **This Project** |
| :--- | :--- | :--- |
| **Data Split** | Standard 70/30 random split (Risk of temporal data leakage) | **Strict 5-Fold Sequence-Grouped Cross-Validation** |
| **Label Space** | Downsampled to 32 beams (Simplifies task) | **Original 64-beam classification** |
| **Robustness** | Evaluated only on clean data | **Illumination Augmentation (Low-light stress test)** |
| **Evaluation** | Focuses on Top-1 Accuracy | **Top-5 Scan Power Regret (Operational efficiency)** |
| **Simulation** | Purely synthetic or real-world | **Sionna RT Angle-Sensitivity Pilot** |

---

## 📊 Summary of Results

### 1. Current-Beam Prediction (Lighting Robustness)
While the standard model failed catastrophically in low-light (dropping to 25.69% Top-1), our illumination-augmented model maintained high accuracy.
- **Clean Data:** 72.70% Top-1
- **Low-Light Data:** 72.64% Top-1

### 2. Next-Beam Prediction (Top-5 Scanning)
Persistence (guessing the beam stays the same) is a notoriously strong baseline. While our temporal model achieved 68.42% Top-1 (slightly below persistence at 69.19%), single-beam accuracy is misleading for real networks. By scanning the model's **Top-5 candidate beams**, we drastically reduce power regret:

| Policy | Beams Scanned | Mean Power Regret | Within 0.01 dB of Best |
| :--- | :--- | :--- | :--- |
| Persistence Baseline | 1 | 0.06899 dB | 71.61% |
| **Temporal Model Top-5** | **5** | **0.01651 dB** | **94.92%** |

*(Scanning just 7.8% of the beam sweep reduces power regret by 76% compared to persistence).*

### 3. Sionna RT Pilot
We discovered that simulated beam rankings are highly sensitive to uncalibrated antenna angle conventions (shifting from Beam 12 to Beam 33 with a 30° offset). This highlights the need for physical hardware calibration before trusting simulation-to-real mappings.

---

## 📂 Repository Structure

To keep the project organized, the repository is divided into the following folders:

- **`docs/`**: Contains the full written report (`FINAL_PROJECT_REPORT.md`), the `RESULTS_SUMMARY.md`, and Sionna guides.
- **`src/`**: All Python scripts used for training, evaluation, and ray tracing (ResNet, GRU, Sionna).
- **`results/`**: Output folders containing `metrics.csv`, `test_predictions.csv`, and JSON summaries from the five-fold CV and Sionna sweeps.
- **`data/`**: Codebook data extracted from the DeepSense 6G dataset.

*(Note: The `.venv-sionna` virtual environment and raw DeepSense image datasets are excluded from this repository via `.gitignore` due to size limits. See the setup guide below).*

---

## 🛠️ Setup & Reproduction Guide

### 1. Main Environment (PyTorch)
Ensure you have Python 3.13 and PyTorch 2.12 installed. To rerun the real-data models:
```bash
pip install torch torchvision pandas scikit-learn matplotlib
python src/temporal_power_aware_future.py