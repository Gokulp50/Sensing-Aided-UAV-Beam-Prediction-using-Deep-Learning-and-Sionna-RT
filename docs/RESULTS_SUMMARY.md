# UAV Beam Prediction: Results Summary

## Evaluation setup

- Data: DeepSense 6G Scenario 23, 11,387 synchronized samples across 35 flight sequences.
- Generalization check: five-fold sequence-grouped cross-validation. Every sequence was held out in one fold; sensor normalization and early stopping were fold-specific.
- Important limitation: these folds were run after model exploration on the same dataset. Treat them as exploratory grouped cross-validation, not an untouched external test.
- Current-beam task: predict the beam at time t from synchronized current image/sensor features.
- Future-beam task: predict the next logged beam from eight preceding image/sensor observations.

## Earlier fixed-split baselines

These are the original project checkpoints/results on a 1,623-sample test split. Keep them as historical baselines; their split/model setup is not established as identical to the later grouped-CV protocol.

| Earlier method | Top-1 | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Camera baseline | 21.38% | 45.59% | 60.51% | 0.0645 |
| Multimodal baseline | 47.20% | 82.87% | 93.84% | 0.1603 |
| Position MLP | 7.52% | 25.69% | 40.48% | 0.1586 |
| Persistence baseline | 68.54% | 90.98% | 95.12% | — |

The later fine-tuned ResNet-18 and sequence-grouped experiments are substantially stronger than the original camera and multimodal baselines. Do not present these historical scores as a controlled architecture-only comparison because the training and split protocols may differ.

## What the supplied EDA adds

- The beam distribution is imbalanced, and the transition matrix is concentrated on the diagonal and nearby beams. This supports persistence and local candidate search as meaningful baselines.
- Sequence-level persistence varies widely (roughly 0.0 to 0.87 among the sequences with adjacent transitions). Results therefore depend on which flight sequences are held out; random row splits can overstate generalization.
- Sensor distributions show large spikes around zero and several quantized altitude/height levels. Use sequence-aware evaluation and per-sequence metrics; avoid treating the crowded beam-versus-sensor scatterplots as evidence of a simple causal relationship.
- The supplied plots are useful for exploration, but the beam-index boxplots have unreadable tick density and the beam-change histogram compresses its main mass near zero. Replot with fewer labels and a zoomed 0–10 change range for final reporting.

## Current-beam results

| Method | Pooled Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Previous-beam persistence | 68.95% | — | — |
| Fine-tuned gated ResNet-18 + sensors | 69.54% | 95.88% | 98.58% |

The multimodal model improves pooled Top-1 by about 0.61 percentage points over persistence. Fold Top-1 varies substantially (67.87% mean, 9.26 percentage-point standard deviation), and persistence wins in two of the five folds. The overall gain is modest and should not be described as a large accuracy improvement.

## Future-beam results

| Method | Pooled Top-1 | Top-3 | Top-5 | Top-1 power regret |
|---|---:|---:|---:|---:|
| Previous-beam persistence | 69.19% | — | — | 0.06899 dB |
| Power-aware temporal model | 67.45% | 87.89% | 94.47% | 0.07698 dB |
| Top-1-focused temporal model | 68.42% | 87.14% | 93.72% | 0.09113 dB |

Neither learned future-beam version beats persistence for one-beam Top-1. The Top-1-focused objective recovers about 0.97 percentage points over the power-aware objective, but remains about 0.76 points below persistence.

## Candidate-scan result

For the power-aware model, scanning its five proposed beams and choosing the one with the best measured power gives:

- Mean power regret: **0.01651 dB**.
- Within 0.01 dB of the best beam: **94.92%** of future windows.
- Five scans are 5/64, or about **7.8%**, of the full beam sweep.

For comparison, persistence uses one beam and has 0.06899 dB mean power regret, with 71.61% of windows within 0.01 dB of the best beam. The Top-5 result assumes the candidates are actually measured; it is not the model's Top-1 accuracy.

The previous beam already appears in the power-aware model's Top-5 list for 99.98% of out-of-fold windows. Adding it explicitly to the shortlist therefore did not change the measured Top-5 result in this analysis.

## Exploratory persistence-gated policy

Using the Top-1-focused model's out-of-fold `p_change`, we tested a simple policy: keep the previous beam unless `p_change` exceeds a threshold. For each held-out fold, the threshold was selected using the other four folds. The pooled result was 69.26% Top-1 versus 69.19% for persistence, and the model was selected for only 4.63% of windows. This is a negligible gain and does not establish a better policy; the selected thresholds also varied by fold (0.70–0.75), so probability calibration is not stable enough to claim a robust switch rule.

## Current-beam robustness diagnostic

The fixed current-beam gated ResNet-18 + sensor checkpoint was evaluated on the 1,623-sample held-out test split with controlled input degradation. These are diagnostic results from the existing development split, not an independent robustness benchmark.

| Test condition | Top-1 | Change vs clean | Top-3 | Top-5 |
|---|---:|---:|---:|---:|
| Clean | 70.86% | — | 94.76% | 98.03% |
| GPS noise, 5 m standard deviation | 70.79% | -0.06 pp | 94.76% | 98.03% |
| GPS noise, 15 m standard deviation | 70.73% | -0.12 pp | 94.82% | 98.03% |
| GPS missing | 71.10% | +0.25 pp | 94.76% | 98.03% |
| Motion sensors missing | 70.67% | -0.18 pp | 94.70% | 98.03% |
| All sensors missing | 70.79% | -0.06 pp | 94.39% | 98.03% |
| Gaussian blur, radius 1.5 | 70.79% | -0.06 pp | 94.39% | 97.91% |
| Low light, brightness 0.5 | 25.69% | -45.16 pp | 46.27% | 57.24% |
| Blur plus low light | 14.36% | -56.50 pp | 33.89% | 43.31% |

The standard model was strongly dependent on usable image brightness under this perturbation and showed little change when sensor inputs were removed. The small apparent GPS-missing gain is only 0.25 percentage points and should not be interpreted as a real benefit. This diagnostic motivated stronger brightness/contrast augmentation, evaluated below with both a fixed split and grouped cross-validation.

### Illumination-augmented model: validation and final test comparison

A separate ResNet-18 run used stronger training-only brightness and contrast augmentation. The original checkpoint was preserved. The new candidate was selected at epoch 8 by validation Top-1, without evaluating the test split. Validation robustness was then measured for both checkpoints:

| Validation condition | Standard model Top-1 | Illumination-augmented Top-1 | Difference |
|---|---:|---:|---:|
| Clean | 75.83% | 76.32% | +0.49 pp |
| Low light | 31.78% | 76.07% | +44.29 pp |
| Blur + low light | 29.08% | 76.56% | +47.48 pp |

The candidate was then evaluated once on the held-out test split. The original standard model's test results are included for direct comparison:

| Test condition | Model | Top-1 | Top-3 | Top-5 | Macro-F1 |
|---|---|---:|---:|---:|---:|
| Clean | Standard | 70.86% | 94.76% | 98.03% | 0.438 |
| Clean | Illumination-augmented | 72.70% | 95.26% | 97.78% | 0.414 |
| Low light | Standard | 25.69% | 46.27% | 57.24% | 0.124 |
| Low light | Illumination-augmented | 72.64% | 95.13% | 97.60% | 0.433 |
| Blur + low light | Standard | 14.36% | 33.89% | 43.31% | 0.101 |
| Blur + low light | Illumination-augmented | 72.27% | 95.19% | 97.78% | 0.428 |

On clean test data, Top-1 rose by 1.85 percentage points and Top-3 by 0.49 points, while Top-5 fell by 0.25 points and macro-F1 fell by 0.024. Under low light, Top-1 improved by 46.95 points; under blur plus low light, it improved by 57.92 points. This supports a strong illumination-robustness improvement for this fixed split, with a modest clean-set class-balance/Top-5 tradeoff. The candidate has now also been evaluated with grouped cross-validation; those results are below and show a modest pooled Top-1 gain, not a large accuracy jump.

### Illumination-augmented model: five-fold grouped cross-validation

The robust-profile run used the same five sequence-grouped folds as the standard-profile run, so the pooled scores can be compared on matched held-out sequences. Each sequence appears in a held-out fold once. These folds are still exploratory because model and augmentation choices were developed using this dataset.

| Current-beam model | Pooled OOF Top-1 | Top-3 | Top-5 | Mean fold Top-1 (SD) |
|---|---:|---:|---:|---:|
| Fine-tuned gated ResNet-18, standard augmentation | 69.54% | 95.88% | 98.58% | 67.87% (9.26 pp) |
| Fine-tuned gated ResNet-18, illumination-robust augmentation | 70.04% | 93.33% | 98.36% | 71.10% (6.67 pp) |
| Change | **+0.49 pp** | **-2.56 pp** | **-0.22 pp** | **+3.23 pp mean; 2.59 pp lower SD** |

On the 11,352 OOF samples with a preceding beam, the illumination-robust model reached 70.05% Top-1 versus 68.95% for persistence, a +1.10 percentage-point difference. Fold-level outcomes were mixed: the robust model's held-out Top-1 ranged from 64.66% to 81.10%, and its Top-1 power-regret mean was 0.0899 dB with a 0.0964 dB fold standard deviation. One fold had 0.2574 dB Top-1 regret, so the mean power result is not uniformly strong. The robust profile therefore modestly improves pooled Top-1 and reduces observed fold variation, but trades away Top-3/Top-5 coverage and does not win on every sequence group. Report the full metric set rather than claiming general accuracy improved across the board.

## Recommended interpretation

- For a single next-beam choice, use persistence as the current reference policy; neither learned future model has shown a reliable Top-1 gain.
- When five candidate measurements are acceptable, the power-aware model can provide a useful shortlist with much lower power regret than using persistence alone.
- For current-beam prediction, the gated ResNet-18 gives a small pooled Top-1 gain over persistence, with strong Top-3/Top-5 coverage.
- The confidence-gated persistence pilot did not materially beat persistence. Do not promote it as a project improvement; report it as an explored policy that did not show useful gain.
- Keep the current-beam and future-beam metrics separate. Their targets and prediction horizons differ.
- Sionna was not used in any reported real-data experiment. Sionna RT 2.2.0, Mitsuba 3.9.1, and Dr.Jit 1.5.0 are installed in the isolated Python 3.13.14 environment. The initial Munich-scene sweeps used 28 GHz and a 1x8 transmit-ULA DFT codebook; beam 0 stayed first along the seven-point line, with a 12.76–22.12 dB best-to-second gap. The Scenario 23 paper/repository audit establishes a 60 GHz link, a 16-element phased array at stationary Unit 1, quasi-omni drone transmission, and 64 measured beam-power values; the paper's learning workflow also reduces these to 32 labels by selecting every other beam. The user's recursive code search confirms their local training scripts use 64 classes, `unit1_beam_index`, and zero-based power argmax. Actual array geometry and measured codebook weights remain unavailable. The separate 60 GHz, 16-element base-station receive-array assumption sweep now runs in the unaligned Munich scene: beam 32 ranked first at each of seven assumed positions, with roughly 0.90 dB gap to beam 31/33. To avoid unsupported Munich ITU material ranges, frequency callbacks were disabled and scene-load material constants retained; these material losses are not calibrated at 60 GHz. The output is a software/geometry pilot, not measured accuracy or evidence of improvement on Scenario 23. Keep all Sionna outputs separate from the frozen real-data evaluation. ([Scenario 23 paper](https://par.nsf.gov/servlets/purl/10404319); [repository README](https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction/blob/main/README.md); [label-generation code](https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction/blob/main/downsample_pwr_gen_train_test_data.py))

## Remaining project work

1. Prepare final result tables/plots and report the paired five-fold results, the fixed-split robustness results, and the Top-1 vs Top-K trade-off with their limitations.
2. Replot the crowded exploratory figures with clearer axes and a zoomed beam-change distribution; retain the existing plots as exploratory outputs.
3. **Current Sionna stage:** obtain Scenario 23's actual codebook weights, array geometry/orientation, and base-station coordinate frame. Then replace the Munich assumptions before attempting a project-aligned Sionna scene or measured-label comparison. The current assumption sweep is complete as a software pilot only.
4. User prepares the presentation; no slide file was edited as part of this work.
