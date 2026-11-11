# Sensing-Aided Beam Prediction for UAV Communications

## Final project report

**Dataset:** DeepSense 6G, Scenario 23  
**Project focus:** Current-beam and next-beam prediction using camera and UAV sensor data  
**Simulation extension:** Sionna RT pilot, reported separately from the real-data experiments  
**Report status:** Final experimental findings assembled from the runs and logs supplied in this project thread

---

## Abstract

Millimeter-wave links can lose received power when a UAV moves and the currently selected beam no longer points along a strong propagation direction. This project evaluates whether synchronized camera images and UAV sensor measurements can help predict a suitable beam. The work uses DeepSense 6G Scenario 23, containing synchronized image, position, motion, and beam-power observations across 35 flight sequences. Two different tasks are studied: predicting the beam associated with the current observation, and predicting the next logged beam from the preceding eight observations.

The evaluation emphasizes sequence-aware generalization. In exploratory five-fold sequence-grouped cross-validation, the illumination-augmented gated ResNet-18 current-beam model achieves 70.04% pooled Top-1 accuracy, compared with 69.54% for the standard-augmentation model and 68.95% for previous-beam persistence on comparable samples. This is a modest improvement, not a large accuracy jump. For next-beam prediction, neither learned single-beam model beats persistence at Top-1. A power-aware temporal model does, however, produce a useful Top-5 candidate list: measuring the five candidates and selecting the strongest measured beam gives 0.01651 dB mean power regret, versus 0.06899 dB for selecting only the previous beam. An illumination stress test also shows that brightness and contrast augmentation greatly improves performance under the particular synthetic low-light transformations used here. These are controlled perturbation results, not measurements from real nighttime flights.

A separate Sionna RT pilot explored beam rankings in an unaligned Munich scene using assumed array geometry and a user-provided angular pattern table. The rankings were sensitive to angle conventions and remained simulation-only. They are not Scenario 23 accuracy results and were not used to train or evaluate the real-data models.

---

## 1. Introduction

Millimeter-wave (mmWave) systems use directional beams to compensate for propagation loss. A beam that is suitable at one moment may become suboptimal as a UAV changes position, orientation, or surrounding propagation conditions. Exhaustively measuring every beam can add overhead. A sensing-aided predictor aims to narrow the search using information already available from a camera and onboard sensors.

This project studies that idea on real-world multimodal data. It also distinguishes two questions that are sometimes conflated:

1. **Current-beam classification:** Given the image and sensor observation at time *t*, predict the beam label recorded at time *t*.
2. **Next-beam prediction:** Given the preceding temporal observations, predict the beam label at the next time step.

These tasks have different inputs, targets, and practical interpretations. Their scores are reported separately throughout this report.

### 1.1 Project objectives

- Establish simple persistence and modality baselines.
- Train image and sensor models for the current-beam task.
- Test whether illumination augmentation improves sensitivity to image degradation.
- Evaluate temporal models for next-beam prediction against persistence.
- Report accuracy and beam-power regret, including the trade-off from measuring a Top-K shortlist.
- Run a limited Sionna RT software pilot without presenting simulated values as real-data model results.

### 1.2 Main findings

- The current-beam model's grouped-CV Top-1 gain over persistence is small: **+0.61 percentage points** for standard augmentation and **+1.10 points** for illumination-robust augmentation on samples with a previous beam.
- The illumination-robust model's pooled Top-1 is slightly higher than standard augmentation, while its Top-3 and Top-5 are slightly lower. The metrics do not support saying that every accuracy measure improved.
- For next-beam prediction, persistence remains the best evaluated one-beam Top-1 reference.
- A power-aware Top-5 shortlist can substantially reduce measured-power regret when five candidate beams can be probed.
- The user's codebook bundle has a provenance discrepancy: the download-page image identifies Testbed 4, while a plotting script in the bundle labels its pattern data Testbed 1. Its precise association must therefore be described cautiously.

### 1.3 Project contributions and novelty positioning

The work's defensible novelty is **project-level and empirical**: it combines a sequence-aware comparison, a targeted robustness study, and a practical shortlist/power-regret evaluation on this Scenario 23 setup. It does not establish a new neural-network architecture, a new propagation model, or a state-of-the-art result. A claim of formal research novelty would require a broader literature review and an independent evaluation set.

The project-specific contributions are:

1. **Separate, sequence-aware task evaluation:** current-beam classification and next-beam prediction are defined and measured separately, with whole flight sequences held out and persistence used as the reference.
2. **Illumination robustness analysis:** a failure under synthetic low-light degradation motivated training-only brightness/contrast augmentation; the candidate improves strongly under the tested transformations, while grouped-CV Top-1 gains remain modest and Top-K trade-offs are reported.
3. **Operational Top-5 scan analysis:** beyond classification accuracy, the work quantifies what happens when the predicted five-beam shortlist is actually measured, including scan fraction, received-power regret, and the within-0.01-dB rate.
4. **Simulation kept distinct from measurements:** the Sionna RT pilot examines codebook/angle sensitivity while explicitly separating assumed simulation rankings from Scenario 23 measured-data performance.

These contributions should be presented as the project's methodology and evaluation findings, not as proof that the underlying model architecture is novel compared with all published work.

---

## 2. Dataset and prediction targets

### 2.1 Dataset

The project uses **DeepSense 6G Scenario 23**, with 11,387 synchronized samples organized into 35 flight sequences. The available modalities used in this work are RGB camera images, position and motion-related sensor values, and a received-power vector over 64 beam candidates. The 11 numerical sensor features used in the project are latitude, longitude, speed, altitude, distance, height, x-speed, y-speed, z-speed, pitch, and roll.

The DeepSense collection description for Testbed 4 identifies a 60 GHz phased array with a 16-element ULA, a 64-beam codebook, and 960 × 540 RGB images. The related Scenario 23 paper also describes the 60 GHz drone communication testbed and beam-power measurements. The paper's own learning workflow uses every other beam to reduce the classification set to 32 labels, whereas the local project scripts use all 64 beam labels. This distinction matters when comparing published numbers with this project's scores. See the [DeepSense data-collection page](https://deepsense6g.net/data-collection) and [Sensing-Aided 6G Drone Communications: Real-World Datasets and Demonstration](https://doi.org/10.1109/OJCOMS.2025.3612320).

### 2.2 Beam-label construction

For the local experiments, the target is the zero-based index of the maximum value in the 64-entry received-power vector (`unit1_beam_index`). In practical terms, the model is trained to identify which logged beam has the highest recorded power for the sample. It is therefore a **beam-index classification task**, with the measured beam-power vector also available for regret calculations.

### 2.3 Task A: current-beam prediction

The model receives a synchronized image and/or the sensor values at time *t* and predicts the beam label associated with that same observation. The principal learned model is a fine-tuned ResNet-18 for image features combined with the sensor branch through a learned gate. The direct persistence comparison uses the preceding beam where one exists.

### 2.4 Task B: next-beam prediction

The temporal model receives eight preceding multimodal observations and predicts the next logged beam. Persistence predicts that the next beam will remain equal to the most recently observed beam. This is a more deployment-oriented temporal question than current-beam classification, but it is still evaluated against labels recorded in the dataset and does not simulate a live radio controller.

### 2.5 Dataset properties relevant to interpretation

Exploratory analysis found an imbalanced beam distribution and a transition matrix concentrated on the diagonal and nearby beams. Many adjacent observations therefore keep the same or a nearby beam, which makes persistence a strong baseline. Sequence-level persistence also varies substantially; reported analysis found values from roughly 0.0 to 0.87 among sequences with adjacent transitions. A random row split could place near-duplicate frames from the same flight in both training and evaluation data, so this report gives priority to sequence-grouped evaluation.

The sensor plots also show quantized or spiky values for some variables. Scatterplots and boxplots are useful for exploring the data but do not establish a causal relationship between a sensor variable and beam choice.

---

## 3. Data preparation and evaluation design

### 3.1 Preprocessing and model selection

The project workflow aligns image, sensor, and beam observations by sample/time; builds the beam target from the maximum-power beam index; scales numerical inputs using training data; and preserves flight-sequence boundaries when constructing train, validation, and test groups. Image models use ImageNet-pretrained ResNet-18 weights. The temporal workflow uses sliding sequences of length eight; its camera features are cached to avoid recomputing the image encoder for every temporal training epoch.

For the fixed split, the grouped assignment contained 26 training sequences, 4 validation sequences, and 5 test sequences. It yielded 8,134 train rows, 1,630 validation rows, and 1,623 test rows for the current-beam experiment. Temporal windows are fewer because a sequence needs a preceding history before a future target can be formed.

Model checkpoints were selected using validation performance. The test split was not used to repeatedly maximize clean Top-1 accuracy or choose an epoch. However, the initial test-set robustness diagnostic revealed the low-light failure and motivated the illumination-augmentation experiment. Therefore, the later fixed-split augmentation comparison is **not a fully untouched final test**: the test-set condition helped shape the next experiment, even though that candidate was selected using validation performance. The five-fold results are out-of-fold predictions using sequence groups, but model and augmentation choices were explored using this same dataset before the final cross-validation runs. Accordingly, these are **exploratory grouped cross-validation results**, not an untouched external validation.

### 3.2 Metrics

- **Top-1 accuracy:** fraction of examples whose highest-ranked predicted beam is the target.
- **Top-3 / Top-5 accuracy:** fraction where the target occurs among the three or five highest-ranked candidates.
- **Macro-F1:** class-wise F1 averaged across represented classes, giving less frequent classes more influence than raw accuracy.
- **Power regret (dB):** difference in received power between the best recorded beam and the selected beam; lower is better. A Top-K scan regret assumes the candidate beams are actually probed and the strongest measured candidate is then selected.
- **Within 0.01 dB:** fraction of examples for which the chosen beam's measured power is within 0.01 dB of the best recorded beam.

Top-K accuracy is not the same as Top-K scan performance: Top-K accuracy asks whether the target is in the candidate list; scan regret assumes the candidates are physically measured and the best one is selected.

### 3.3 Evaluation groups

This report separates results into three groups:

1. **Original project baselines:** historical fixed-split results, whose protocol equivalence to later experiments is not verified.
2. **Controlled fixed-split comparisons and robustness:** use the existing grouped 26/4/5 train/validation/test assignment.
3. **Exploratory grouped cross-validation:** pooled out-of-fold predictions across five sequence-held-out folds; useful for sequence-aware comparison, but not external validation because the dataset informed model development.

---

## 4. Baselines and original project results

The original project had camera-only, multimodal, position-only, and persistence results on a 1,623-sample test split. They are retained here as historical context. These numbers should not be treated as an architecture-only controlled comparison against later models because the split and training protocol are not established as identical.

| Historical method | Top-1 | Top-3 | Top-5 | Macro-F1 |
|---|---:|---:|---:|---:|
| Camera baseline | 21.38% | 45.59% | 60.51% | 0.0645 |
| Multimodal baseline | 47.20% | 82.87% | 93.84% | 0.1603 |
| Position-only MLP | 7.52% | 25.69% | 40.48% | 0.1586 |
| Previous-beam persistence | 68.54% | 90.98% | 95.12% | — |

The unusually strong persistence result is consistent with the observed high rate of repeated or nearby beam labels. It is why later learned models are judged against persistence rather than only against the original neural baselines.

---

## 5. Current-beam prediction results

### 5.1 Fixed-split model development

The fixed-split ResNet-18 experiment improved substantially over the original camera-only baseline. Adding sensor fusion gave a further improvement in the best fixed-split run. The learned fusion gate assigned an average of 97.15% of its weight to the visual branch and 2.85% to sensors on that run. This is evidence that the trained model relied primarily on its image branch in that configuration; it does not establish that sensors are generally unhelpful.

| Fixed-split method | Test Top-1 | Top-3 | Top-5 | Macro-F1 | Top-1 power regret |
|---|---:|---:|---:|---:|---:|
| Fine-tuned ResNet-18, image | 69.50% | 92.85% | 96.73% | 0.398 | 0.0500 dB |
| Gated ResNet-18 + sensors | 70.86% | 94.76% | 98.03% | 0.438 | 0.0381 dB |
| Persistence, comparable subset | 68.54% | — | — | — | — |

The gated model's Top-1 is 2.32 percentage points above persistence on this fixed split. The persistence comparison has 1,618 samples with a preceding beam, while the learned model has 1,623 test samples; the exact comparison is therefore on the comparable subset when calculating the paired number.

### 5.2 Five-fold sequence-grouped results

The grouped-CV numbers provide a more sequence-aware comparison. They show a smaller gain than the fixed-split test result.

| Current-beam method | Pooled OOF Top-1 | Top-3 | Top-5 | Mean fold Top-1 (SD) |
|---|---:|---:|---:|---:|
| Previous-beam persistence | 68.95% | — | — | — |
| Gated ResNet-18, standard augmentation | 69.54% | 95.88% | 98.58% | 67.87% (9.26 pp) |
| Gated ResNet-18, illumination-robust augmentation | 70.04% | 93.33% | 98.36% | 71.10% (6.67 pp) |

Relative to the standard model, illumination-robust augmentation raised pooled Top-1 by 0.49 points and reduced the observed fold Top-1 standard deviation by 2.59 points. Top-3 fell by 2.56 points and Top-5 fell by 0.22 points. On the 11,352 out-of-fold samples with a preceding beam, illumination-robust Top-1 was 70.05% versus 68.95% for persistence, a 1.10-point increase. The model did not win on every sequence group, and its fold-level results varied.

**Interpretation:** the project demonstrates a modest current-beam Top-1 improvement in grouped evaluation, not a large or universal accuracy improvement. The Top-1-focused claim should be accompanied by Top-3, Top-5, fold variation, and the development-data limitation.

---

## 6. Robustness and illumination augmentation

### 6.1 Standard model degradation diagnostic

The standard fixed-split checkpoint was evaluated with controlled synthetic input perturbations. Sensor and mild blur perturbations changed Top-1 little, while low-light transformations caused a large drop.

| Test condition | Top-1 | Top-3 | Top-5 |
|---|---:|---:|---:|
| Clean | 70.86% | 94.76% | 98.03% |
| GPS noise, 5 m standard deviation | 70.79% | 94.76% | 98.03% |
| GPS noise, 15 m standard deviation | 70.73% | 94.82% | 98.03% |
| GPS missing | 71.10% | 94.76% | 98.03% |
| Motion sensors missing | 70.67% | 94.70% | 98.03% |
| All sensors missing | 70.79% | 94.39% | 98.03% |
| Gaussian blur, radius 1.5 | 70.79% | 94.39% | 97.91% |
| Low-light transform, brightness 0.5 | 25.69% | 46.27% | 57.24% |
| Blur plus low-light transform | 14.36% | 33.89% | 43.31% |

The small increase under GPS-missing input is within the observed evaluation variation and should not be described as a benefit. These transformations are diagnostic and do not reproduce a calibrated camera or real nighttime flight.

### 6.2 Illumination-augmented candidate

A separate candidate was trained with stronger training-only brightness and contrast augmentation, selected using validation Top-1, and evaluated on the held-out test split. The original standard checkpoint was preserved.

| Test condition | Standard model Top-1 | Illumination-augmented Top-1 | Augmented Top-3 | Augmented Top-5 | Augmented Macro-F1 |
|---|---:|---:|---:|---:|---:|
| Clean | 70.86% | 72.70% | 95.26% | 97.78% | 0.414 |
| Low light | 25.69% | 72.64% | 95.13% | 97.60% | 0.433 |
| Blur plus low light | 14.36% | 72.27% | 95.19% | 97.78% | 0.428 |

On clean test data, the candidate gained 1.85 points in Top-1 and 0.49 points in Top-3, while Top-5 fell by 0.25 points and macro-F1 fell from 0.438 to 0.414. Under the synthetic low-light transform, Top-1 improved by 46.95 points; under blur plus low light, it improved by 57.92 points. The grouped-CV result is more modest and mixed, as shown in Section 5.2.

**Interpretation:** illumination augmentation is the clearest robustness advancement in the current-beam experiments. It supports robustness to the particular image transformations used. It does not prove performance under real night, weather, exposure, or camera hardware changes.

---

## 7. Next-beam prediction and scan efficiency

### 7.1 Five-fold grouped results

| Next-beam method | Pooled OOF Top-1 | Top-3 | Top-5 | Top-1 power regret |
|---|---:|---:|---:|---:|
| Previous-beam persistence | 69.19% | — | — | 0.06899 dB |
| Power-aware temporal model | 67.45% | 87.89% | 94.47% | 0.07698 dB |
| Top-1-focused temporal model | 68.42% | 87.14% | 93.72% | 0.09113 dB |

Neither learned model beats persistence for the one-beam Top-1 decision. The Top-1-focused objective is 0.97 points above the power-aware model, but still 0.76 points below persistence. Its Top-1 power regret is higher, so the Top-1 ranking result does not translate into a better received-power decision in this comparison.

### 7.2 Top-5 candidate scan

The power-aware temporal model's practical result is its shortlist. If the system measures the five proposed beams and chooses the strongest measured candidate, the grouped out-of-fold evaluation gives:

| Policy | Beams measured | Fraction of 64-beam sweep | Mean power regret | Within 0.01 dB of best |
|---|---:|---:|---:|---:|
| Previous-beam persistence | 1 | 1.56% | 0.06899 dB | 71.61% |
| Power-aware model Top-5 shortlist | 5 | 7.81% | 0.01651 dB | 94.92% |

This result assumes the five candidate beams are actually measured; it does not mean the model's Top-1 prediction has 94.92% accuracy. The five-beam scan uses 5/64 of the full search. It trades additional probing over persistence for lower measured-power regret.

### 7.3 Persistence-gated policy pilot

A fold-held-out pilot kept the previous beam unless a Top-1-focused model estimated a sufficiently high probability of a beam change. The pooled result was 69.26% Top-1 versus 69.19% for persistence, a 0.07-point difference; the learned model was selected for only 4.63% of windows. The thresholds varied by fold. This is a negligible gain and is not presented as a successful policy improvement.

---

## 8. Sionna RT simulation pilot

### 8.1 Purpose and separation from the real-data experiments

Sionna RT was installed in an isolated Python environment and used to verify that a basic ray-tracing and beam-ranking workflow runs. The official [Sionna RT documentation](https://nvlabs.github.io/sionna/rt/index.html) describes it as a radio-propagation and ray-tracing tool. The pilot did not use Scenario 23's measured images or measured beam powers to generate its results. No simulated records were added to model training, and none of the Sionna rankings contribute to the accuracy tables above.

The software smoke test and exploratory sweeps used the integrated Munich scene. Earlier checks used an illustrative 28 GHz link and an 8-beam DFT codebook. Later assumption sweeps used a 60 GHz, 16-element receive-array assumption and a 64-beam codebook. A separate analysis used the user-provided 64-beam angular-pattern table as a relative-power lookup against simulated arrival azimuths. The table contains angular patterns, not complex element weights, and it does not by itself specify the actual antenna array or propagation geometry.

### 8.2 Sensitivity to angle assumptions

Across seven assumed receiver positions, the pattern-table sweep returned the following top beam IDs under different angular conventions:

| Assumed mapping | Top beam IDs across seven points | Largest reported best-to-second gap |
|---|---|---:|
| 0° offset, positive sign | 12, 12, 10, 10, 9, 9, 9 | 0.44 dB |
| −30° offset | 8, 4, 4, 4, 4, 8, 8 | 0.14 dB |
| +30° offset | 33, 31, 31, 31, 30, 30, 30 | 0.46 dB |
| Corrected negative-sign mapping | 49, 49, 50, 52, 52, 53, 53 | 0.18 dB |

The top rankings shift substantially when the angle convention changes, and the top-two power gaps are small. These outputs show that codebook interpretation depends on a correctly known coordinate frame and antenna orientation. They do not validate a real beam predictor.

### 8.3 Provenance and physical-alignment limitations

The supplied download-page screenshot identifies **Testbed 4** hardware as 60 GHz, 16-element ULA, and 64-beam codebook. However, the downloaded bundle's `plot_codebook_beams.m` labels its patterns as **Testbed 1**. Until the original source package or documentation resolves this mismatch, the table should be described as a user-provided beam-pattern bundle whose Testbed 4 provenance is unverified.

The Munich scene is not geographically aligned with Scenario 23. Physical mounting orientation, coordinate frames, array element response, complex codebook weights, elevation response, and calibrated 60 GHz material parameters are not established in this pilot. The printed Dr.Jit LLVM initialization warning was non-fatal in the successful smoke runs, but successful software execution does not make the physical setup calibrated.

**Sionna conclusion:** the software pilot is complete and useful as a feasibility demonstration. It did not improve or validate the Scenario 23 real-data accuracy. Further Sionna-to-real comparison should wait until codebook provenance and coordinate/antenna metadata can be verified.

---

## 9. Relation to prior work

The project is motivated by published work on sensing-aided drone communication and vision-aided beam prediction. The Scenario 23 demonstration paper describes the real-world multimodal testbed and reports its own learning protocol, including reduction from 64 measured beams to 32 labels by selecting every other beam. This project instead uses the local scripts' 64-class, zero-based maximum-power labels and sequence-grouped splits. Consequently, a published accuracy figure should not be placed next to this report's accuracy as if the experiment were reproduced under the same conditions.

Earlier work such as [Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction](https://arxiv.org/abs/2205.12187) provides related context for camera/position-based prediction. Differences in dataset subset, label construction, number of classes, split design, sequence leakage controls, image processing, and prediction horizon make direct score ranking unreliable. The appropriate comparison here is to the project's own baselines under its documented protocol.

The methodological contribution of this project should therefore be framed as an **experimental comparison and robustness study on Scenario 23**, with a practical Top-5 scan-efficiency analysis and a separately scoped simulation pilot. It should not claim a new radio propagation model or a state-of-the-art accuracy result.

---

## 10. Limitations and threats to validity

1. **Limited number of independent sequences:** the dataset has 35 flight sequences; the fixed test group contains five. Performance can vary depending on which routes are held out.
2. **Exploratory cross-validation:** the models and augmentation profiles were developed using this same dataset. The five-fold grouped results reduce row-level leakage but are not a final external benchmark.
3. **Class imbalance:** accuracy can be dominated by frequent beams. Macro-F1 and per-class support are relevant, and the illumination-robust model's Top-1 gain comes with lower Top-3/Top-5 in grouped CV.
4. **Strong persistence baseline:** beam labels commonly repeat or change locally. A model can look accurate without outperforming simply retaining the previous beam.
5. **Artificial image perturbations:** low-light, blur, and sensor-noise conditions are software transformations, not controlled real-world camera or flight tests.
6. **Potential dataset dependence:** sensor quantization and sequence-level structure may limit transfer to different hardware, flight routes, or environments.
7. **Power regret interpretation:** regret is calculated from the dataset's measured beam powers. The Top-5 scan result assumes those candidate beams are probed and cannot be interpreted as Top-1 classification accuracy.
8. **Simulation alignment:** the Sionna scene, coordinates, array response, material parameters, and angular mapping are assumptions; beam-pattern bundle provenance is unresolved.
9. **No live deployment measurements:** latency, power consumption, controller integration, and live-radio scan overhead were not measured.
10. **Not an exact paper reproduction:** the local 64-class labels, grouped split, selected model, and training protocol differ from the related paper's described 32-class workflow.

---

## 11. Conclusion

This project established separate current-beam and next-beam evaluations on DeepSense 6G Scenario 23, compared learned models with persistence, tested illumination robustness, and ran a bounded Sionna RT feasibility pilot.

For **current-beam classification**, the illumination-robust gated ResNet-18 obtains 70.04% pooled grouped-CV Top-1 versus 68.95% persistence on comparable samples. The gain is real in the recorded evaluation but modest, and Top-3/Top-5 slightly decline relative to the standard-augmentation model. The strongest fixed-split result is 72.70% clean Top-1, but the grouped-CV result is the more appropriate sequence-aware headline and must be described as exploratory.

For **next-beam prediction**, neither learned temporal model beats persistence for Top-1. The power-aware model's Top-5 shortlist is the practical result: a five-beam scan yields 0.01651 dB mean regret and is within 0.01 dB of the best beam in 94.92% of windows, compared with 0.06899 dB and 71.61% for one-beam persistence. This makes the project's efficiency result more compelling than a claim of single-beam accuracy improvement.

The illumination-augmentation experiment substantially improves results under the synthetic low-light transformations. It demonstrates that a known visual failure mode can be addressed through training augmentation, while leaving real-world low-light validation as future work. Sionna RT verifies that a simulation workflow can run, but the current scene and beam mappings are not aligned well enough to support a real-data accuracy claim.

The final defensible message is: **the project achieves a modest sequence-aware current-beam Top-1 gain, a strong synthetic illumination-robustness gain, and a useful Top-5 beam-search regret reduction; it does not yet establish a next-beam Top-1 gain over persistence or a calibrated Sionna-to-Scenario-23 result.**

---

## 12. Final project status and remaining deliverables

| Workstream | Status | What can be stated in the final project |
|---|---|---|
| Scenario 23 data inspection and beam/sensor EDA | Complete | Data characteristics motivate sequence-aware evaluation and a persistence reference. |
| Original and improved current-beam models | Complete | Grouped-CV Top-1 improvement is modest; report the full Top-K and fold variability. |
| Illumination robustness | Complete for synthetic transforms | Large improvement under the tested low-light transforms; real camera-condition validation remains untested. |
| Future-beam temporal models | Complete | Persistence remains stronger for one-beam Top-1. |
| Top-5 scan/power analysis | Complete | Five-candidate scan materially reduces measured power regret in grouped OOF evaluation. |
| Sionna RT software pilot | Complete as simulation feasibility work | Report separately, with assumptions and unresolved codebook provenance. |
| Additional model tuning against completed test split | Closed | No further tuning should use the completed test results. The initial low-light test diagnostic did motivate the augmentation experiment, so the fixed-split augmentation comparison is not fully untouched; grouped-CV results are exploratory as well. A clean final claim requires new held-out flights or a predeclared independent evaluation protocol. |
| Presentation edits | User-owned | This report is separate; no presentation changes are required to use it. |

No additional model-testing run is required to support this report's conclusions. The remaining project work is to select the figures and tables for the presentation, use the caveats in this report, and resolve the codebook provenance only if a physically aligned Sionna comparison is still desired.

---

## References

1. DeepSense 6G, [Data Collection and Testbed Information](https://deepsense6g.net/data-collection).
2. [Sensing-Aided 6G Drone Communications: Real-World Datasets and Demonstration](https://doi.org/10.1109/OJCOMS.2025.3612320), IEEE Open Journal of the Communications Society, 2025.
3. [Sionna RT documentation](https://nvlabs.github.io/sionna/rt/index.html), NVIDIA.
4. [Sionna RT: Introduction tutorial](https://nvlabs.github.io/sionna/rt/tutorials/Introduction.html), NVIDIA.
5. [Towards Real-World 6G Drone Communication: Position and Camera Aided Beam Prediction](https://arxiv.org/abs/2205.12187), 2022.
6. [Project repository](https://github.com/gourangc/Sensing-Aided-Drone-Beam-Prediction).

## Local records used to assemble the results

- `RESULTS_SUMMARY.md` — run metrics, grouped-CV comparisons, robustness results, and interpretation notes.
- `PROJECT_ROADMAP.md` — workstream status and evaluation protocol.
- User-provided console logs — fixed-split fine-tuning, robustness, and Sionna run outputs.
- `METHODOLOGY_DIAGRAM.svg` — standalone, slide-ready methodology flowchart; it is not embedded into or applied to the presentation.

