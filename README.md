# CNN Deep Features with Classical Machine Learning Classifiers for Lithography Hotspot Detection

---

## Overview

This repository contains the experiment pipeline, raw outputs, result tables, figures, and reproducibility notes for the study *“CNN Deep Features with Classical Machine Learning Classifiers for Lithography Hotspot Detection.”*

**Research Question:** Can features learned by a CNN and subsequently classified using classical machine-learning models improve lithography hotspot detection compared with direct end-to-end CNN classification on the ICCAD-12 benchmark suite?

The study evaluates five ICCAD-12 benchmarks separately. It compares a lightweight reference CNN against a proposed residual depthwise-separable CNN, and examines the effects of backbone, loss function, feature depth, and classical classifier. The experiment grid covers **5 benchmarks × 2 backbones × 2 losses × 5 seeds = 100 run configurations**.

> **Main result:** In the current five-seed results, the reference CNN has higher average balanced accuracy than the proposed CNN trained end-to-end. Classical heads are useful as an ablation, but the results do not support a blanket claim that CNN features with classical classifiers outperform the reference baseline.

## Repository Structure

```text
.
├── ICCAD12_CNN_Classical_ML_Reproducible.ipynb
├── README.md
├── RELEASE_MANIFEST.json
├── da3_raw.zip
├── da3_tables_figures.zip
├── outputs/
│   ├── config.json
│   └── README_reproducibility.md
├── results/
│   ├── tables/                 # CSV tables from the experiment outputs
│   └── figures/                # Existing and supplementary result figures
└── report/
    └── figures/                # Report-ready figure copies and architectures
```

The raw archive is retained as `da3_raw.zip`. Because it is large, consider publishing it as a GitHub Release asset and linking it from this README rather than committing it to the main Git history.

## Dataset

**ICCAD-12 benchmark suite** — five separate binary classification benchmarks for lithography hotspot (HS) versus non-hotspot (NHS) detection.

The benchmarks are evaluated independently rather than merged for the primary experiments. The project uses a training/validation/test workflow: hyperparameters and decision thresholds are selected using validation data, and final metrics are computed on the held-out test split.

**Dataset note:** The dataset copy used for these experiments was obtained from a shared download source rather than independently verified against the official contest release. The ICCAD-1 test non-hotspot count in the current dataset copy differs from a count reported in one literature source. This provenance/count discrepancy should be resolved or disclosed before treating cross-paper metrics as directly comparable.

## Architecture

### Reference Baseline CNN

![Reference baseline CNN architecture](report/figures/baseline_architecture.png)

The reference model is a lightweight CNN with two convolutional basic blocks, batch normalization, ELU activations, max pooling, flattening, dropout, and a sigmoid output. It has **6,949 trainable parameters** according to the model configuration used in this project.

The reference CNN is trained end-to-end and serves as the baseline; it is not used as the feature extractor for the proposed classical-head experiments.

### Proposed Architecture

![Proposed residual depthwise-separable CNN architecture](report/figures/proposed_architecture.png)

The proposed backbone uses a stem followed by three residual depthwise-separable convolution blocks (64, 128, and 192 filters), with batch normalization and spatial dropout. It has **163,489 parameters** according to the experiment configuration.

Feature taps are evaluated as follows:

- **Early features:** global-average-pooled activations after the first residual block (64 dimensions).
- **Late features:** the 128-dimensional representation immediately before the end-to-end classification head.
- **Combined features:** concatenation of early and late features (192 dimensions).

Classical classifiers are trained on the extracted representations. The evaluated heads include **SVM, Random Forest, Logistic Regression, Decision Tree, and an ensemble**. The ensemble combines SVM, Random Forest, and Logistic Regression on late features only. Early and combined features are separate ablation settings, not inputs to that ensemble.

## Methodology

### 1. Backbone and loss comparison

Both backbones are evaluated with binary cross-entropy (BCE) and focal loss. The focal-loss configuration uses **γ = 2.0 and α = 0.75**. This factorial setup is intended to separate the effects of backbone and loss rather than attributing every performance difference to a single design choice.

### 2. CNN feature extraction

The proposed backbone is trained end-to-end and then reused to extract early, late, and combined representations. The extracted features are passed to classical classifiers, allowing the experiment to compare end-to-end CNN classification with two-stage feature-extraction pipelines.

### 3. Classical classifiers

The notebook evaluates:

- **SVM:** RBF kernel; its hyperparameter selection uses validation data.
- **Random Forest:** tree-based classifier.
- **Logistic Regression:** linear probabilistic classifier.
- **Decision Tree:** individual tree classifier.
- **Ensemble:** soft-voting combination of SVM, Random Forest, and Logistic Regression on late features.

Where applicable, scaling and classifier settings are fitted using training/validation data only. Test examples must not be used to choose model settings or thresholds.

### 4. Threshold selection and evaluation

Decision thresholds and tunable hyperparameters are selected on validation data. Final test metrics include balanced accuracy, recall, precision, F1, specificity, false-alarm counts, ROC-AUC, and average precision where available.

The analysis also includes confidence intervals, statistical comparisons with multiple-test correction, calibration metrics, threshold sensitivity, and compute-cost measurements. Results are reported across five seeds; GPU nondeterminism means exact bitwise reproduction is not guaranteed.

## Setup and Reproducibility

The notebook is the main entry point:

`ICCAD12_CNN_Classical_ML_Reproducible.ipynb`

A GPU-enabled Kaggle or Colab environment is recommended. Install the dependencies required by the notebook and make the ICCAD-12 dataset available at the path expected by its dataset-loading cells.

### Run the experiments

1. Open the consolidated notebook.
2. Configure the runtime and dataset path.
3. Run the cells in order to execute the experiment pipeline.
4. Preserve generated raw JSON/NPZ results so interrupted runs can be resumed where supported.
5. Run the analysis section to regenerate tables and figures from the saved outputs.

### Reproduce analysis without retraining

1. Download and extract `da3_raw.zip`.
2. Point the notebook's raw-results path (`RAW_DIR`) to the extracted raw-output directory.
3. Run the analysis/export cells that consume those saved runs.

See `outputs/config.json` for the recorded configuration and `outputs/README_reproducibility.md` for the run notes. The two original Kaggle notebooks may also be retained as provenance for the original training run and resumed analysis; the consolidated notebook is the repository's main entry point.

## Results Summary

### End-to-end CNNs

Average balanced accuracy across the five benchmarks, reported as mean ± standard deviation over five seeds:

| Method | Average balanced accuracy |
|---|---:|
| Reference CNN (end-to-end, focal loss) | **0.924 ± 0.019** |
| Proposed CNN (end-to-end, focal loss) | 0.886 ± 0.014 |

The reference CNN is stronger on average in these results. The proposed CNN performs better on some individual benchmarks, but its average performance does not establish an overall improvement.

### Interpretation

- The reference CNN is a strong baseline and should remain in the main comparison.
- Effects of backbone, loss, feature depth, and classical head should be described using the matching ablation tables and statistical tests.
- Performance can vary substantially by benchmark, especially on smaller and more imbalanced test sets.
- Do not claim that the proposed pipeline is state of the art or consistently superior to published detectors without a verified, protocol-aware comparison.

For per-benchmark results and uncertainty estimates, see the CSV files in `results/tables/`. For visual summaries, see `results/figures/`.

## Figures

The figure directory includes the benchmark class distribution, balanced-accuracy comparison, confusion matrices, calibration, threshold sensitivity, and supplementary plots of method averages and backbone effects. The two architecture diagrams are included separately:

- `report/figures/baseline_architecture.png`
- `report/figures/proposed_architecture.png`

Figure filenames are intended to be descriptive. When inserting them into a report, use figure numbers and captions that match the final order in the document.

## Statistical Analysis

The saved tables include mean ± standard deviation over seeds, confidence intervals, paired statistical comparisons, and Holm correction for multiple comparisons. Interpret significance cautiously: five seeds provide limited statistical power, and tests over repeated runs do not eliminate all uncertainty due to data splitting or GPU nondeterminism.

Published comparisons are not automatically statistically comparable to these results. Literature values may come from different image resolutions, dataset versions, splits, threshold protocols, and false-alarm definitions. Cite and verify the original source for every published value.

## Compute Cost

The experiment outputs include model parameter counts, available FLOP measurements, feature-extraction time, head inference latency, and training/fit times. Latency depends on the hardware and measurement setup; it should be reported as a measured result from this setup, not as a universal runtime guarantee.

## Limitations

- Only five seeds were used, limiting power to detect small differences.
- Some validation sets, particularly for the smaller benchmarks, contain few hotspot examples; threshold/model selection can therefore be unstable.
- GPU operations were not fully deterministic, so exact bitwise reproduction is not guaranteed.
- The dataset provenance and ICCAD-1 non-hotspot count discrepancy need to be documented when comparing against published work.
- Published methods may use different input resolutions, splits, thresholds, and false-alarm definitions.
- The latency measurements represent a specific execution setup.
- Cross-benchmark generalization was not established by this experiment.

## References

The report's bibliography should contain the verified papers used to motivate hotspot detection, CNN architectures, classical classifiers, calibration, class imbalance, and statistical testing. Keep the bibliography in the report and cite each reference next to the claim it supports. Verify the complete bibliographic details and published comparison values against the original papers before submission.
