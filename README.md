# tvnn-manuscript-revision-package

This repository contains the experiment code, results, and figures produced during an independent robustness and baseline reassessment of the public repository for the manuscript **"Time-Varying Biological Time-Series Prediction and Pattern Interpretation using Koopman Theory and Large Language Models"**.

## Scope

Included here:
- forecasting-side robustness audits for the TVNN pipeline;
- Fourier decomposition comparison experiments;
- a robustness atlas over `alpha`, forecast horizon, and random seed;
- TVPRLLM baseline reassessment experiments;
- generated JSON result files and figure artifacts.

Excluded here:
- manuscript PDFs;
- Word documents and article packaging files;
- manuscript-oriented insertion packages;
- the cloned upstream repository itself.

## Main findings

1. The released Fourier masking strategy materially changes forecasting behavior and stability.
2. A symmetric low-pass alternative can improve performance in some regimes and strongly changes stability behavior.
3. The released TVPRLLM tasks are highly separable for simple non-LLM classifiers.

## Repository structure

- `experiments/` — experiment scripts
- `experiments/results/` — raw JSON outputs
- `experiments/figures/` — generated plots
- `notes/` — experiment plan notes
- `CHANGELOG.md` — lab notebook style progress log

## Experiments

### Forecasting-side
- `experiments/tvnn_fourier_audit.py`
- `experiments/tvnn_fourier_extended.py`
- `experiments/tvnn_fourier_robustness_atlas.py`

### TVPRLLM-side
- `experiments/tvprllm_baseline_audit.py`
- `experiments/tvprllm_baseline_cv.py`

## Figures

### Pendulum robustness line plots

![Pendulum robustness](experiments/figures/tvnn_pendulum_lineplots.png)

### Pendulum RMSE improvement heatmap

![Pendulum improvement heatmap](experiments/figures/tvnn_pendulum_improvement_heatmap.png)

### Pendulum stability heatmap

![Pendulum stability heatmap](experiments/figures/tvnn_pendulum_stability_heatmap.png)

### Lorenz robustness line plots

![Lorenz robustness](experiments/figures/tvnn_lorenz_lineplots.png)

### Lorenz RMSE improvement heatmap

![Lorenz improvement heatmap](experiments/figures/tvnn_lorenz_improvement_heatmap.png)

### Lorenz stability heatmap

![Lorenz stability heatmap](experiments/figures/tvnn_lorenz_stability_heatmap.png)

## Reproducing the experiments

The experiment scripts were written against a local clone of the public upstream repository placed at `repo_tvnn/`. That upstream clone is not included here. To rerun the scripts, first clone:

- https://github.com/347251369/Time-Varying-Biological-Time-Series-Prediction

and place it locally as:

`repo_tvnn/`

Then run the scripts from the repository root.

## Upstream source

- Public repository analyzed: https://github.com/347251369/Time-Varying-Biological-Time-Series-Prediction

# -tvnn-manuscript-revision-package
