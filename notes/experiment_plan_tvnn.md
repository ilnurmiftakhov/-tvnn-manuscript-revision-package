# TVNN experiment plan

## Objective
Run the highest-value initial experiment for the manuscript revision:
1. reproduce the public TVNN code path;
2. audit paper-code consistency for the forecasting pipeline;
3. establish a minimal reproducibility baseline;
4. prepare the ground for a stronger validation section.

## Minimal verification set
- V1: repository clones successfully and code structure matches manuscript sections.
- V2: core forecasting script runs or fails with a specific reproducible error.
- V3: Fourier decomposition semantics in code are compared against the manuscript text.
- V4: if execution is possible, capture at least one benchmark run with logged RMSE outputs.
- V5: if execution is blocked, identify the exact blocker and the smallest patch needed to continue.

## Immediate tasks
1. Clone repository and inspect files.
2. Read dataset loaders and training scripts.
3. Attempt a reduced local run of TVNN.
4. Record discrepancies, runtime blockers, and first results.
5. Decide next experiment branch: reproduction, baseline extension, or dataset integration.
