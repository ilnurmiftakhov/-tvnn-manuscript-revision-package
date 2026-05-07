# CHANGELOG

## 2026-05-07
- Started experimental audit of the TVNN manuscript and public code.
- Created a plan artifact (`notes/experiment_plan_tvnn.md`) with explicit verification targets.
- Cloned the public repository into `repo_tvnn/` and confirmed the reduced TVNN forecasting script runs locally.
- Implemented `experiments/tvnn_fourier_audit.py` to compare the public Fourier decomposition against a symmetric low-pass alternative.
- Logged first results in `experiments/results/tvnn_fourier_audit.json` and summarized them in `outputs/tvnn_initial_experiment_log.md`.
- Initial finding: the public FFT mask does not cleanly separate low/high frequencies, creating a paper-code consistency concern and materially changing reduced-run forecasting behavior.
- Ran an extended 5-seed, 60-epoch comparison in `experiments/tvnn_fourier_extended.py` and saved results to `experiments/results/tvnn_fourier_extended.json`.
- Extended result: the corrected symmetric low-pass decomposition reduced RMSE on the pendulum dataset at all tested horizons and strongly reduced long-horizon instability on Lorenz.
- Wrote the extended report to `outputs/tvnn_extended_experiment_report.md`.
- Added `experiments/tvprllm_baseline_audit.py` to compare the released TVPRLLM task against simple non-LLM baselines under the repository's 80/20 split.
- Added `experiments/tvprllm_baseline_cv.py` for a 5-fold stratified cross-validation sanity check.
- Saved raw baseline results to `experiments/results/tvprllm_baseline_audit.json` and `experiments/results/tvprllm_baseline_cv.json`.
- Wrote `outputs/tvprllm_baseline_audit_report.md`; initial finding: the released TVPRLLM benchmark is nearly solved by simple classifiers, weakening any strong claim that an LLM is necessary for this task.
- Added and ran `experiments/tvnn_fourier_robustness_atlas.py` over alpha, horizon, and seed to map how the Fourier mask affects forecasting accuracy and stability.
- Saved the robustness atlas to `experiments/results/tvnn_fourier_robustness_atlas.json` and generated six figure artifacts in `experiments/figures/`.
- Wrote the interpretation report to `outputs/tvnn_fourier_robustness_atlas_report.md`; key finding: the exact FFT-mask design materially changes both RMSE and stability, including catastrophic divergence for some Lorenz settings.
- Assembled a revision package with tables, figures, and manuscript-ready text in `outputs/coauthor_revision_package.md`.
- Exported the package to Word format as `outputs/coauthor_revision_package.docx` using Pandoc with embedded figure resources.
- Produced a more academic, numbered, TOC-enabled version in `outputs/coauthor_revision_package_academic.md` and `outputs/coauthor_revision_package_academic.docx`.
- Prepared an insertion-ready English manuscript package in `outputs/manuscript_insertion_package_en.md` and `outputs/manuscript_insertion_package_en.docx`.
- Prepared a Russian insertion-ready version in `outputs/manuscript_insertion_package_ru.md` and `outputs/manuscript_insertion_package_ru.docx`.
