# FAIR-IRS: burden and evasion in history-dependent insider-risk alerting

This repository contains the code, Colab notebooks, result files and extended results for the paper

> R. Ben Salem, "Who Gets Suspected, and Who Gets Away? Burden and Evasion in History-Dependent Insider-Risk Alerting," submitted to *IEEE Transactions on Dependable and Secure Computing*.

It covers burden metrics, outcome-based audits, cooldowns and similarity scheduling, parity rules with memory, private burden reports, and the adversarial evaluation (discount farming and timing attacks, with behavior relocated and rescored by the saved detectors, including single-insider attacks and defenses against farming in notebook 11, baselines and sensitivity analyses in notebook 12, and late verdicts in notebook 13).

## Contents

| Folder | Contents |
|---|---|
| `src/` | Python modules. `fairirs.py` has feature extraction, the detectors (IF, AE, LR, GBM, LSTM), the policies and the metrics. The other modules add experiments; later modules import earlier ones. |
| `notebooks/` | Colab notebooks, numbered in the order they must first be run. Each one writes its modules with `%%writefile`, so it runs with *Runtime → Run all* and no edits. |
| `results/` | The result CSV files of each notebook, as produced on Colab (CPU runtime). |
| `scripts/` | Scripts that turn the CSV files into the LaTeX tables and Fig. 2 of the paper. Run them from the repository root. |
| `tables/` | The tables produced by `scripts/`. |
| `figures/` | Figures of the paper and of the extended results. |
| `paper/` | Extended results (tables and figures prefixed E in the paper), as PDF and LaTeX. |

## Data

The experiments use the CERT Insider Threat Test Dataset, releases r4.2, r5.2 and r6.2 (J. Glasser and B. Lindauer, "Bridging the gap: A pragmatic approach to generating insider threat data," IEEE S&P Workshops, 2013), distributed by Carnegie Mellon University's Software Engineering Institute through the KiltHub repository. **The dataset is not redistributed here.** Notebook 01 downloads r4.2 and r6.2; notebook 07 downloads r5.2, or uses a copy of `r5.2.tar.bz2` that you place anywhere in your Google Drive if the download server refuses automated requests. Use of the data is subject to its license and terms on KiltHub. The CERT data are synthetic; no real employee is represented.

The notebooks cache the extracted features and the detector scores in `MyDrive/FAIR-IRS_results/cache/`, so later notebooks need neither the raw data nor retraining.

## How to reproduce

1. Open each notebook in Google Colab, in numerical order, and choose *Runtime → Run all* (CPU runtime; High-RAM helps for r5.2). Accept the Google Drive prompt.
2. Each notebook writes its results to a subfolder of `MyDrive/FAIR-IRS_results/` and zips them. If Colab disconnects, run it again: finished parts are reloaded.
3. Copy the CSV files into `results/<notebook name>/` and run the scripts in `scripts/` from the repository root to regenerate the tables.

Running all notebooks takes about one day of Colab CPU time, most of it in notebooks 01 (feature extraction and LSTM training), 07 (r5.2), 09, 10 and 11 (attacks with rescoring). Notebooks 11 to 13 reuse the caches written by the earlier notebooks, so run them last, in order.

Seeds: detectors use seeds 0–4 (r4.2) and 0–2 (r5.2); attacks use draw numbers as seeds; bootstraps use fixed seeds given in the code.

## From paper to code

Run the scripts from the repository root. For the final paper: `tables_main_final_and_fig2.py`, then `tables_supplement_final.py`. The older scripts reproduce the tables of earlier rounds, some of which the supplement still uses.

| Paper | Notebook | Result files | Script |
|---|---|---|---|
| Tables II and III (burden and benefit, seed means), Fig. 2 and Tables IV and V (farming, timing one insider at a time) | 11, 12, 13 | `11_unilateral_defenses_power/`, `12_baselines_sensitivity_clearance/`, `13_late_verdicts_permutation/` | `tables_main_final_and_fig2.py` (`tab_rq1_11.tex`, `tab_benefit11.tex`, `tab_farm11.tex`, `tab_timing11.tex`, `figures/security.pdf`) |
| Tables S5 to S14 (seeds, per-employee baselines, power, cooldown sensitivity, farming configurations and clearance, timing for every rule, baselines, late verdicts, permutation without the most-flagged employees) | 11, 12, 13 | as above | `tables_supplement_final.py` (`sup11_tables.tex`) |
| Tables S2 to S4, S15 and Fig. S1 (scheduler, exact rules, all-at-once timing, plausibility) | 06 to 08, 10 | `10_attack_realism/m1_*`, `m2_*`, `j2_*`, `k2_*` | `tables_main_I_to_III.py`, `tables_main_IV_V_figures.py`, `tables_main_narrowed_and_fig2.py` (`figures/realism.pdf`) |
| Fig. S2 (burden--security trade-off), Table S25 (social cost) | 09 | `09_security_rescored/l5_social_cost_*.csv` | `tables_main_IV_V_figures.py` |
| Tables S20, S22, S23, S26 (claims, all-at-once timing attackers, decomposition, non-inferiority margins) | 06 to 08, 10 | `m1_*`, `i2_pooled.csv`, `j6_*`, `j8_*`, `k2_*` | `tables_supplement_realism_claims_margins.py` |
| Tables S21, S24, S27 (rescoring checks, farming grid, verdict models) | 09 | `l0_*`, `l3_*`, `l4_*` | `tables_supplement_S5_to_S8.py` |
| Extended results (tables and figures prefixed E) | 01 to 10 | all earlier result folders | `tables_extended_*.py`, `tables_r52_cooldown.py`, `tables_noninferiority_attacks.py`, `tables_part_D.py` |

The late-verdict cooldown rows of `11_unilateral_defenses_power/n3_farming_*.csv` (variants "verdicts 7 days late" and "verdicts 14 days late") come from a simulator in which each new alert overwrote a clearance whose verdict was still pending, which disabled the cooldown for employees alerted every day. They are superseded by `13_late_verdicts_permutation/`, produced with the corrected simulator (`src/fairirs_rev5.py`); the scripts above exclude them.

## Use of generative AI

During the preparation of this work, the author used ChatGPT (OpenAI) and Claude (Anthropic) to assist with language editing and phrasing, as well as with code documentation, debugging, and implementation troubleshooting. All methodological choices, simulation design, parameterisation, evaluation criteria, and interpretation of results were determined by the author. The final manuscript, simulation implementation, and generated outputs were reviewed and verified by the author, who retains full responsibility for the content of the manuscript, the code, and the reported results.

## Requirements

Python 3.10+, NumPy, pandas, SciPy, scikit-learn, PyTorch (LSTM) and matplotlib, all preinstalled on Colab. DP-SGD and its Rényi accountant are implemented in `src/fairirs.py`.

## License

Code: MIT License (`LICENSE`). Result files, tables, figures and extended results: CC BY 4.0. The CERT data keep their own license.

## Citation

See `CITATION.cff`. Please cite the paper and the archived version of this repository (Zenodo DOI).
