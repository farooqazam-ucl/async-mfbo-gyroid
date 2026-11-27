# Asynchronous multi-fidelity Bayesian optimisation with in-solver pruning for functionally graded gyroid energy absorbers

Code and data for the article of the same title by Farooq I. Azam and Abdullah
Jamil, submitted to *Structural and Multidisciplinary Optimization*.  Section,
figure and table numbers below refer to the article (Figs. 1-5, Tables 1-2)
and to its Online Resource 1 (Sects. S1-S7, Figs. S1-S15, Tables S1-S5).

Archived versions (Zenodo): this repository, release v1.0.0: DOI to be
added; raw-data archive (see below): DOI to be added.

The repository contains:

- an **explicit finite element (FE) solver** for the quasi-static crush of
  voxel models of graded sheet-gyroid columns, written in Python with Numba
  (one-point hexahedra with hourglass control, J2 plasticity with Voce
  hardening, self-contact, rigid platens, periodic lateral faces and an
  in-solver monitor that stops analyses which violate a stress limit);
- the **geometry generator**: gyroid level set, gradient-normalised distance,
  cubic Bézier thickness grading, density normalisation and periodic voxel
  meshing;
- the **optimisation framework**: multi-task Gaussian-process surrogates
  (BoTorch/GPyTorch) over two FE fidelities and an analytical Gibson–Ashby
  prior, a feasibility classifier for pruned analyses, and a discrete-event
  simulation of synchronous (Mode A), asynchronous (Mode B) and elastic
  (Mode C) execution;
- the **results**: the histories and metrics of all 78 FE analyses (the 68
  analyses of the FE campaign, i.e. design of experiments, test set,
  convergence, loading-rate and validation studies, plus 4 calibration
  analyses, the 4 re-analysed Pareto designs with one unstable first attempt,
  and 1 pruning demonstration), the logs of all 50 optimisation campaigns
  (ten arms, five seeds each), and the scripts that regenerate every number
  and figure in the article and in Online Resource 1 from them.

## Contents

| Folder or file | What it holds |
|---|---|
| `code/` | All source code and scripts (see the code guide below) |
| `data/` | Processed results: metrics of every FE analysis, campaign logs (`data/campaigns/`), statistics, surrogate accuracy, settings and the fitted FE emulator |
| `runs/` | Stress–strain, energy, lateral-period and solve-time histories (`*_hist.npz`) and metadata (`*_meta.json`) of every FE analysis; `runs/job_lists/` holds the exact solver arguments |
| `reproduce.sh` | Regenerates the processed results, every number quoted in the article and Online Resource 1, and all figures |
| `docs/RAW_DATA.md` | Description of the companion raw-data archive (deformed shapes, plastic-strain fields and logs) |
| `requirements.txt` | Python packages, pinned to the versions used for the article |
| `CITATION.cff` | Citation information |

## Installation

1. Install Python 3.13 (the version used for the article; 3.11 or later should
   work).
2. Create an environment and install the packages:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate            # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

   PyTorch is only used on the CPU.  On Linux, the smaller CPU-only build can
   be installed first with
   `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
   If a pinned version is not available for your Python version, remove the
   version numbers from `requirements.txt`; BoTorch and GPyTorch are the
   packages whose interfaces are most likely to change between versions.

3. The figure scripts typeset all text with LaTeX (Matplotlib `text.usetex`)
   in Latin Modern, so they also need a LaTeX installation with the `lmodern`
   and `cm-super` packages (TeX Live or MiKTeX; on Ubuntu:
   `sudo apt install texlive-latex-extra lmodern cm-super dvipng`).  The
   numerical scripts do not need LaTeX.

## Reproducing the numbers and figures of the article

```bash
bash reproduce.sh
```

This takes a few minutes (under one minute on the machine described below).
It runs the steps below, which can
also be run one at a time from the repository folder (on Windows, run them in
the same order without `bash`):

```bash
python3 code/analyse_fe.py 7.0 7.59      # metrics of every FE analysis -> data/fe_records.json
python3 code/select_confirm.py           # four representative Pareto designs -> data/confirm_designs.json
python3 code/collect_confirm.py          # their FE re-analyses -> data/confirm_results.json
python3 code/campaign_stats.py           # campaign statistics and Mann-Whitney tests -> data/campaign_stats.json
python3 code/fig_parity.py               # surrogate accuracy -> data/parity.json and Fig. S11
python3 code/make_macros.py              # every number quoted in the article and Online Resource 1 -> output/results_macros.tex
python3 code/fig_geometry.py fig1 fig2 fig3
python3 code/fig_fe.py fig4 fig5 fig7 fig8 figS1
python3 code/fig_fe.py fig6 fig16        # Figs. S7 and 5; need the raw-data archive (see below)
python3 code/fig_schematic.py
python3 code/fig_campaign.py fig9 fig12 fig13 fig14 fig_baselines fig15 figS2 figS3
```

The arguments `7.0 7.59` of `analyse_fe.py` are the payload stress limit (MPa)
and the corresponding threshold for the low-fidelity model.
`bash reproduce.sh --all` also repeats the ten surrogate fits with different
random seeds that give the ranges quoted in Online Resource 1, Sect. S7.2
(`code/parity_seeds.py`, under a minute more).

The processed results in `data/` are rewritten by these steps; if the
repository is a Git clone, `git status` afterwards shows that they are
unchanged.  The numbers are written as LaTeX macros
(`\newcommand{\name}{value}`) to `output/results_macros.tex`; every number
quoted in the article and in Online Resource 1 was taken from this file.
Figures are written to `output/figures/` with the file names below; the
article and Online Resource 1 use them unchanged, renamed `Fig1.pdf` ...
`Fig5.pdf` and `FigS1.pdf` ... `FigS15.pdf`.

| Script and argument | File | Figure |
|---|---|---|
| `fig_schematic.py` | `fig10_framework_schematic.pdf` | Fig. 1 |
| `fig_campaign.py fig12` | `fig12_hypervolume_convergence.pdf` | Fig. 2 |
| `fig_campaign.py fig_baselines` | `fig14b_baseline_comparison.pdf` | Fig. 3 |
| `fig_campaign.py fig15` | `fig15_pareto_front.pdf` | Fig. 4 |
| `fig_fe.py fig16` | `fig16_collapse_mechanism.pdf` | Fig. 5 (needs the raw-data archive) |
| `fig_geometry.py fig1` | `fig01_geometry_parameterisation.pdf` | Fig. S1 |
| `fig_geometry.py fig2` | `fig02_voxel_pipeline.pdf` | Fig. S2 |
| `fig_geometry.py fig3` | `fig03_through_thickness_resolution.pdf` | Fig. S3 |
| `fig_fe.py figS1` | `figS1_energy_diagnostics.pdf` | Fig. S4 |
| `fig_fe.py fig4` | `fig04_mesh_convergence.pdf` | Fig. S5 |
| `fig_fe.py fig5` | `fig05_voxel_discretisation.pdf` | Fig. S6 |
| `fig_fe.py fig6` | `fig06_experimental_validation.pdf` | Fig. S7 (needs the raw-data archive) |
| `fig_fe.py fig7` | `fig07_metric_definitions.pdf` | Fig. S8 |
| `fig_fe.py fig8` | `fig08_pruning_in_action.pdf` | Fig. S9 |
| `fig_campaign.py fig9` | `fig09_feasibility_classifier.pdf` | Fig. S10 |
| `fig_parity.py` | `fig11_surrogate_parity.pdf` | Fig. S11 |
| `fig_campaign.py fig13` | `fig13_worker_occupancy.pdf` | Fig. S12 |
| `fig_campaign.py fig14` | `fig14_h2_ablation.pdf` | Fig. S13 |
| `fig_campaign.py figS2` | `figS2_fidelity_allocation.pdf` | Fig. S14 |
| `fig_campaign.py figS3` | `figS3_per_seed_hypervolume.pdf` | Fig. S15 |

## Raw-data archive

The deformed shapes and plastic-strain fields of the analyses shown in the
figures (`runs/*/*_snaps.npz`, about 150 MB), the solver and campaign logs and
the development runs are not kept in this repository.  They are in the
companion archive `sheet_gyroid_raw_snapshots_and_logs.zip`, which is
deposited separately on Zenodo (its DOI is listed at the top of this README).
Unzip it into the repository folder; its `runs/` and `data/` folders merge
with the ones here.  It is needed only to redraw the deformed sections of
Figs. 5 and S7.  The plastic-strain profiles that `make_macros.py` uses are
also stored in `data/slab_profiles.json` (written from the snapshots by
`code/make_slab_cache.py`), so every number in the article and Online
Resource 1 can be regenerated without the archive.  `docs/RAW_DATA.md` describes its contents.

## Repeating the computations

The steps above start from stored results.  The computations themselves can
be repeated as follows.  The timings in the article were measured on a
virtual machine with two cores of an Intel Xeon processor (2.1 GHz), running
two single-threaded analyses at a time; solve times on other hardware differ.

**One FE analysis.**  For example, in a new folder:

```bash
mkdir scratch && cd scratch
python3 ../code/run_design.py --q 1 1 1 1 --N 36 --sig0 32.4 --Qv 21.7 --Tf 16 --ramp 0.1 --hgfloor 0.1 --out uniform_hf
python3 ../code/run_design.py --q 1 1 1 1 --N 36 --sig0 32.4 --Qv 21.7 --Tf 16 --ramp 0.1 --hgfloor 0.1 --limit 7.0 --out uniform_hf_pruned
```

Each analysis writes `<out>_hist.npz` and `<out>_meta.json` (and
`<out>_snaps.npz` when `--snaps` is given).

`--q` are the four Bézier control values, `--N` the number of voxels per
unit-cell edge (24: low fidelity; 36: high fidelity), `--sig0` and `--Qv` the
calibrated Voce parameters (MPa), `--limit` the stress limit for in-solver
pruning (MPa) and `--snaps` a list of strains at which the deformed shape is
saved.  A high-fidelity analysis takes about nine minutes and a low-fidelity
one about one and a half minutes on the machine above.

**The FE campaign.**  `runs/job_lists/` holds the exact arguments of the
queued analyses: `jobs_main2.json` (design of experiments, test set,
convergence, validation and loading-rate analyses), `jobs_final.json` (six
analyses of `jobs_main2.json` that had not finished when that queue was
stopped; `conv/u_N42` in `jobs_main2.json` was not completed and is not used)
and `jobs_confirm.json` (the four re-analysed Pareto designs).  The calibration
analyses (`runs/calib/`), the pruning demonstration (`runs/prune/`) and the
unstable first attempt of the minimum-variance design
(`runs/confirm/minVAR_failed`) were run individually; their logs are in the
raw-data archive.  To repeat the campaign in an empty folder with two
analyses at a time:

```bash
mkdir rerun && cd rerun
mkdir doe test conv val rate confirm
python3 ../code/jobqueue.py ../runs/job_lists/jobs_main2.json 2
```

Explicit crush analyses are sensitive to round-off after buckling, so
repeated analyses agree closely but not always to the last digit.

**The optimisation campaigns.**  The campaigns run against an emulator of the
FE model, stored in `data/oracle.pkl`.  If that file is deleted (for example
because it cannot be loaded with your version of scikit-learn), it is rebuilt
from the FE results.  Existing campaign logs are skipped, so move
`data/campaigns/` aside first; then

```bash
python3 code/run_campaigns.py A,B,C,A0,B0,Bc,A0c,S,Sb,R 0,1,2,3,4
```

runs all ten arms for seeds 0–4 (about two and a half hours).  The arms are
A, B, C (Modes A, B and C with pruning), A0, B0 (Modes A and B without
pruning), Bc (Mode B without the analytical prior) and the four comparison
arms A0c (synchronous multi-fidelity BO without pruning or prior), S and Sb
(single-fidelity BO, synchronous and asynchronous) and R (space-filling
search).  Repeated campaigns do not reproduce the stored logs exactly: after
the first few designs they follow different paths.  Their results agree with
the stored ones within the spread between seeds.  In a check, seed 0 of arms
B and S reached terminal hypervolumes of 2.22 and 1.62 (stored: 2.01 and
1.56; medians over five seeds 2.19 and 1.62) and the same wall-clock times to
within 1%.

**Elastic homogenisation** (Online Resource 1, Sect. S4.3): `python3 code/run_elastic.py`,
then `python3 code/run_elastic2.py` for the finer meshes at low density
(algebraic multigrid; results in `data/elastic.json`).

## Code guide

| File | Purpose |
|---|---|
| `geometry.py` | Gyroid field, gradient-normalised distance, Bézier thickness law, density normalisation, periodic voxel meshing |
| `fesolver.py` | Explicit dynamic FE solver: one-point hexahedra with hourglass control, J2 plasticity, pinball self-contact, rigid platens, periodic lateral faces with free periods, conditional mass scaling, in-solver monitor |
| `run_design.py` | Command-line interface for one crush analysis |
| `jobqueue.py` | Minimal job queue that runs a list of analyses with a fixed number of concurrent processes and records their wall times |
| `doe.py` | Design of experiments (scrambled Sobol sequence) |
| `metrics.py`, `analyse_fe.py` | Performance metrics (Sect. 4; Online Resource 1, Sect. S5) and collection of the FE results |
| `elastic.py`, `run_elastic.py`, `run_elastic2.py` | Linear-elastic homogenisation with algebraic multigrid (Online Resource 1, Sect. S4.3) |
| `prior_model.py` | Gibson–Ashby layered analytical prior (Sect. 5.2; Online Resource 1, Sect. S6.3) |
| `campaign.py` | Multi-task surrogates, acquisition, feasibility classifier and discrete-event simulation of Modes A, B and C |
| `oracle.py`, `run_campaigns.py` | Emulator of the FE model and the campaign driver (Sect. 6.2; Online Resource 1, Sect. S6.7) |
| `campaign_stats.py` | Campaign statistics and Mann–Whitney tests |
| `select_confirm.py`, `collect_confirm.py` | Selection of the four re-analysed designs from the campaigns of Modes A and B, and collection of their re-analyses |
| `fig_parity.py`, `parity_seeds.py` | Surrogate accuracy on the independent test set (Fig. S11) and its spread over ten random initialisations |
| `make_macros.py`, `campaign_macros.py` | Every number quoted in the article and Online Resource 1, written as LaTeX macros |
| `make_slab_cache.py` | Writes `data/slab_profiles.json` from the snapshot files |
| `fig_*.py`, `figstyle.py` | All figures of the article and Online Resource 1 |
| `test_block.py`, `test_block2.py` | Checks of the material update and platen contact on a uniaxial block (run from `code/`) |
| `test_campaign.py` | Short smoke test of the campaign driver against a cheap analytical stand-in for the FE model, e.g. `python3 code/test_campaign.py A B C 12` |
| `viz_slices.py` | Quick visualisation of deformed sections (needs the raw-data archive) |

All scripts find `data/` and `runs/` relative to the folder that contains
`code/`; set the environment variable `GYROID_ROOT` to use another location.

## Citation

If you use this code or data, please cite the article (see `CITATION.cff`):

> Azam FI, Jamil A (2026) Asynchronous multi-fidelity Bayesian optimisation
> with in-solver pruning for functionally graded gyroid energy absorbers.
> Submitted to Structural and Multidisciplinary Optimization.

## Licence

The code is released under the MIT Licence (`LICENSE`).  The data in `data/`
and `runs/` are released under the Creative Commons Attribution 4.0
International licence (CC BY 4.0, `LICENSE-DATA`).
