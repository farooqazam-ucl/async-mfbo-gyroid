# Raw-data archive: deformed shapes, logs and development runs

The archive `sheet_gyroid_raw_snapshots_and_logs.zip` (about 150 MB) completes
this repository.  It is deposited separately on Zenodo; its DOI is listed at
the top of the main `README.md`.  Unzip it into the repository folder:
its `runs/` and `data/` folders merge with the ones here.  It is needed only to
redraw the deformed sections of Fig. 5 of the article and Fig. S7 of Online
Resource 1 (`python3 code/fig_fe.py fig6 fig16`) and to recompute `data/slab_profiles.json`
(`python3 code/make_slab_cache.py`); everything else runs from the repository
alone.

## `runs/*/*_snaps.npz`: deformed shapes and plastic strain

Snapshots of every analysis that saved them: the 13 high-fidelity analyses of
the design of experiments and the uniform low-fidelity analysis (`doe/`), the
N = 48 reference mesh (`conv/u_N48`), the two validation densities (`val/`)
and the four re-analysed Pareto-optimal designs (`confirm/`).  Each file holds,
at nominal strains 0.05, 0.1, 0.2, 0.3, 0.4, 0.5 and 0.6:

| Key | Content |
|---|---|
| `x_<strain>` | node coordinates (mm), one row per node |
| `L_<strain>` | current lateral periods Lx, Ly (mm) |
| `ep_<strain>` | equivalent plastic strain, one value per element |
| `conn` | element connectivity (8 node indices per hexahedron) |
| `sx`, `sy` | periodic image shifts of each element node |
| `eijk` | voxel indices (i, j, k) of each element |

## Logs and job lists

- `runs/job_lists/jobs_*.json`: the exact command-line arguments of every
  queued analysis, and `jobs_*.json.log`: the start and end of each job with
  its wall-clock time.  `jobs_main.json` is a first pass with a preliminary
  calibration (σ0 = 33.2 MPa, Q = 22.2 MPa) whose results were superseded by
  `jobs_main2.json`; it is kept only for completeness.
- `runs/calib/log_*.txt`, `runs/confirm/*_rerun.log`, `runs/prune/log.txt`:
  console output of the calibration analyses, the repeated confirmation
  analyses and the pruning demonstration.
- `data/logs/`: console output of the optimisation campaigns and of the
  elastic homogenisation.

## `development/`: runs not used in the article

Exploratory analyses made while the solver was being developed (earlier
hourglass and loading-time settings, stability checks), and
`fesolver_v1_backup.py`, the earlier solver version that produced some of
them.  None of these results appears in the article.

## What does not exist

- The displacement fields of the elastic homogenisation were not stored; only
  the homogenised moduli are kept (`data/elastic.json`).
- No raw experimental curves were used.  The comparison with experiments uses
  the power-law fits published by Abueidda et al. (2019),
  E* = 892 ρ^1.39 MPa and σ* = 54 ρ^1.63 MPa, and the PA2200 data-sheet values
  listed in the material table (Online Resource 1, Table S2).
- The unstable first attempts of design-of-experiments analysis `hf_12` and of
  the knee confirmation analysis were not kept; only the repeated analyses
  with element deletion (Online Resource 1, Sect. S3.4) are stored.  The unstable
  first attempt of the minimum-variance design is kept as
  `runs/confirm/minVAR_failed_*` in this repository.
