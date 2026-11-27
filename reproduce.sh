#!/usr/bin/env bash
# Regenerate the processed results, every number quoted in the article and
# all figures from the stored FE histories and campaign logs.
#
#   bash reproduce.sh          numbers and figures (a few minutes)
#   bash reproduce.sh --all    also repeats the ten surrogate fits of data/parity_seeds.json
#
# Set PYTHON to choose the interpreter (default: python3).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "[1/6] Metrics of every FE analysis -> data/fe_records.json"
$PY code/analyse_fe.py 7.0 7.59 > /dev/null
echo "[2/6] Re-analysed Pareto designs -> data/confirm_designs.json, data/confirm_results.json"
$PY code/select_confirm.py > /dev/null
$PY code/collect_confirm.py > /dev/null
echo "[3/6] Campaign statistics and tests -> data/campaign_stats.json"
$PY code/campaign_stats.py > /dev/null
echo "[4/6] Surrogate accuracy -> data/parity.json and Fig. 11"
$PY code/fig_parity.py > /dev/null
if [ "${1:-}" = "--all" ]; then
  $PY code/parity_seeds.py
fi
echo "[5/6] Numbers quoted in the article -> output/results_macros.tex"
$PY code/make_macros.py
echo "[6/6] Figures -> output/figures/"
$PY code/fig_geometry.py fig1 fig2 fig3
$PY code/fig_fe.py fig4 fig5 fig7 fig8 figS1
if ls runs/val/*_snaps.npz > /dev/null 2>&1; then
  $PY code/fig_fe.py fig6 fig16
else
  echo "      Figs. 6 and 17 skipped: they need the snapshot files of the raw-data archive (see README.md)."
fi
$PY code/fig_schematic.py
$PY code/fig_campaign.py fig9 fig12 fig13 fig14 fig_baselines fig15 figS2 figS3
echo "Done.  Numbers: output/results_macros.tex   Figures: output/figures/"
