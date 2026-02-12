#!/bin/bash
set -euo pipefail

BASE_DIR="$HOME/telegramdaily"
LOG_DIR="$BASE_DIR/logs"
OUT_DIR="$BASE_DIR/data/energy"

PY="/opt/anaconda3/envs/finance-env/bin/python"
FETCH="$BASE_DIR/Telegram_Energy_Fetch_Filter.py"
SUM="$BASE_DIR/Summarize_and_Email.py"

mkdir -p "$LOG_DIR" "$OUT_DIR"
cd "$BASE_DIR"

STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
LOG_OUT="$LOG_DIR/energy_weekly_${STAMP}.out"
LOG_ERR="$LOG_DIR/energy_weekly_${STAMP}.err"
ln -sf "$LOG_OUT" "$LOG_DIR/energy_weekly_latest.out"
ln -sf "$LOG_ERR" "$LOG_DIR/energy_weekly_latest.err"

HOURS="${1:-168}"

echo "=== Energy weekly started: $(date -u) ===" | tee -a "$LOG_OUT"

# Step 1: fetch/filter
"$PY" "$FETCH" --hours "$HOURS" >>"$LOG_OUT" 2>>"$LOG_ERR"

# Step 2: parse saved CSV path
SAVED_CSV="$(grep -Eo '/[^ ]*/filtered_energy_[0-9]{4}-[0-9]{2}-[0-9]{2}\.csv' "$LOG_OUT" | tail -n 1 || true)"
if [[ -z "${SAVED_CSV}" || ! -f "${SAVED_CSV}" ]]; then
  echo "ERROR: Could not detect saved energy CSV path from log." | tee -a "$LOG_OUT"
  exit 2
fi
echo "Energy CSV: $SAVED_CSV" | tee -a "$LOG_OUT"

# Step 3: summarise + email
"$PY" "$SUM" --csv "$SAVED_CSV" --hours "$HOURS" >>"$LOG_OUT" 2>>"$LOG_ERR"

echo "✅ ENERGY WEEKLY SUCCESS" | tee -a "$LOG_OUT"