#!/bin/bash
set -euo pipefail

BASE_DIR="$HOME/telegramdaily"
PY="/opt/anaconda3/envs/finance-env/bin/python"
DEDUP="$BASE_DIR/Deduplicate_AfterFetch.py"

IN_CSV="${1:-$BASE_DIR/data/filtered/filtered_2025-12-28.csv}"

OUT_DIR="$BASE_DIR/data/filtered_deduped"
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
OUT_CSV="$OUT_DIR/filtered_deduped_${STAMP}.csv"
DUPS_CSV="$OUT_DIR/dups_${STAMP}.csv"

mkdir -p "$OUT_DIR"
cd "$BASE_DIR"

echo "=== TEST DEDUP START $(date -u) ==="
echo "IN_CSV=$IN_CSV"
echo "OUT_CSV=$OUT_CSV"
echo "DUPS_CSV=$DUPS_CSV"
echo

if [[ ! -f "$IN_CSV" ]]; then
  echo "❌ Input CSV not found: $IN_CSV"
  exit 2
fi

"$PY" "$DEDUP" \
  --in_csv "$IN_CSV" \
  --out_csv "$OUT_CSV" \
  --dups_report "$DUPS_CSV" \
  --bucket_by global \
  --sim_threshold 0.90 \
  --jaccard_min 0.40

echo
echo ">>> Verifying outputs..."

[[ -s "$OUT_CSV" ]] || { echo "❌ OUT_CSV missing/empty: $OUT_CSV"; exit 3; }
[[ -f "$DUPS_CSV" ]] || { echo "❌ DUPS_CSV missing: $DUPS_CSV"; exit 4; }

"$PY" - <<PY
import pandas as pd
df_out = pd.read_csv("$OUT_CSV")
print("✅ OUT rows:", len(df_out))

# DUPS file might be header-only or empty; handle both.
try:
    df_dups = pd.read_csv("$DUPS_CSV")
    print("✅ DUPS rows:", len(df_dups))
except Exception as e:
    print("⚠️ DUPS unreadable/empty (ok if no duplicates):", type(e).__name__, str(e)[:120])
PY

echo
echo "=== TEST DEDUP SUCCESS $(date -u) ==="
echo "OUT:  $OUT_CSV"
echo "DUPS: $DUPS_CSV"
