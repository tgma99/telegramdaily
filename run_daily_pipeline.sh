#!/bin/bash
set -euo pipefail

BASE_DIR="$HOME/telegramdaily"
LOG_DIR="$BASE_DIR/logs"
DATA_DIR="$BASE_DIR/data/filtered"
ALERT_DIR="$BASE_DIR/data/alerts"
DEDUP_DIR="$BASE_DIR/data/filtered_deduped"

PY="/opt/anaconda3/envs/finance-env/bin/python"

# ✅ NEW fetch/filter script (OpenAI-based)
FETCH_SCRIPT="$BASE_DIR/Telegram_OpenAI_Filter_Translate.py"

DEDUP_SCRIPT="$BASE_DIR/Deduplicate_AfterFetch.py"
OPENAI_DEDUP_SCRIPT="$BASE_DIR/Deduplicate_LMStudio_AfterFetch.py"
ALERT_SCRIPT="$BASE_DIR/Keyword_Alerts_PreFilter.py"
SUM_SCRIPT="$BASE_DIR/Summarize_and_Email.py"
SECRETS="$BASE_DIR/config/secrets.json"

ALERT_LATEST="$ALERT_DIR/alerts_latest.csv"

mkdir -p "$LOG_DIR" "$DATA_DIR" "$ALERT_DIR" "$DEDUP_DIR"
cd "$BASE_DIR"

STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
LOG_OUT="$LOG_DIR/pipeline_${STAMP}.out"
LOG_ERR="$LOG_DIR/pipeline_${STAMP}.err"
TMP_OUT="$LOG_DIR/pipeline_${STAMP}.tmp.out"

ln -sf "$LOG_OUT" "$LOG_DIR/pipeline_latest.out"
ln -sf "$LOG_ERR" "$LOG_DIR/pipeline_latest.err"

echo "=== Daily pipeline started: $(date -u) ===" | tee -a "$LOG_OUT"
echo "BASE_DIR=$BASE_DIR" | tee -a "$LOG_OUT"
echo "PY=$PY" | tee -a "$LOG_OUT"
echo "FETCH_SCRIPT=$FETCH_SCRIPT" | tee -a "$LOG_OUT"
echo "DEDUP_SCRIPT=$DEDUP_SCRIPT" | tee -a "$LOG_OUT"
echo "OPENAI_DEDUP_SCRIPT=$OPENAI_DEDUP_SCRIPT" | tee -a "$LOG_OUT"
echo "ALERT_SCRIPT=$ALERT_SCRIPT" | tee -a "$LOG_OUT"
echo "SUM_SCRIPT=$SUM_SCRIPT" | tee -a "$LOG_OUT"
echo "SECRETS=$SECRETS" | tee -a "$LOG_OUT"
echo "" | tee -a "$LOG_OUT"

# -----------------------------
# Step 1: Fetch/filter via OpenAI
# -----------------------------
echo "Running Telegram_OpenAI_Filter_Translate.py..." | tee -a "$LOG_OUT"

set +e
# Default is 24h; keep that behavior for daily runs
"$PY" "$FETCH_SCRIPT" 24 >"$TMP_OUT" 2>>"$LOG_ERR"
FETCH_RC=$?
set -e

cat "$TMP_OUT" >>"$LOG_OUT"

if [[ $FETCH_RC -ne 0 ]]; then
  echo "ERROR: Telegram_OpenAI_Filter_Translate.py exited with code $FETCH_RC" | tee -a "$LOG_OUT"
  echo "Check: $LOG_ERR" | tee -a "$LOG_OUT"
  exit $FETCH_RC
fi

# -----------------------------
# Step 2: Parse saved CSV from fetch output
# (expects: ✅ Saved N rows → /.../filtered_translated_YYYY-MM-DD.csv)
# -----------------------------
SAVED_CSV="$(grep -Eo '/[^ ]*/filtered_translated_[0-9]{4}-[0-9]{2}-[0-9]{2}\.csv' "$TMP_OUT" | tail -n 1 || true)"

if [[ -z "${SAVED_CSV}" ]]; then
  echo "ERROR: Could not detect saved CSV path from fetch output." | tee -a "$LOG_OUT"
  echo "Expected a line like: ✅ Saved ... → /.../filtered_translated_YYYY-MM-DD.csv" | tee -a "$LOG_OUT"
  echo "Dumping last 80 lines of fetch output:" | tee -a "$LOG_OUT"
  tail -n 80 "$TMP_OUT" | tee -a "$LOG_OUT"
  exit 1
fi

echo "Fetch produced CSV: $SAVED_CSV" | tee -a "$LOG_OUT"

if [[ ! -f "$SAVED_CSV" ]]; then
  echo "ERROR: Parsed CSV does not exist: $SAVED_CSV" | tee -a "$LOG_OUT"
  exit 1
fi

TODAY_UTC="$(date -u +%Y-%m-%d)"
if [[ "$SAVED_CSV" != *"filtered_translated_${TODAY_UTC}.csv" ]]; then
  echo "WARNING: Fetch produced CSV not dated today (UTC). Today is ${TODAY_UTC}." | tee -a "$LOG_OUT"
  echo "         OK if fetch labels by 'yesterday'." | tee -a "$LOG_OUT"
fi

# -----------------------------
# Step 3: Text dedupe (cheap baseline, keeps history)
# -----------------------------
echo "" | tee -a "$LOG_OUT"
echo "Running Deduplicate_AfterFetch.py (text)..." | tee -a "$LOG_OUT"

TEXT_DEDUP_OUT="$DEDUP_DIR/$(basename "$SAVED_CSV" .csv)_deduped_${STAMP}.csv"
TEXT_DEDUP_DUPS="$DEDUP_DIR/dups_$(basename "$SAVED_CSV" .csv)_${STAMP}.csv"

"$PY" "$DEDUP_SCRIPT" \
  --in_csv "$SAVED_CSV" \
  --out_csv "$TEXT_DEDUP_OUT" \
  --dups_report "$TEXT_DEDUP_DUPS" \
  --bucket_by global \
  --sim_threshold 0.90 \
  --jaccard_min 0.40 \
  >>"$LOG_OUT" 2>>"$LOG_ERR"

if [[ ! -s "$TEXT_DEDUP_OUT" ]]; then
  echo "ERROR: Text dedupe output missing/empty: $TEXT_DEDUP_OUT" | tee -a "$LOG_OUT"
  exit 2
fi

echo "Text-deduped CSV: $TEXT_DEDUP_OUT" | tee -a "$LOG_OUT"
echo "Text dups report: $TEXT_DEDUP_DUPS" | tee -a "$LOG_OUT"

# -----------------------------
# Step 4: OpenAI dedupe (THIS IS THE ONE WE USE)
# IMPORTANT: run it on the text-deduped CSV so OpenAI has less to process.
# -----------------------------
echo "" | tee -a "$LOG_OUT"
echo "Running Deduplicate_OpenAI_AfterFetch.py (OpenAI)..." | tee -a "$LOG_OUT"

OPENAI_OUT="$DEDUP_DIR/$(basename "$SAVED_CSV" .csv)_openai_deduped_${STAMP}.csv"
OPENAI_DUPS="$DEDUP_DIR/dups_openai_$(basename "$SAVED_CSV" .csv)_${STAMP}.csv"

if [[ ! -f "$OPENAI_DEDUP_SCRIPT" ]]; then
  echo "ERROR: OpenAI dedupe script not found: $OPENAI_DEDUP_SCRIPT" | tee -a "$LOG_OUT"
  exit 3
fi

"$PY" "$OPENAI_DEDUP_SCRIPT" \
  --in_csv "$TEXT_DEDUP_OUT" \
  --out_csv "$OPENAI_OUT" \
  --dups_report "$OPENAI_DUPS" \
  --bucket_by country \
  --max_per_bucket 250 \
  --base_url http://127.0.0.1:1234/v1 \
  --model llama-3.1-8b-instruct \
  --api_key lm-studio \
  --min_confidence 0.55 \
  --max_compare_per_row 12 \
  --fallback_when_low_conf \
  >>"$LOG_OUT" 2>>"$LOG_ERR"

if [[ ! -s "$OPENAI_OUT" ]]; then
  echo "ERROR: OpenAI dedupe output missing/empty: $OPENAI_OUT" | tee -a "$LOG_OUT"
  echo "Check: $LOG_ERR" | tee -a "$LOG_OUT"
  exit 4
fi

echo "OpenAI-deduped CSV: $OPENAI_OUT" | tee -a "$LOG_OUT"
echo "OpenAI dups report: $OPENAI_DUPS" | tee -a "$LOG_OUT"

# ✅ From here on, use OpenAI output
SAVED_CSV="$OPENAI_OUT"

# -----------------------------
# Step 5: Keyword alerts prefilter (always write alerts_latest.csv)
# -----------------------------
echo "" | tee -a "$LOG_OUT"
echo "Running Keyword_Alerts_PreFilter.py..." | tee -a "$LOG_OUT"

"$PY" "$ALERT_SCRIPT" --csv "$SAVED_CSV" --out_dir "$ALERT_DIR" >>"$LOG_OUT" 2>>"$LOG_ERR"

LATEST_ALERT="$(ls -1t "$ALERT_DIR"/alerts_*.csv 2>/dev/null | head -n 1 || true)"

if [[ -n "$LATEST_ALERT" && -f "$LATEST_ALERT" ]]; then
  cp -f "$LATEST_ALERT" "$ALERT_LATEST"
else
  echo "datetime,channel,message_id,url,kw_terms,snippet" > "$ALERT_LATEST"
fi

ALERT_ROWS="$(($(wc -l < "$ALERT_LATEST") - 1))"
if [[ "$ALERT_ROWS" -lt 0 ]]; then ALERT_ROWS=0; fi
echo "Alerts latest: $ALERT_LATEST (rows=${ALERT_ROWS})" | tee -a "$LOG_OUT"

# -----------------------------
# Step 6: Summarise + email
# -----------------------------
echo "" | tee -a "$LOG_OUT"
echo "Running Summarize_and_Email.py on: $SAVED_CSV" | tee -a "$LOG_OUT"
echo "Command: $PY $SUM_SCRIPT --csv \"$SAVED_CSV\" --hours 24" | tee -a "$LOG_OUT"

set +e
"$PY" "$SUM_SCRIPT" --csv "$SAVED_CSV" --hours 24 >>"$LOG_OUT" 2>>"$LOG_ERR"
SUM_RC=$?
set -e

if [[ $SUM_RC -ne 0 ]]; then
  echo "ERROR: Summarize_and_Email.py exited with code $SUM_RC" | tee -a "$LOG_OUT"
  echo "Check: $LOG_ERR" | tee -a "$LOG_OUT"
  exit $SUM_RC
fi

echo "" | tee -a "$LOG_OUT"
echo "✅ PIPELINE SUCCESS" | tee -a "$LOG_OUT"
echo "=== Daily pipeline finished: $(date -u) ===" | tee -a "$LOG_OUT"
echo "Logs:"
echo "  OUT: $LOG_OUT"
echo "  ERR: $LOG_ERR"
echo "  LATEST OUT: $LOG_DIR/pipeline_latest.out"
echo "  LATEST ERR: $LOG_DIR/pipeline_latest.err"