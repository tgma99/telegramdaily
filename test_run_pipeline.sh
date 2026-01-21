#!/bin/bash
set -euo pipefail

# ------------------------------------------------------------
# test_run_pipeline.sh
# End-to-end test runner for telegramdaily pipeline:
#  1) Telegram_Fetch_Filter.py
#  2) Keyword_Alerts_PreFilter.py
#  3) Summarize_and_Email.py
#
# Supports:
#  - preview mode override (no SMTP send)
#  - tail logs at end
# ------------------------------------------------------------

BASE_DIR="${HOME}/telegramdaily"
PY="/opt/anaconda3/envs/finance-env/bin/python"

FETCH_SCRIPT="${BASE_DIR}/Telegram_Fetch_Filter.py"
ALERT_SCRIPT="${BASE_DIR}/Keyword_Alerts_PreFilter.py"
SUM_SCRIPT="${BASE_DIR}/Summarize_and_Email.py"

SECRETS="${BASE_DIR}/config/secrets.json"
ALERT_DIR="${BASE_DIR}/data/alerts"
LOG_DIR="${BASE_DIR}/logs"
TMP_DIR="${LOG_DIR}/tmp"

mkdir -p "${LOG_DIR}" "${TMP_DIR}" "${ALERT_DIR}"
cd "${BASE_DIR}"

STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
OUT="${LOG_DIR}/test_pipeline_${STAMP}.out"
ERR="${LOG_DIR}/test_pipeline_${STAMP}.err"
FETCH_STDOUT="${TMP_DIR}/fetch_${STAMP}.stdout"

# --- options ---
# If PREVIEW_ONLY=1, we will temporarily force mail_mode="preview"
PREVIEW_ONLY="${PREVIEW_ONLY:-1}"
HOURS="${HOURS:-24}"

echo "=== TEST PIPELINE START $(date -u) ===" | tee -a "${OUT}"
echo "BASE_DIR=${BASE_DIR}" | tee -a "${OUT}"
echo "PY=${PY}" | tee -a "${OUT}"
echo "SECRETS=${SECRETS}" | tee -a "${OUT}"
echo "PREVIEW_ONLY=${PREVIEW_ONLY}" | tee -a "${OUT}"
echo "HOURS=${HOURS}" | tee -a "${OUT}"
echo "" | tee -a "${OUT}"

# Basic sanity checks
for f in "${FETCH_SCRIPT}" "${ALERT_SCRIPT}" "${SUM_SCRIPT}" "${SECRETS}"; do
  if [[ ! -f "$f" ]]; then
    echo "❌ Missing required file: $f" | tee -a "${OUT}"
    exit 1
  fi
done

if [[ ! -x "${PY}" ]]; then
  echo "❌ Python not found/executable: ${PY}" | tee -a "${OUT}"
  exit 1
fi

# Optional: force preview mode for safe testing
BACKUP_SECRETS=""
if [[ "${PREVIEW_ONLY}" == "1" ]]; then
  echo "Forcing mail_mode=preview for this test run..." | tee -a "${OUT}"
  BACKUP_SECRETS="${SECRETS}.bak_${STAMP}"
  cp -f "${SECRETS}" "${BACKUP_SECRETS}"

  # very small JSON edit using python (safe; preserves formatting minimally)
  "${PY}" - <<PY
import json
p="${SECRETS}"
with open(p,"r",encoding="utf-8") as f:
    d=json.load(f)
d["mail_mode"]="preview"
# ensure alerts_csv points to stable latest file (Summarize reads this)
d["alerts_csv"]="data/alerts/alerts_latest.csv"
with open(p,"w",encoding="utf-8") as f:
    json.dump(d,f,ensure_ascii=False,indent=2)
print("✅ secrets.json temporarily updated for preview mode")
PY
fi

# -----------------------------
# Step 1: Fetch
# -----------------------------
echo "" | tee -a "${OUT}"
echo ">>> Step 1: Telegram_Fetch_Filter.py" | tee -a "${OUT}"
set +e
"${PY}" "${FETCH_SCRIPT}" > "${FETCH_STDOUT}" 2>> "${ERR}"
RC=$?
set -e
cat "${FETCH_STDOUT}" >> "${OUT}"

if [[ $RC -ne 0 ]]; then
  echo "❌ Fetch failed with exit code $RC" | tee -a "${OUT}"
  echo "See: ${ERR}" | tee -a "${OUT}"
  [[ -n "${BACKUP_SECRETS}" ]] && mv -f "${BACKUP_SECRETS}" "${SECRETS}"
  exit $RC
fi

# Parse CSV path from fetch stdout (expects "... filtered_YYYY-MM-DD.csv")
SAVED_CSV="$(grep -Eo '/[^ ]*/filtered_[0-9]{4}-[0-9]{2}-[0-9]{2}\.csv' "${FETCH_STDOUT}" | tail -n 1 || true)"
if [[ -z "${SAVED_CSV}" ]]; then
  echo "❌ Could not detect output CSV path from fetch output." | tee -a "${OUT}"
  echo "Last 50 lines of fetch output:" | tee -a "${OUT}"
  tail -n 50 "${FETCH_STDOUT}" | tee -a "${OUT}"
  [[ -n "${BACKUP_SECRETS}" ]] && mv -f "${BACKUP_SECRETS}" "${SECRETS}"
  exit 1
fi

if [[ ! -f "${SAVED_CSV}" ]]; then
  echo "❌ Parsed CSV does not exist: ${SAVED_CSV}" | tee -a "${OUT}"
  [[ -n "${BACKUP_SECRETS}" ]] && mv -f "${BACKUP_SECRETS}" "${SECRETS}"
  exit 1
fi

echo "✅ Fetch produced CSV: ${SAVED_CSV}" | tee -a "${OUT}"

# -----------------------------
# Step 2: Keyword Alerts (pre-filter)
# -----------------------------
echo "" | tee -a "${OUT}"
echo ">>> Step 2: Keyword_Alerts_PreFilter.py" | tee -a "${OUT}"

"${PY}" "${ALERT_SCRIPT}" --csv "${SAVED_CSV}" --out_dir "${ALERT_DIR}" --write_flagged_source >> "${OUT}" 2>> "${ERR}"

LATEST_ALERT="$(ls -1t "${ALERT_DIR}"/alerts_*.csv 2>/dev/null | head -n 1 || true)"
if [[ -n "${LATEST_ALERT}" ]]; then
  cp -f "${LATEST_ALERT}" "${ALERT_DIR}/alerts_latest.csv"
  echo "✅ alerts_latest.csv updated -> ${ALERT_DIR}/alerts_latest.csv" | tee -a "${OUT}"
else
  echo "⚠️ No alerts_*.csv produced (maybe zero matches)." | tee -a "${OUT}"
fi

# -----------------------------
# Step 3: Summarize + Email/Preview
# -----------------------------
echo "" | tee -a "${OUT}"
echo ">>> Step 3: Summarize_and_Email.py" | tee -a "${OUT}"
echo "Command: ${PY} ${SUM_SCRIPT} --csv \"${SAVED_CSV}\" --hours ${HOURS} --secrets \"${SECRETS}\"" | tee -a "${OUT}"

set +e
"${PY}" "${SUM_SCRIPT}" --csv "${SAVED_CSV}" --hours "${HOURS}" --secrets "${SECRETS}" >> "${OUT}" 2>> "${ERR}"
RC=$?
set -e

if [[ $RC -ne 0 ]]; then
  echo "❌ Summarize failed with exit code $RC" | tee -a "${OUT}"
  echo "See: ${ERR}" | tee -a "${OUT}"
  [[ -n "${BACKUP_SECRETS}" ]] && mv -f "${BACKUP_SECRETS}" "${SECRETS}"
  exit $RC
fi

echo "✅ Summarize step completed." | tee -a "${OUT}"

# Restore secrets.json if we changed it
if [[ -n "${BACKUP_SECRETS}" ]]; then
  mv -f "${BACKUP_SECRETS}" "${SECRETS}"
  echo "✅ secrets.json restored to original." | tee -a "${OUT}"
fi

echo "" | tee -a "${OUT}"
echo "=== TEST PIPELINE SUCCESS $(date -u) ===" | tee -a "${OUT}"
echo "OUT: ${OUT}"
echo "ERR: ${ERR}"

echo ""
echo "---- Tail OUT ----"
tail -n 80 "${OUT}" || true
echo ""
echo "---- Tail ERR ----"
tail -n 80 "${ERR}" || true

# If preview was generated, try to open it (macOS)
PREVIEW_PATH="$(/usr/bin/python3 - <<PY
import json,sys
p="${SECRETS}"
try:
  d=json.load(open(p,"r",encoding="utf-8"))
  print(d.get("preview_out",""))
except Exception:
  print("")
PY
)"

if [[ -n "${PREVIEW_PATH}" && -f "${PREVIEW_PATH}" ]]; then
  echo ""
  echo "Preview HTML exists: ${PREVIEW_PATH}"
  echo "Open with: open \"${PREVIEW_PATH}\""
fi