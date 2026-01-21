#!/bin/bash
# ==========================================================
# DAILY TELEGRAM PIPELINE — Fully Cron-safe version
# ==========================================================

set -Eeuo pipefail

PROJECT_DIR="/Users/tomadshead/telegramdaily"
ENV_FILE="$PROJECT_DIR/.email_env"
DATE_TAG=$(date +"%y%m%d")
LOG_DIR="$PROJECT_DIR/logs"
LOG_FILE="$LOG_DIR/run_$DATE_TAG.log"

FIRST_SCRIPT="$PROJECT_DIR/Telegram_LMStudio_Filter_Translate1.py"
PDF_SCRIPT="$PROJECT_DIR/Telegram_NewsSummary_toPDF.py"
EMAIL_SCRIPT="$PROJECT_DIR/email_pdf.py"

PDF_OUT="$PROJECT_DIR/MacroAdvisory_News_Summary_${DATE_TAG}.pdf"

PYTHON_BIN="/opt/anaconda3/envs/finance-env/bin/python"

SESSION_FILE="$PROJECT_DIR/telegram.session"
LOCKFILE="/tmp/telegram_daily_news.lock"

mkdir -p "$LOG_DIR"

# ----------------------------------------------------------
# STEP -1: Acquire lock — macOS-safe version (no flock)
# ----------------------------------------------------------
if [ -f "$LOCKFILE" ]; then
  if kill -0 "$(cat "$LOCKFILE")" 2>/dev/null; then
    echo "=== $(date) === Another run_daily_news.sh (PID $(cat "$LOCKFILE")) is still running. Aborting." | tee -a "$LOG_FILE"
    exit 1
  else
    echo "Stale lockfile found. Removing." | tee -a "$LOG_FILE"
    rm -f "$LOCKFILE"
  fi
fi

echo $$ > "$LOCKFILE"
trap "rm -f \"$LOCKFILE\"" EXIT

# ----------------------------------------------------------
# STEP 0: Kill stale Telethon processes if any
# ----------------------------------------------------------
echo "Checking for stale Telethon processes..." | tee -a "$LOG_FILE"
PIDS=$(pgrep -f "Telegram_LMStudio_Filter_Translate1.py" || true)

if [ -n "$PIDS" ]; then
  echo "Found stale Telethon processes: $PIDS" | tee -a "$LOG_FILE"
  echo "Killing them..." | tee -a "$LOG_FILE"
  kill -9 $PIDS || true
fi

# ----------------------------------------------------------
# STEP 1: Remove stale SQLite lock files (macOS-safe)
# ----------------------------------------------------------
echo "Cleaning SQLite lock files..." | tee -a "$LOG_FILE"
rm -f "${SESSION_FILE}-wal" "${SESSION_FILE}-shm"

# ----------------------------------------------------------
# STEP 2: Load email environment variables
# ----------------------------------------------------------
if [ -f "$ENV_FILE" ]; then
  echo "Loading environment variables..." | tee -a "$LOG_FILE"
  # shellcheck source=/dev/null
  source "$ENV_FILE"
else
  echo "ERROR: .email_env file not found" | tee -a "$LOG_FILE"
  exit 1
fi

# ----------------------------------------------------------
# STEP 3: Run Telethon pipeline
# ----------------------------------------------------------
echo "Running first-pass fetch/translate..." | tee -a "$LOG_FILE"

$PYTHON_BIN "$FIRST_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
FIRST_STATUS=${PIPESTATUS[0]}

if [ "$FIRST_STATUS" -ne 0 ]; then
  echo "ERROR: First-pass fetch/translate FAILED (exit $FIRST_STATUS)" | tee -a "$LOG_FILE"
  exit 1
fi

# ----------------------------------------------------------
# STEP 4: Identify CSV output
# ----------------------------------------------------------
INPUT_FILE=$(ls -t "$PROJECT_DIR"/*.csv 2>/dev/null | head -n 1)

if [ ! -f "$INPUT_FILE" ]; then
  echo "ERROR: No CSV found after first step." | tee -a "$LOG_FILE"
  exit 1
fi

echo "Using CSV: $INPUT_FILE" | tee -a "$LOG_FILE"

# ----------------------------------------------------------
# STEP 5: CSV sanity check (must contain header 'country')
# ----------------------------------------------------------
if ! head -n 1 "$INPUT_FILE" | grep -q "country"; then
  echo "ERROR: CSV does not contain expected header 'country'. Aborting." | tee -a "$LOG_FILE"
  exit 1
fi

# ----------------------------------------------------------
# STEP 6: Generate PDF
# ----------------------------------------------------------
echo "Generating PDF: $PDF_OUT ..." | tee -a "$LOG_FILE"

$PYTHON_BIN "$PDF_SCRIPT" "$INPUT_FILE" "$PDF_OUT" 2>&1 | tee -a "$LOG_FILE"

if [ ! -f "$PDF_OUT" ]; then
  echo "ERROR: PDF not created!" | tee -a "$LOG_FILE"
  exit 1
fi

echo "PDF confirmed at $PDF_OUT" | tee -a "$LOG_FILE"

# ----------------------------------------------------------
# STEP 7: Email PDF
# ----------------------------------------------------------
echo "Sending email..." | tee -a "$LOG_FILE"

if ! $PYTHON_BIN "$EMAIL_SCRIPT" "$PDF_OUT" 2>&1 | tee -a "$LOG_FILE"; then
  echo "ERROR: Email sending FAILED" | tee -a "$LOG_FILE"
  exit 1
else
  echo "Email sent successfully." | tee -a "$LOG_FILE"
fi

echo "=== Run complete at $(date) ===" | tee -a "$LOG_FILE"