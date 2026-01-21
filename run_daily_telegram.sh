#!/bin/bash
ROOT="/Users/tomadshead/telegramdaily"
PYTHON="/opt/anaconda3/envs/finance-env/bin/python"

cd "$ROOT"
mkdir -p "$ROOT/logs"

echo "=== $(date) === Starting Telegram daily pipeline" >> "$ROOT/logs/telegram_daily.log" 2>&1

# Macro/political news
"$PYTHON" "$ROOT/Telegram_Fetch_Filter.py" >> "$ROOT/logs/telegram_daily.log" 2>&1
"$PYTHON" "$ROOT/Summarize_and_Email.py" >> "$ROOT/logs/telegram_daily.log" 2>&1

# Medical news
"$PYTHON" "$ROOT/Telegram_Medical_Fetch_Filter.py" >> "$ROOT/logs/telegram_daily.log" 2>&1
"$PYTHON" "$ROOT/Medical_Summarize_and_Email.py" >> "$ROOT/logs/telegram_daily.log" 2>&1

echo "=== $(date) === Finished Telegram daily pipeline" >> "$ROOT/logs/telegram_daily.log" 2>&1