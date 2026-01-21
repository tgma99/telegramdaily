
Telegram Daily Pipeline

A fully automated pipeline for fetching, classifying, translating, deduplicating, alerting, and emailing high-volume Telegram news feeds, designed for macro-economic and geopolitical monitoring.

Built for:
	•	macOS (Apple Silicon supported)
	•	Python + Conda
	•	OpenAI and/or local LLMs via LM Studio
	•	Daily unattended execution via launchd

⸻

📐 Architecture Overview

Telegram channels
      ↓
Telegram_OpenAI_Filter_Translate.py
  (fetch + classify + translate)
      ↓
Deduplicate_AfterFetch.py
  (cheap text dedupe)
      ↓
Deduplicate_LMStudio_AfterFetch.py
  (semantic LLM dedupe)
      ↓
Keyword_Alerts_PreFilter.py
  (keyword alerts)
      ↓
Summarize_and_Email.py
  (email + PDF/HTML output)

All orchestration is handled by:

run_daily_pipeline.sh


⸻

🧠 LLM Strategy

The pipeline supports two interchangeable backends:

Option A — OpenAI API

Used for:
	•	classification
	•	translation
	•	deduplication

Configured via config/secrets.json.

Option B — Local LLM via LM Studio (recommended)

Used for:
	•	semantic deduplication
	•	optionally classification/translation

Advantages:
	•	Zero marginal cost
	•	Faster iteration
	•	No data leaves machine

Tested models:
	•	llama-3.1-8b-instruct
	•	qwen2.5-7b-instruct

LM Studio must expose an OpenAI-compatible endpoint:

http://127.0.0.1:1234/v1


⸻

🛠 Environment Setup

1) Conda environment

conda create -n finance-env python=3.11
conda activate finance-env
pip install -r requirements.txt

Key dependencies:
	•	telethon
	•	openai
	•	pandas
	•	python-dateutil

⸻

2) Secrets configuration (NOT committed)

Create:

config/secrets.json

Example:

{
  "openai_api_key": "sk-REPLACE",
  "mail_mode": "smtp",
  "smtp_host": "smtp.example.com",
  "smtp_user": "user",
  "smtp_pass": "password",
  "preview_out": "/Users/USERNAME/telegramdaily/out_email_preview.html"
}

A template is provided:

config/secrets.example.json


⸻

🚀 Running the Pipeline Manually

chmod +x run_daily_pipeline.sh
./run_daily_pipeline.sh

Outputs:
	•	logs → logs/
	•	filtered CSVs → data/filtered/
	•	deduped CSVs → data/filtered_deduped/
	•	alerts → data/alerts/

⸻

⏰ Daily Automation (launchd)

1) Create plist

Example:

<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.telegramdaily.run</string>

  <key>ProgramArguments</key>
  <array>
    <string>/Users/USERNAME/telegramdaily/run_daily_pipeline.sh</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key><integer>2</integer>
    <key>Minute</key><integer>0</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>/Users/USERNAME/telegramdaily/logs/launchd.out</string>

  <key>StandardErrorPath</key>
  <string>/Users/USERNAME/telegramdaily/logs/launchd.err</string>

  <key>RunAtLoad</key>
  <true/>
</dict>
</plist>

Save as:

~/Library/LaunchAgents/com.telegramdaily.run.plist

2) Load it

launchctl load ~/Library/LaunchAgents/com.telegramdaily.run.plist

Check:

launchctl list | grep telegramdaily


⸻

📊 Deduplication Logic

Stage 1 — Text Deduplication

Script:

Deduplicate_AfterFetch.py

Techniques:
	•	SequenceMatcher
	•	Jaccard similarity
	•	Fast, cheap, conservative

Stage 2 — Semantic LLM Deduplication

Script:

Deduplicate_LMStudio_AfterFetch.py

Key safeguards:
	•	Bucketing by country (fallback to channel)
	•	Max comparisons per row
	•	Confidence threshold
	•	Conservative default = keep

This avoids over-deduplication of unrelated news.

⸻

🚨 Keyword Alerts

Script:

Keyword_Alerts_PreFilter.py

Outputs:
	•	rolling alerts_latest.csv
	•	per-run timestamped alerts

Designed for:
	•	company monitoring
	•	sanctions
	•	policy triggers
	•	client-specific watchlists

⸻

✉️ Email Output

Script:

Summarize_and_Email.py

Supports:
	•	SMTP send
	•	Preview-only HTML mode
	•	PDF attachment generation
	•	Per-category summaries

⸻

🧹 Git Hygiene (Important)

This repo intentionally ignores:
	•	data/
	•	logs/
	•	*.csv
	•	*.log
	•	config/secrets.json

Secrets must never be committed.

GitHub push protection is enforced.

⸻

🧭 Known Gotchas
	•	Telethon SQLite lock: never run multiple fetchers simultaneously
	•	LM Studio 400 errors: ensure no response_format argument is sent
	•	launchd PATH issues: always use absolute paths

⸻

🧱 Suggested Next Enhancements
	•	Pre-commit hooks to block secrets
	•	Embedding-based dedupe cache
	•	SQLite state DB instead of JSON
	•	Per-client alert profiles
	•	Daily PDF bundling

⸻

🏷 Versioning
	•	v1.0-clean — clean, secrets-free baseline
	•	main — active development

⸻

📜 License

Private / internal use.
No warranty.

