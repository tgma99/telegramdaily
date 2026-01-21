#!/usr/bin/env python3
"""
Pharma survey runner (30-day interval, 30-day lookback).

- Reads medical/pharma CSVs from data_medical/filtered
- Only runs the survey if !=less than 30 days since last run
- When it runs, it uses only the last 30 days of data
"""

import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ============================= PATHS & CONFIG ================================

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "secrets.json"
INPUT_DIR = ROOT / "data_medical" / "filtered"
STATE_FILE = ROOT / "data_medical" / "pharma_survey_last_run.txt"

print("Using config:", CONFIG_PATH)
print("Looking for medical filtered CSVs in:", INPUT_DIR)

cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

EMAIL_FROM = cfg["email"]["from"]
EMAIL_TO = cfg["email"]["recipients"]
SMTP_SERVER = cfg["email"]["smtp_server"]
SMTP_PORT = cfg["email"]["smtp_port"]
SMTP_USER = cfg["email"]["username"]
SMTP_PASS = cfg["email"]["password"]


# ============================= TIME HELPERS ==================================

def get_last_survey_run(state_file: Path) -> datetime | None:
    """
    Reads last survey run timestamp from the state file.
    Returns None if file is missing or invalid.
    """
    try:
        if not state_file.exists():
            return None
        raw = state_file.read_text().strip()
        if not raw:
            return None
        return datetime.fromisoformat(raw)
    except Exception:
        return None


def should_run_pharma_survey(
    state_file: Path,
    interval_days: int = 30,
) -> bool:
    """
    Returns True if the pharma survey should run now:
    - state file missing, or
    - last run is >= interval_days ago
    """
    last_run = get_last_survey_run(state_file)
    now = datetime.utcnow()

    if last_run is None:
        return True

    return (now - last_run) >= timedelta(days=interval_days)


def update_last_survey_run(state_file: Path) -> None:
    """
    Writes the current UTC timestamp to the state file.
    """
    now = datetime.utcnow().isoformat()
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(now)


# ============================= DATA LOADING ==================================

def load_all_medical_data() -> pd.DataFrame:
    """
    Load all filtered_medical_*.csv files into a single DataFrame.
    Assumes they all share the same schema.
    """
    files = sorted(INPUT_DIR.glob("filtered_medical_*.csv"))
    print("Found medical CSV files:", [f.name for f in files])

    if not files:
        raise RuntimeError("No filtered_medical_*.csv files found")

    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    return df


def filter_last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows from the last `days` days,
    based on the 'datetime' column.
    """
    if df.empty:
        return df

    # Parse datetime column robustly
    if "datetime" not in df.columns:
        raise RuntimeError("Expected 'datetime' column in medical CSVs")

    df = df.copy()
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    now = datetime.utcnow()
    cutoff = now - timedelta(days=days)

    mask = df["datetime_parsed"] >= cutoff
    filtered = df[mask].copy()

    print(
        f"Filtered to last {days} days: "
        f"{len(filtered)} / {len(df)} rows remain"
    )

    return filtered


# ============================= EMAIL SENDER ==================================

def send_email(subject: str, body: str) -> None:
    """
    Simple plaintext email sender.
    """
    msg = MIMEMultipart()
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())


# ============================= PHARMA SURVEY LOGIC ===========================

def run_pharma_survey(df_30d: pd.DataFrame) -> str:
    """
    YOUR EXISTING SURVEY LOGIC GOES HERE.

    `df_30d` contains ONLY the last 30 days of medical messages.

    Typical columns (based on your other scripts):
    - datetime
    - channel
    - message_id
    - url
    - category
    - country
    - subject
    - original
    - english

    This function should:
    - filter for pharma-related content (e.g. companies or tags)
    - compile counts / summaries
    - return a plaintext string that will be used as the email body
    """

    if df_30d.empty:
        return "No medical or pharma messages in the last 30 days."

    # Example stub: group by channel and count messages mentioning key pharma words
    # Replace this with your real logic
    PHARMA_KEYWORDS = [
        "Pfizer", "Moderna", "AstraZeneca", "Novartis", "Roche",
        "Sanofi", "Johnson & Johnson", "Gilead", "Merck", "Bayer",
    ]

    text_col = (
        df_30d.get("english", "").fillna("").astype(str)
        + " "
        + df_30d.get("original", "").fillna("").astype(str)
    )

    mask_any = False
    for kw in PHARMA_KEYWORDS:
        mask_any |= text_col.str.contains(kw, case=False, na=False)

    pharma_df = df_30d[mask_any].copy()

    if pharma_df.empty:
        return "No pharma-related messages found in the last 30 days."

    # Simple example: count messages per keyword & per country
    lines = []
    lines.append("Pharma Survey – Last 30 Days")
    lines.append("")

    # Per keyword counts
    lines.append("Mentions by company/keyword:")
    for kw in PHARMA_KEYWORDS:
        count_kw = text_col.str.contains(kw, case=False, na=False).sum()
        if count_kw > 0:
            lines.append(f"- {kw}: {count_kw} messages")
    lines.append("")

    # Per country counts (if country column exists)
    if "country" in pharma_df.columns:
        lines.append("Mentions by country:")
        country_counts = pharma_df["country"].fillna("unknown").value_counts()
        for country, cnt in country_counts.items():
            lines.append(f"- {country}: {cnt} messages")
        lines.append("")

    # Optionally list top few individual stories
    lines.append("Sample items:")
    sample = pharma_df.sort_values("datetime").head(10)
    for _, row in sample.iterrows():
        dt = row.get("datetime", "")
        ch = row.get("channel", "")
        url = row.get("url", "")
        english = row.get("english") or row.get("original") or ""
        english = str(english).strip().replace("\n", " ")
        line = f"- {dt} {ch}: {english}"
        if url:
            line += f" ({url})"
        lines.append(line)

    return "\n".join(lines)


# ============================= MAIN ==========================================

def main():
    # 1) Check if we should run at all
    if not should_run_pharma_survey(STATE_FILE, interval_days=30):
        last = get_last_survey_run(STATE_FILE)
        print("Skipping pharma survey: last run was less than 30 days ago.")
        if last is not None:
            print(f"Last run at (UTC): {last.isoformat()}")
        return

    print("Running pharma survey – 30-day interval, 30-day lookback...")

    # 2) Load data (all historical medical CSVs)
    df_all = load_all_medical_data()

    # 3) Filter to last 30 days
    df_30d = filter_last_n_days(df_all, days=30)

    # 4) Generate survey text from last 30 days only
    survey_body = run_pharma_survey(df_30d)

    run_date = datetime.utcnow().strftime("%Y-%m-%d")
    subject = f"Pharma Survey – Last 30 Days (as of {run_date})"

    print("---------- EMAIL PREVIEW ----------")
    print("Subject:", subject)
    print(survey_body[:4000])
    print("---------- END PREVIEW ------------")

    # 5) Send email
    send_email(subject, survey_body)
    print("📨 Pharma survey email sent")

    # 6) Update last-run timestamp
    update_last_survey_run(STATE_FILE)
    print("Updated last-run timestamp.")


if __name__ == "__main__":
    main()