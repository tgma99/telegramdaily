#!/usr/bin/env python3
"""
Medical_Summarize_and_Email.py

Pharma survey (30-day interval, 30-day lookback).

- Reads medical/pharma CSVs from data_medical/filtered
- Only runs the survey if the last run was >= 30 days ago
- When it runs, it uses only the last 30 days of data
- Sends a plain-text email with the survey results
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

def get_last_survey_run(state_file: Path):
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


from datetime import datetime, timedelta, timezone  # make sure timezone is imported at top

def filter_last_n_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """
    Filter the DataFrame to only include rows from the last `days` days,
    based on the 'datetime' column.
    """
    if df.empty:
        return df

    if "datetime" not in df.columns:
        raise RuntimeError("Expected 'datetime' column in medical CSVs")

    df = df.copy()
    # Make this column timezone-aware in UTC
    df["datetime_parsed"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    # Make cutoff also timezone-aware in UTC
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)

    # Now both sides are tz-aware UTC, comparison is valid
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
    Pharma survey logic over the last 30 days of medical messages.

    `df_30d` contains ONLY the last 30 days of messages.

    This function:
    - filters for pharma-related content (using a list of keywords)
    - compiles simple counts by company and by country
    - returns a plaintext survey report as a string
    """

    if df_30d.empty:
        return "No medical or pharma messages in the last 30 days."

    PHARMA_KEYWORDS = [
        "Pfizer", "Moderna", "AstraZeneca", "Novartis", "Roche",
        "Sanofi", "Johnson & Johnson", "Gilead", "Merck", "Bayer",
        "Eli Lilly", "AbbVie", "Bristol-Myers Squibb", "GlaxoSmithKline",
    ]

    # --- Build a robust text column even if english/original are missing ---

    # english part
    if "english" in df_30d.columns:
        eng_series = df_30d["english"].fillna("").astype(str)
    elif "text" in df_30d.columns:
        # fallback if you ever have a generic 'text' column
        eng_series = df_30d["text"].fillna("").astype(str)
    else:
        eng_series = pd.Series([""] * len(df_30d), index=df_30d.index)

    # original part
    if "original" in df_30d.columns:
        orig_series = df_30d["original"].fillna("").astype(str)
    else:
        orig_series = pd.Series([""] * len(df_30d), index=df_30d.index)

    text_col = eng_series + " " + orig_series

    # --- Filter for pharma-related messages ---

    mask_any = False
    for kw in PHARMA_KEYWORDS:
        mask_any |= text_col.str.contains(kw, case=False, na=False)

    pharma_df = df_30d[mask_any].copy()

    if pharma_df.empty:
        return "No pharma-related messages found in the last 30 days."

    lines = []
    lines.append("Pharma Survey – Last 30 Days")
    lines.append("")
    lines.append(
        "This report covers pharma-related mentions in the last 30 days, "
        "based on Telegram medical channels in your pipeline."
    )
    lines.append("")

    # --- Per keyword counts (computed on text_col) ---

    lines.append("Mentions by company/keyword:")
    for kw in PHARMA_KEYWORDS:
        count_kw = text_col.str.contains(kw, case=False, na=False).sum()
        if count_kw > 0:
            lines.append(f"- {kw}: {count_kw} messages")
    lines.append("")

    # --- Per country counts (if present) ---

    if "country" in pharma_df.columns:
        lines.append("Mentions by country:")
        country_counts = pharma_df["country"].fillna("unknown").value_counts()
        for country, cnt in country_counts.items():
            lines.append(f"- {country}: {cnt} messages")
        lines.append("")

    # --- Sample items ---

    lines.append("Sample items:")
    # If 'datetime' is missing, sort_index() is a safe fallback
    if "datetime" in pharma_df.columns:
        sample = pharma_df.sort_values("datetime").head(10)
    else:
        sample = pharma_df.head(10)

    for _, row in sample.iterrows():
        dt = row.get("datetime", "")
        ch = row.get("channel", "")
        url = row.get("url", "")

        # row.get will happily return None if missing
        english = row.get("english") or row.get("original") or row.get("text") or ""
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

    # 2) Load all historical medical CSVs
    df_all = load_all_medical_data()

    # 3) Filter to last 30 days
    df_30d = filter_last_n_days(df_all, days=30)

    # 4) Generate survey text
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