#!/usr/bin/env python3
import asyncio
import aiohttp
import pandas as pd
import re
from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
import html
import os
import json
from pathlib import Path
from difflib import SequenceMatcher

# ============================= PATHS & CONFIG =================================

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config" / "secrets.json"
cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))

API_ID = cfg["telegram"]["api_id"]
API_HASH = cfg["telegram"]["api_hash"]
LMSTUDIO_URL = cfg["lmstudio"]["url"]

OUTPUT_DIR = ROOT / "data_medical" / "filtered"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MEDICAL_CHANNELS = cfg["telegram_medical"]["channels"]

# How many days back from today to include
# For daily cron runs, keep this at 1.
# For a one-off backfill of the previous month, temporarily set to 30.
DAYS_BACK = 1

# ============================= TIME RANGE =====================================

def time_range_utc():
    today = datetime.now(timezone.utc).date()
    end = datetime(today.year, today.month, today.day, tzinfo=timezone.utc)
    start = end - timedelta(days=DAYS_BACK)
    return start, end

# ============================= LM STUDIO PROMPT ===============================

MEDICAL_CLASSIFIER_PROMPT = """
You are a classifier working for the REGULATORY AFFAIRS department of a multinational pharmaceutical company.

Output ONLY a single compact JSON object and nothing else.

Your job is to detect Telegram messages that matter for:
- the REGULATORY ENVIRONMENT for medicinal products
- DEMAND for medicines and vaccines

Geographic focus ONLY:
Russia, Belarus, Kazakhstan, Kyrgyz Republic, Uzbekistan, Tajikistan, Turkmenistan,
Armenia, Azerbaijan, Georgia, Ukraine, Mongolia.

If the message is NOT primarily about:
- one or more of these countries AND
- regulation, policy, reimbursement, pricing, registration, import/export controls, parallel imports,
  sanctions affecting medicines, tenders/procurement for drugs, or
- demand for medicines (epidemics, outbreaks, shortages, new treatment protocols, large procurement plans),

then return: {"skip": true}

If relevant, classify:

- country: primary country affected (string, one of the Eurasian countries above, or "regional" or "global" if really unclear).
- topic: short comma-separated list using any of:
  regulation, registration, reimbursement, pricing, tenders,
  import_export, sanctions, shortage, demand, epidemic, vaccine,
  hospital_policy, guideline, pharmacovigilance, safety, clinical_trial.
- impact: one of: "high", "medium", "low" – from the perspective of regulatory affairs and market access for medicines.
- english: clear English translation or summary of the message, max 3–4 sentences, focusing ONLY on what matters to:
    - regulatory filings
    - compliance
    - pricing and reimbursement
    - demand for medicines and vaccines

If not relevant, again return exactly: {"skip": true}

If relevant, output JSON in this exact schema:
{
  "skip": false,
  "country": "...",
  "topic": "...",
  "impact": "...",
  "english": "..."
}
"""

async def lmstudio_classify(session, text: str) -> str:
    payload = {
        "messages": [
            {"role": "system", "content": MEDICAL_CLASSIFIER_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": 256,
        "temperature": 0,
    }

    async with session.post(LMSTUDIO_URL, json=payload) as resp:
        data = await resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            print("⚠️ Unexpected LM Studio response:", data)
            return ""

# ============================= TEXT CLEANING ==================================

def clean_html(s):
    return html.unescape(re.sub(r"<.*?>", "", s))

def normalize_text(s):
    return re.sub(r"\s+", " ", s).strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

# ============================= MAIN ===========================================

async def main():
    client = TelegramClient("medical_fetch_session", API_ID, API_HASH)
    await client.start()

    start, end = time_range_utc()
    print(f"Scanning medical channels from {start} to {end} (UTC)")
    rows = []

    async with aiohttp.ClientSession() as session:
        for ch in MEDICAL_CHANNELS:
            print(f"Fetching from {ch}...")
            try:
                async for msg in client.iter_messages(ch, offset_date=end):
                    if msg.date < start:
                        break
                    if not msg.message:
                        continue

                    text = clean_html(msg.message)
                    text = normalize_text(text)
                    if not text:
                        continue

                    lm_raw = await lmstudio_classify(session, text)

                    rows.append({
                        "datetime": msg.date.isoformat(),
                        "channel": ch,
                        "message_id": msg.id,
                        "url": f"https://t.me/{ch.strip('@')}/{msg.id}",
                        "raw": text,
                        "lm_json": lm_raw,
                    })
            except Exception as e:
                print(f"❗ Error processing {ch}: {e}")

    if not rows:
        print("No messages collected.")
        return

    df = pd.DataFrame(rows)

    # Simple deduplication by similarity on 'raw'
    deduped = []
    for _, row in df.iterrows():
        if not any(similarity(row["raw"], r["raw"]) > 0.90 for r in deduped):
            deduped.append(row)

    df2 = pd.DataFrame(deduped)

    outname = OUTPUT_DIR / f"filtered_medical_{start.date().isoformat()}.csv"
    df2.to_csv(outname, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df2)} filtered medical messages → {outname}")

if __name__ == "__main__":
    asyncio.run(main())