#!/usr/bin/env python3
import asyncio
import aiohttp
import pandas as pd
import re
from telethon import TelegramClient
from datetime import datetime, timedelta, timezone
import html
import os

from pathlib import Path
from difflib import SequenceMatcher
import ast, json


# ============================= LOAD CONFIG ====================================


CONFIG_PATH = Path(__file__).resolve().parent / "config" / "secrets.json"
cfg = json.loads(open(CONFIG_PATH).read())

API_ID = cfg["telegram"]["api_id"]
API_HASH = cfg["telegram"]["api_hash"]
LMSTUDIO_URL = cfg["lmstudio"]["url"]

OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "filtered"
OUTPUT_DIR.mkdir(exist_ok=True)

CLASSIFIER_PROMPT = """
You are a strict government-relations and macro-policy news filter.

You will receive ONE Telegram post (often short; may be Russian or other language).
Your job is to decide whether it is relevant for a multinational company’s head of government relations.

ONLY keep posts that match at least ONE of the allowed relevance types below.

ALLOWED RELEVANCE TYPES (keep = true):
A) POLICY / LAW / REGULATION:
- new laws, decrees, draft bills, regulations, enforcement actions
- sanctions, export/import controls, licensing, foreign investment rules
- major government programmes, industrial policy, state aid, taxation changes
- court / prosecutor actions that create or change policy direction

B) CENTRAL GOVERNMENT PERSONNEL / INSTITUTIONAL CHANGES:
- appointments, dismissals, resignations, reshuffles
- creation/abolition of ministries/agencies, reorganisations
- leadership changes in key regulators or SOEs (central level)

C) MACROECONOMIC DATA / POLICY (NOT routine market ticks):
- inflation, GDP, industrial output, wages, employment, budget, deficit, debt, reserves,
  current account, trade, FDI, major forecasts by central bank/finance ministry/IMF/EBRD
- central bank policy decisions (rates, capital controls, major regulation)
- IMPORTANT: exclude routine daily exchange-rate updates and “today’s FX rates”.

ALWAYS SKIP (keep = false):
- holidays, New Year greetings, fireworks, celebrations, cultural events
- sports results (even if famous people)
- crimes/accidents/fires/weather disruptions unless they trigger national policy response
- local municipal issues unless clearly national-level policy or central government action
- generic human-interest stories, social media content, promotional/advertising posts
- routine “rates/quotes/today’s exchange rate/FX table” posts

COUNTRIES:
Choose ONE primary country from this list exactly:
Russia, Kazakhstan, Kyrgyz Republic, Uzbekistan, Tajikistan, Turkmenistan, Azerbaijan, Georgia, Armenia, Ukraine, Belarus.
If unclear or multi-country, use "unknown". Do NOT output countries outside this list.

OUTPUT:
Return ONLY one compact JSON object and nothing else.

If NOT relevant: {"skip": true}

If relevant:
{
  "skip": false,
  "country": "<one country from list or unknown>",
  "category": "<one of: policy, personnel, macro>",
  "subject": "<for macro only: comma-separated from: GDP, inflation, interest rates, reserves, current account, budget, deficit, debt, employment, wages, industrial production, trade, exports, imports, investment, FDI; else empty string>",
  "english": "<clear English translation, remove emojis/decorative symbols>"
}

Post:
<PASTE POST HERE>
"""

# ============================= FUNCTIONS ======================================

def yesterday_range_utc():
    today = datetime.now(timezone.utc).date()
    start = datetime(today.year, today.month, today.day, tzinfo=timezone.utc) - timedelta(days=1)
    end = start + timedelta(days=1)
    return start, end

async def lmstudio_classify(session, text: str) -> str:
    """
    Call LM Studio with the classifier prompt and return the raw JSON string
    the model outputs. We expect it to follow the schema described in
    CLASSIFIER_PROMPT.
    """
    payload = {
        "messages": [
            {"role": "system", "content": CLASSIFIER_PROMPT},
            {"role": "user", "content": text},
        ],
        "max_tokens": 256,
        "temperature": 0
        # If your LM Studio endpoint requires a model name, you can add:
        # "model": "YOUR_MODEL_NAME"
    }

    async with session.post(LMSTUDIO_URL, json=payload) as resp:
        data = await resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except Exception:
            # Failsafe: log whatever we got back and return empty string
            print("⚠️ Unexpected LM Studio response:", data)
            return ""

def clean_html(s):
    return html.unescape(re.sub(r"<.*?>", "", s))

def normalize_text(s):
    return re.sub(r"\s+", " ", s).strip()

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def safe_parse_lm_json(x):
    if isinstance(x, dict):
        return x
    if x is None:
        return {}
    if isinstance(x, float) and pd.isna(x):
        return {}
    if not isinstance(x, str):
        return {}
    s = x.strip()
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        # try extract first {...}
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                pass
        # try python literal
        try:
            v = ast.literal_eval(s)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

# ============================= MAIN ===========================================

async def main():
    client = TelegramClient("fetch_session", API_ID, API_HASH)
    await client.start()

    start, end = yesterday_range_utc()

    channels = cfg["telegram"]["channels"]
    rows = []

    async with aiohttp.ClientSession() as session:
        for ch in channels:
            try:
                async for msg in client.iter_messages(ch, offset_date=end):
                    if msg.date < start:
                        break
                    if not msg.message:
                        continue

                    text = clean_html(msg.message)
                    text = normalize_text(text)

                    # Call LM Studio (returns raw JSON string or similar)
                    lm_raw = await lmstudio_classify(session, text)

                    rows.append({
                        "datetime": msg.date.isoformat(),
                        "channel": ch,
                        "message_id": msg.id,
                        "url": f"https://t.me/{ch.strip('@')}/{msg.id}",
                        "raw": text,
                        "lm_json": lm_raw
                    })

            except Exception as e:
                print(f"❗ Error processing {ch}: {e}")

    # -----------------------------
    # Convert to DF
    # -----------------------------
    df = pd.DataFrame(rows)
    if df.empty:
        outname = OUTPUT_DIR / f"filtered_{start.date().isoformat()}.csv"
        outname.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outname, index=False, encoding="utf-8")
        print(f"✅ Saved 0 filtered messages → {outname}")
        return

    # Parse datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    # -----------------------------
    # NEW: Parse lm_json → real columns + REAL filtering
    # -----------------------------
    parsed = df["lm_json"].map(safe_parse_lm_json) if "lm_json" in df.columns else [{}] * len(df)

    df["skip"] = [bool((p or {}).get("skip", True)) for p in parsed]
    df["country"] = [((p or {}).get("country", "unknown") or "unknown") for p in parsed]
    df["category"] = [((p or {}).get("category", "") or "") for p in parsed]
    df["subject"] = [((p or {}).get("subject", "") or "") for p in parsed]
    df["english"] = [((p or {}).get("english", "") or "") for p in parsed]

    # Optional: normalize legacy category names if your prompt still outputs them
    CATEGORY_MAP = {
        "foreign relations": "policy",
        "foreign_relations": "policy",
        "domestic politics": "policy",
        "macroeconomics": "macro",
    }
    df["category"] = df["category"].map(lambda x: CATEGORY_MAP.get(x, x))

    # ✅ ACTUAL filter based on lm_json.skip
    df = df[df["skip"] == False].copy()

    if df.empty:
        outname = OUTPUT_DIR / f"filtered_{start.date().isoformat()}.csv"
        outname.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(outname, index=False, encoding="utf-8")
        print(f"✅ Saved 0 filtered messages → {outname}")
        return

    # -----------------------------
    # Deduplicate (text similarity) AFTER filtering
    # Prefer english if present; fallback to raw
    # -----------------------------
    def row_text(r):
        e = str(r.get("english", "") or "").strip()
        return e if e else str(r.get("raw", "") or "").strip()

    df["_dedupe_text"] = df.apply(row_text, axis=1)

    deduped_rows = []
    for _, row in df.iterrows():
        t = row["_dedupe_text"]
        if not t:
            continue

        # compare against already kept rows
        is_dup = False
        for kept in deduped_rows:
            if similarity(t, kept["_dedupe_text"]) > 0.90:
                is_dup = True
                break

        if not is_dup:
            deduped_rows.append(row)

    df2 = pd.DataFrame(deduped_rows).drop(columns=["_dedupe_text"], errors="ignore")

    # -----------------------------
    # Save
    # -----------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outname = OUTPUT_DIR / f"filtered_{start.date().isoformat()}.csv"
    df2.to_csv(outname, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df2)} filtered messages → {outname}")


if __name__ == "__main__":
    asyncio.run(main())