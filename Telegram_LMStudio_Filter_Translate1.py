#!/usr/bin/env python3
# ======================================
# Telegram_LMStudio_Filter_Translate1.py
#
# Fetches Telegram messages, calls LM Studio for
# classification + translation,_FILTERS OUT PRICE UPDATES_,
# cleans Unicode, and writes a clean CSV:
#   filtered_translated_macro.csv
#
# It also maintains channel_state.json with per-channel last_id
# so each run only processes new messages.
# ======================================

import asyncio
import aiohttp
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

from telethon import TelegramClient
from telethon.errors.rpcerrorlist import FloodWaitError
from telethon.tl.types import Message

# ============================= CONFIG ====================================

PROJECT_DIR = Path("/Users/tomadshead/telegramdaily")
STATE_FILE = PROJECT_DIR / "channel_state.json"
OUTPUT_CSV = PROJECT_DIR / "filtered_translated_macro.csv"

# Telegram API credentials from environment (.email_env)
API_ID = int(os.environ.get("TG_API_ID", "0"))
API_HASH = os.environ.get("TG_API_HASH", "")

# Local session file (Telethon SQLite)
SESSION_PATH = PROJECT_DIR / "telegram_session.session"
SESSION_NAME = str(SESSION_PATH)

# LM Studio endpoint (OpenAI-compatible)
LMSTUDIO_URL = os.environ.get(
    "LMSTUDIO_URL",
    "http://127.0.0.1:1234/v1/chat/completions"
)
LMSTUDIO_MODEL = os.environ.get("LMSTUDIO_MODEL", "gpt-4o-mini")

# Channels to monitor
CHANNELS = [
    "@anthabar",
    "@asiaplus",
    "@asiavcentre",
    "@avesta05",
    "@bagramyan26",
    "@baidildinovoil",
    "@bessimptomno",
    "@caspiankhodja",
    "@CivilGe_Live",
    "@currenttime",
    "@daryo_global",
    "@dashimbayev",
    "@economist_kg",
    "@ekoasia",
    "@ekho_kavkaza",
    "@energytodaygroup",
    "@eurobri",
    "@faridaily24",
    "@FINANCEkaz",
    "@gazetauz",
    "@gaziz1984",
    "@gazmyaso",
    "@insiderUKR",
    "@insider_uz",
    "@jurttyn_balasy",
    "@kaktus_mediakg",
    "@kirillrodionov",
    "@kompromat_za",
    "@kursivm",
    "@kyrgyzforpost",
    "@logistan",
    "@mislinemisli",
    "@newsarmenia",
    "@NGnewsgeorgia",
    "@nlevshitstelegram",
    "@papagaz",
    "@pressauz"
    "@polit_lombard",
    "@proimport",
    "@rbc_news",
    "@russicaRU",
    "@sova_georgia",
    "@tass_agency",
    "@tazabek_official",
    "@tradkz",
    "@turanagency",
    "@UkrPravdaMainNews",
    "@uzmdk",
    "@uznews",
    "@Vesti_Kyrgyzstan"
]

# Countries of interest
APPROVED_COUNTRIES = {
    "Russia",
    "Kazakhstan",
    "Kyrgyz Republic",
    "Uzbekistan",
    "Tajikistan",
    "Turkmenistan",
    "Azerbaijan",
    "Georgia",
    "Armenia",
    "Ukraine",
    "Belarus",
}

ALLOWED_CATEGORIES = {"domestic politics", "foreign relations", "macroeconomics"}

PROMPT_TEMPLATE = """You are a classifier. Output ONLY a single compact JSON object and nothing else.

Decide if the Telegram message is about one of:
- "domestic politics"
- "foreign relations"
- "macroeconomics"

If NOT relevant to these countries, return: {{"skip": true}}
Countries list (primary subject only): Russia, Kazakhstan, Kyrgyz Republic, Uzbekistan, Tajikistan, Turkmenistan, Azerbaijan, Georgia, Armenia, Ukraine, Belarus.

If relevant:
- category: one of the three exactly as above (lowercase)
- country: ONE primary country from the list above (string). If unclear, use "unknown".
- subject: for macroeconomics only, a comma-separated short list from: GDP, inflation, interest rates, exchange rate, reserves, current account, budget, deficit, debt, employment, wages, industrial production, trade, exports, imports, investment, FDI. For non-macroeconomics, use "".
- english: clear English translation or summary of the message, with emojis and non-text removed. Avoid line breaks.

Output JSON schema:
{{
  "skip": false,
  "category": "...",
  "country": "...",
  "subject": "...",
  "english": "..."
}}

Telegram message:
\"\"\"{message}\"\"\""""

# ============================= UNICODE / TEXT CLEANING =====================

def strip_problematic_chars(text: str) -> str:
    """
    Remove emoji, flags, invisible characters, and known 'black box' chars.
    Keeps letters, digits, punctuation, basic symbols.
    """
    if not text:
        return ""

    cleaned: List[str] = []
    for ch in text:
        code = ord(ch)

        # Drop typical black boxes / replacement chars
        if ch == "■" or code == 0xFFFD:
            continue

        # Drop emoji & pictographs (rough ranges)
        if 0x1F000 <= code <= 0x1FAFF:
            continue
        if 0x2600 <= code <= 0x27BF:
            continue
        # Drop regional indicator (flags)
        if 0x1F1E6 <= code <= 0x1F1FF:
            continue

        # Drop various invisible / control chars
        if 0x200B <= code <= 0x200F:
            continue
        if 0x202A <= code <= 0x202E:
            continue
        if 0x2060 <= code <= 0x206F:
            continue

        cleaned.append(ch)

    return "".join(cleaned)


def clean_for_llm(text: str) -> str:
    """
    Clean text before sending to LM Studio.
    We remove emoji/invisible junk, but keep non-ASCII (Cyrillic etc)
    so the model has full context.
    """
    if not text:
        return ""
    text = strip_problematic_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def clean_for_output(text: str) -> str:
    """
    Clean text for CSV/PDF output: remove emoji, flags, invisible chars,
    then restrict to safe ASCII-ish characters so we don't get black squares.
    """
    if not text:
        return ""
    text = strip_problematic_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    safe_chars: List[str] = []
    for ch in text:
        code = ord(ch)
        if ch in "\n\t":
            safe_chars.append(ch)
        elif 32 <= code <= 126:
            safe_chars.append(ch)
        # drop other characters

    out = "".join(safe_chars)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


# ============================= JUNK FILTERS ================================

def is_useless_price_update(text: str) -> bool:
    """
    Detect and remove pure price / ticker updates (FX, commodities, etc.)
    across ALL countries. We do NOT want these.
    """
    if not text:
        return False

    t = text.lower()

    # Strong patterns for FX / tickers
    patterns = [
        r"\$\s*1\s*=\s*\w+",    # $1 = ...
        r"€\s*1\s*=\s*\w+",
        r"£\s*1\s*=\s*\w+",
        r"₽\s*1\s*=\s*\w+",
        r"\busd\b.*=\s*\w+",
        r"\beur\b.*=\s*\w+",
        r"\brub\b.*=\s*\w+",
        r"\bkzt\b.*=\s*\w+",
        r"\buzs\b.*=\s*\w+",
        r"\bbyr\b.*=\s*\w+",
        r"\buah\b.*=\s*\w+",
        r"foreign.currency.exchange.rates",
        r"обменн[ао]й курс",       # Russian "exchange rate"
        r"курс.*валют",            # "currency rate"
    ]

    for pat in patterns:
        if re.search(pat, t):
            return True

    # Heuristic: message is mostly numbers/symbols and mentions a currency
    currency_tokens = ["usd", "eur", "rub", "kzt", "uzs", "byn", "uah", "cny", "gbp"]
    has_currency = any(tok in t for tok in currency_tokens)

    digits = sum(c.isdigit() for c in t)
    letters = sum(c.isalpha() for c in t)

    if has_currency and digits > 30 and digits > letters:
        return True

    return False


def is_mostly_numeric_junk(text: str) -> bool:
    """
    Filter out messages that are overwhelmingly numeric/symbolic and
    unlikely to be informative even if not strictly FX.
    """
    if not text:
        return True

    t = text.replace(" ", "")
    digits = sum(c.isdigit() for c in t)
    letters = sum(c.isalpha() for c in t)

    # Very short texts with hardly any letters and lots of digits: drop
    if letters < 20 and digits > letters:
        return True

    return False


# ============================= JSON UTIL ================================

def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract the first {...} block from a string. Useful when the LLM
    wraps JSON in extra text or fences.
    """
    if not text:
        return None
    text = text.strip()
    # remove code fences if present
    text = re.sub(r"^```(?:json)?", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


# ============================= STATE HANDLING ==============================

def load_state() -> Dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(state: Dict[str, Any]) -> None:
    tmp = STATE_FILE.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    tmp.replace(STATE_FILE)


# ============================= LM STUDIO CALL ==============================

async def classify_message(session: aiohttp.ClientSession, text: str) -> Dict[str, Any]:
    """
    Send a single message to LM Studio, return a dict with:
    - skip (bool)
    - category
    - country
    - subject
    - english (sanitised for output)
    On any error, we fall back to a very safe default.
    """
    cleaned_for_llm = clean_for_llm(text)

    payload = {
        "model": LMSTUDIO_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict JSON-only classifier and translator.",
            },
            {
                "role": "user",
                "content": PROMPT_TEMPLATE.format(message=cleaned_for_llm),
            },
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }

    try:
        async with session.post(LMSTUDIO_URL, json=payload, timeout=60) as resp:
            resp.raise_for_status()
            data = await resp.json()
    except Exception as e:
        english = clean_for_output(cleaned_for_llm)
        return {
            "skip": False,
            "category": "domestic politics",
            "country": "unknown",
            "subject": "",
            "english": english,
            "_error": f"lmstudio_error: {e}",
        }

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        english = clean_for_output(cleaned_for_llm)
        return {
            "skip": False,
            "category": "domestic politics",
            "country": "unknown",
            "subject": "",
            "english": english,
            "_error": f"no_content: {e}",
        }

    content = strip_problematic_chars(content)
    json_text = extract_json_from_text(content)
    if not json_text:
        english = clean_for_output(cleaned_for_llm)
        return {
            "skip": False,
            "category": "domestic politics",
            "country": "unknown",
            "subject": "",
            "english": english,
            "_error": "json_not_found",
        }

    try:
        result = json.loads(json_text)
    except Exception as e:
        english = clean_for_output(cleaned_for_llm)
        return {
            "skip": False,
            "category": "domestic politics",
            "country": "unknown",
            "subject": "",
            "english": english,
            "_error": f"json_parse_error: {e}",
        }

    # Normalise fields
    skip = bool(result.get("skip", False))
    category = str(result.get("category", "")).strip().lower()
    country = str(result.get("country", "")).strip()
    subject = str(result.get("subject", "")).strip()
    english_raw = str(result.get("english", "")).strip()

    if category not in ALLOWED_CATEGORIES and not skip:
        if "macro" in category:
            category = "macroeconomics"
        elif "foreign" in category:
            category = "foreign relations"
        elif "domestic" in category or "politic" in category:
            category = "domestic politics"
        else:
            category = "domestic politics"

    # If model hallucinated a country, keep it but normalise unknowns
    if country not in APPROVED_COUNTRIES and country.lower() != "unknown":
        country = "unknown"

    english = clean_for_output(english_raw or cleaned_for_llm)

    return {
        "skip": skip,
        "category": category,
        "country": country,
        "subject": subject,
        "english": english,
        "_error": None,
    }


# ============================= MESSAGE PROCESSING ==========================

async def process_single_message(
    channel: str,
    msg: Message,
    session: aiohttp.ClientSession,
) -> Optional[Dict[str, Any]]:
    """
    Apply junk filters, classify+translate via LM Studio,
    and produce one CSV row dict or None.
    """
    if not msg.message:
        return None

    text_raw = msg.message

    # Drop all price / ticker updates globally
    if is_useless_price_update(text_raw):
        return None

    # Drop mostly numeric junk
    if is_mostly_numeric_junk(text_raw):
        return None

    # Clean for display (original)
    text_for_output = clean_for_output(text_raw)
    if not text_for_output:
        return None

    # Classify via LM Studio
    result = await classify_message(session, text_raw)
    if result.get("skip"):
        return None

    category = result["category"]
    country = result["country"]
    subject = result["subject"]
    english = result["english"]

    timestamp = msg.date.astimezone(timezone.utc).replace(tzinfo=None)

    row = {
        "datetime": timestamp.isoformat(sep=" "),
        "channel": channel,
        "message_id": msg.id,
        "url": f"https://t.me/{channel.lstrip('@')}/{msg.id}",
        "category": category,
        "country": country,
        "subject": subject,
        "original": text_for_output,
        "english": english,
    }
    return row


# ============================= TELEGRAM FETCH ==============================

async def fetch_channel_messages(
    client: TelegramClient,
    channel: str,
    state: Dict[str, Any],
    session: aiohttp.ClientSession,
    rows: List[Dict[str, Any]],
) -> None:
    """
    Fetch messages for a single channel.
    - On first run (no last_id): fetch ONLY last N messages.
    - On subsequent runs: fetch incrementally (min_id = last_id).
    """
    # --- determine last_id / migrate old format ---
    last_id_entry = state.get(channel)
    if isinstance(last_id_entry, dict):
        last_id = int(last_id_entry.get("last_id", 0))
    elif isinstance(last_id_entry, int):
        last_id = last_id_entry
        state[channel] = {"last_id": last_id}
    else:
        last_id = 0  # first run, no state

    print(f"🔎 Fetching {channel} starting from id > {last_id} ...")

    # --- first run: limit to last N messages ---
    if last_id == 0:
        LIMIT = 200  # adjust if you want more/less history
        print(f"   (First run: fetching last {LIMIT} messages only)")
        try:
            async for msg in client.iter_messages(channel, limit=LIMIT):
                row = await process_single_message(channel, msg, session)
                if row:
                    rows.append(row)
                state[channel] = {"last_id": msg.id}
            print(f"💾 Updated last_id for {channel} → {state[channel]['last_id']}")
            return
        except Exception as e:
            print(f"❌ Error during first-run fetch for {channel}: {e}")
            return

    # --- subsequent runs: incremental fetch using min_id ---
    try:
        async for msg in client.iter_messages(
            channel,
            min_id=last_id,
            reverse=True,
        ):
            row = await process_single_message(channel, msg, session)
            if row:
                rows.append(row)
            state[channel] = {"last_id": msg.id}
            print(f"✅ {channel}: processed {msg.id}")

        if channel in state:
            print(f"💾 Updated last_id for {channel} → {state[channel]['last_id']}")
    except FloodWaitError as e:
        print(f"⚠️ FloodWaitError for {channel}: wait {e.seconds} seconds")
    except Exception as e:
        print(f"❌ Error while fetching {channel}: {e}")


# ============================= CSV WRITING ================================

def write_csv(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("ℹ️ No rows to write; skipping CSV.")
        return

    fieldnames = [
        "datetime",
        "channel",
        "message_id",
        "url",
        "category",
        "country",
        "subject",
        "original",
        "english",
    ]

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"💾 CSV written to {OUTPUT_CSV}")


# ============================= MAIN =======================================

async def main():
    if API_ID == 0 or not API_HASH:
        print("ERROR: TG_API_ID / TG_API_HASH not set in environment.")
        sys.exit(1)

    state = load_state()
    rows: List[Dict[str, Any]] = []

    async with TelegramClient(SESSION_NAME, API_ID, API_HASH) as client:
        async with aiohttp.ClientSession() as session:
            for channel in CHANNELS:
                await fetch_channel_messages(client, channel, state, session, rows)

    write_csv(rows)
    save_state(state)
    print("💾 State saved to", STATE_FILE)


if __name__ == "__main__":
    asyncio.run(main())