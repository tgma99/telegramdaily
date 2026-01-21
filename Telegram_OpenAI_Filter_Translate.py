#!/usr/bin/env python3
"""
Telegram_OpenAI_Filter_Translate.py
==================================
Fetch Telegram messages from channels, then classify + translate via OpenAI,
and output a filtered CSV.

Output columns:
datetime,channel,message_id,url,category,country,subject,original,english
"""

import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from telethon import TelegramClient
from telethon.tl.types import Message
from openai import OpenAI


# ---------------------------- CONFIG ---------------------------------

APPROVED_COUNTRIES = [
    "Russia", "Kazakhstan", "Kyrgyz Republic", "Uzbekistan", "Tajikistan",
    "Turkmenistan", "Azerbaijan", "Georgia", "Armenia", "Ukraine", "Belarus"
]

MACRO_SUBJECTS = [
    "GDP", "inflation", "interest rates", "exchange rate", "reserves",
    "current account", "budget", "deficit", "debt", "employment", "wages",
    "industrial production", "trade", "exports", "imports", "investment", "FDI"
]

CATEGORIES = ["domestic politics", "foreign relations", "macroeconomics"]

URL_RX = re.compile(r"https?://\S+|t\.me/\S+", re.I)
WS_RX = re.compile(r"\s+")


@dataclass
class Secrets:
    telegram_api_id: int
    telegram_api_hash: str
    telegram_session: str
    openai_api_key: str
    openai_model: str


def load_secrets(secrets_path: Path) -> Secrets:
    data = json.loads(secrets_path.read_text(encoding="utf-8"))

    def pick(*keys, default=None):
        """Return the first non-empty value among keys, supporting dotted paths."""
        for k in keys:
            cur: Any = data
            ok = True
            for part in k.split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    ok = False
                    break
            if ok and cur not in (None, "", []):
                return cur
        return default

    tg_api_id = pick("TELEGRAM_API_ID", "telegram_api_id", "api_id", "telegram.api_id")
    tg_api_hash = pick("TELEGRAM_API_HASH", "telegram_api_hash", "api_hash", "telegram.api_hash")
    tg_session = pick("TELEGRAM_SESSION", "telegram_session", "session", "telegram.session", default="telegramdaily")

    oa_key = pick("OPENAI_API_KEY", "openai_api_key", "OPENAI_KEY", "openai_key", "openai.api_key")
    oa_model = pick("OPENAI_MODEL", "openai_model", "openai.model", default="gpt-4o-mini")

    missing = []
    if not tg_api_id:
        missing.append("TELEGRAM_API_ID (or api_id / telegram.api_id)")
    if not tg_api_hash:
        missing.append("TELEGRAM_API_HASH (or api_hash / telegram.api_hash)")
    if not oa_key:
        missing.append("OPENAI_API_KEY (or openai_key / openai.api_key)")

    if missing:
        raise RuntimeError("Missing required keys in secrets.json: " + ", ".join(missing))

    return Secrets(
        telegram_api_id=int(tg_api_id),
        telegram_api_hash=str(tg_api_hash),
        telegram_session=str(tg_session),
        openai_api_key=str(oa_key),
        openai_model=str(oa_model),
    )


def normalize_text(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\u200b", "")  # zero-width space
    s = WS_RX.sub(" ", s).strip()
    return s


def build_prompt(message_text: str) -> str:
    return f"""
You are a classifier. Output ONLY a single compact JSON object and nothing else.

Decide if the Telegram message is about one of:
- "domestic politics"
- "foreign relations"
- "macroeconomics"

If NOT relevant to these countries, return: {{"skip": true}}
Countries list (primary subject only): {", ".join(APPROVED_COUNTRIES)}.

If relevant:
- category: one of the three exactly as above (lowercase)
- country: ONE primary country from the list above (string). If unclear, use "unknown".
- subject: for macroeconomics only, a comma-separated short list from: {", ".join(MACRO_SUBJECTS)}.
  For non-macroeconomics, use "".
- english: clear English translation with emojis/non-text removed.

Output JSON schema:
{{
  "skip": false,
  "category": "...",
  "country": "...",
  "subject": "...",
  "english": "..."
}}

Telegram message:
{message_text}
""".strip()


def _normalize_category(cat: str) -> str:
    cat = (cat or "").replace("_", " ").strip().lower()
    if cat in ("foreign policy", "international relations", "external relations"):
        return "foreign relations"
    if cat in ("domestic policy", "internal politics", "internal policy"):
        return "domestic politics"
    if cat in ("macro", "macro economics", "macro-economic", "macroeconomic"):
        return "macroeconomics"
    return cat


def openai_classify_translate(
    client: OpenAI,
    model: str,
    message_text: str,
    max_retries: int = 5,
) -> Dict[str, Any]:
    prompt = build_prompt(message_text)

    last_err: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "Return ONLY a valid JSON object matching the schema."},
                    {"role": "user", "content": prompt},
                ],
            )
            content = resp.choices[0].message.content or "{}"
            data = json.loads(content)

            if data.get("skip") is True:
                return {"skip": True}

            category = _normalize_category(str(data.get("category", "")))
            country = str(data.get("country", "unknown")).strip()
            subject = str(data.get("subject", "")).strip()
            english = normalize_text(str(data.get("english", "")).strip())

            if category not in CATEGORIES:
                return {"skip": True}

            if country not in APPROVED_COUNTRIES and country != "unknown":
                country = "unknown"

            return {
                "skip": False,
                "category": category,
                "country": country,
                "subject": subject,
                "english": english,
            }

        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))

    # don't crash the whole run; mark as skipped
    return {"skip": True, "error": str(last_err)}


def message_url(channel_username: str, msg: Message) -> str:
    return f"https://t.me/{channel_username.lstrip('@')}/{msg.id}"


async def fetch_recent_messages(
    client: TelegramClient,
    channel: str,
    since_dt: datetime,
    limit: int = 500,
) -> List[Message]:
    out: List[Message] = []

    if since_dt.tzinfo is None:
        since_dt = since_dt.replace(tzinfo=timezone.utc)

    async for msg in client.iter_messages(channel, limit=limit):
        if not msg.date:
            continue

        msg_dt = msg.date
        if msg_dt.tzinfo is None:
            msg_dt = msg_dt.replace(tzinfo=timezone.utc)

        if msg_dt < since_dt:
            break

        if not getattr(msg, "message", None):
            continue

        out.append(msg)

    return out


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def main():
    base_dir = Path(__file__).resolve().parent
    secrets_path = base_dir / "config" / "secrets.json"

    out_dir = base_dir / "data" / "filtered"
    ensure_dir(out_dir)

    hours = 24
    if len(sys.argv) >= 2:
        hours = int(sys.argv[1])

    secrets = load_secrets(secrets_path)

    CHANNELS = [
        "@anthabar",
        "@asiaplus",
        "@asiavcentre",
        "@avesta05",
        "@bagramyan26",
        "@baidildinovoil",
        "@bessimptomno",
        "@blackponed",
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
        "@pressauz",
        "@polit_lombard",
        "@proimport",
        "@pul_1",
        "@rbc_news",
        "@russicaRU",
        "@sova_georgia",
        "@selhoz_agroprom",
        "@tass_agency",
        "@tazabek_official",
        "@tradkz",
        "@turanagency",
        "@UkrPravdaMainNews",
        "@uznews",
        "@Vesti_Kyrgyzstan",
    ]

    now_utc = datetime.now(timezone.utc)
    since_dt = now_utc - timedelta(hours=hours)

    oa_client = OpenAI(api_key=secrets.openai_api_key)

    tg_client = TelegramClient(
        secrets.telegram_session,
        secrets.telegram_api_id,
        secrets.telegram_api_hash,
    )

    rows: List[Dict[str, Any]] = []
    stats = {
        "channels": 0,
        "fetched_msgs": 0,
        "text_msgs": 0,
        "openai_calls": 0,
        "kept": 0,
        "skipped": 0,
        "errors": 0,
    }

    async def run():
        await tg_client.start()

        for ch in CHANNELS:
            stats["channels"] += 1
            print(f"Fetching from {ch}...")
            try:
                msgs = await fetch_recent_messages(tg_client, ch, since_dt=since_dt, limit=500)
            except Exception as e:
                stats["errors"] += 1
                print(f"  ⚠️ Failed to fetch {ch}: {e}")
                continue

            print(f"  -> fetched {len(msgs)} messages since {since_dt.isoformat()}")
            stats["fetched_msgs"] += len(msgs)

            for msg in reversed(msgs):
                original = normalize_text(getattr(msg, "message", "") or "")
                original = original.replace("\x00", "").strip()
                if not original:
                    continue

                stats["text_msgs"] += 1

                # keep this ON for cost control; turn OFF if you want more coverage
                if len(original) < 40 and URL_RX.search(original):
                    stats["skipped"] += 1
                    continue

                try:
                    stats["openai_calls"] += 1
                    result = openai_classify_translate(
                        client=oa_client,
                        model=secrets.openai_model,
                        message_text=original,
                    )
                except Exception as e:
                    stats["errors"] += 1
                    print(f"  ⚠️ OpenAI failed on {ch}/{msg.id}: {e}")
                    print(f"     snippet: {original[:200]}")
                    continue

                if result.get("skip") is True:
                    stats["skipped"] += 1
                    if stats["skipped"] <= 5:
                        print(f"  (skip) {ch}/{msg.id} -> {result}")
                        print(f"         text: {original[:200]}")
                    continue

                rows.append({
                    "datetime": msg.date.isoformat(),
                    "channel": ch,
                    "message_id": msg.id,
                    "url": message_url(ch, msg),
                    "category": result.get("category", ""),
                    "country": result.get("country", "unknown"),
                    "subject": result.get("subject", ""),
                    "original": original,
                    "english": result.get("english", ""),
                })
                stats["kept"] += 1

                if stats["kept"] <= 3:
                    print(f"  (keep) {ch}/{msg.id} -> {result.get('country')} | {result.get('category')}")

        await tg_client.disconnect()

    asyncio.run(run())

    print("STATS:", stats)

    df = pd.DataFrame(rows, columns=[
        "datetime", "channel", "message_id", "url",
        "category", "country", "subject", "original", "english"
    ])

    if not df.empty:
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        df = df.sort_values(["datetime", "country", "channel"], ascending=[True, True, True])

    out_file = out_dir / f"filtered_translated_{now_utc.strftime('%Y-%m-%d')}.csv"
    df.to_csv(out_file, index=False, encoding="utf-8")
    print(f"✅ Saved {len(df)} rows → {out_file}")


if __name__ == "__main__":
    main()