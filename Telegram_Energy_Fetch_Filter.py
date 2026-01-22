#!/usr/bin/env python3
"""
Telegram_Energy_Fetch_Filter.py

Fetch Telegram messages from a dedicated ENERGY channel list and keep only
energy-related posts (oil/gas/renewables/electricity), using:

1) Fast keyword filter (default)
2) Optional LLM judge (OpenAI-compatible: OpenAI OR LM Studio) for borderline cases

Outputs a CSV:
  data/energy/filtered_energy_YYYY-MM-DD.csv

It prints a line like:
  ✅ Saved N rows → /.../filtered_energy_YYYY-MM-DD.csv

Usage (keyword-only, last 7 days):
  python Telegram_Energy_Fetch_Filter.py --hours 168

Usage (add LM Studio judge for borderline):
  python Telegram_Energy_Fetch_Filter.py --hours 168 \
    --use_llm \
    --base_url http://127.0.0.1:1234/v1 \
    --model llama-3.1-8b-instruct \
    --api_key lm-studio

Channels file (one per line):
  config/channels_energy.txt

Keywords file (one per line):
  config/energy_keywords.txt
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Telethon
from telethon import TelegramClient
from telethon.errors import FloodWaitError

# Optional: OpenAI SDK (v1), for OpenAI-compatible endpoints (OpenAI or LM Studio)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore


# -----------------------------
# Defaults
# -----------------------------

DEFAULT_CHANNELS_FILE = "config/channels_energy.txt"
DEFAULT_KEYWORDS_FILE = "config/energy_keywords.txt"
DEFAULT_OUT_DIR = "data/energy"
DEFAULT_STATE_FILE = "config/channel_state_energy.json"

DEFAULT_KEYWORDS = [
    # broad
    "energy", "power", "electricity", "grid", "generation", "transmission", "distribution",
    # oil
    "oil", "brent", "wti", "opec", "opec+", "refinery", "refining", "diesel", "gasoline", "petrol", "fuel",
    # gas
    "gas", "lng", "pipeline", "gazprom", "storage", "gas hub",
    # renewables
    "renewable", "solar", "wind", "hydro", "geothermal", "biomass",
    # nuclear
    "nuclear", "reactor", "uranium", "enrichment",
    # coal / carbon
    "coal", "emissions", "carbon", "co2", "cap-and-trade", "ets",
    # hydrogen / storage / EV infra
    "hydrogen", "electrolyser", "electrolyzer", "battery", "storage", "charger", "charging",
    # RU/CIS common words (optional; add more in file)
    "нефть", "газ", "уголь", "электроэнерг", "энергетик", "гэс", "аэс", "вдe", "ветро", "солнеч",
]

LLM_SYSTEM = (
    "You are a strict classifier for ENERGY news in Telegram posts.\n"
    "Decide if the message is about energy: oil, gas, LNG, refining, pipelines, "
    "electricity/power grids, renewables, nuclear, coal, emissions/carbon policy, "
    "energy infrastructure, energy markets/prices.\n\n"
    "Return ONLY valid JSON:\n"
    '{ "is_energy": true/false, "sector": "oil|gas|power|renewables|nuclear|coal|carbon|other",'
    '  "confidence": 0.0-1.0, "reason": "short <= 20 words" }\n'
    "If unsure, set is_energy=false."
)


# -----------------------------
# Helpers
# -----------------------------

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def utcnow() -> datetime:
    return datetime.now(timezone.utc)

def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def load_lines(path: Path) -> List[str]:
    if not path.exists():
        return []
    out: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.append(line)
    return out

def build_kw_regex(keywords: List[str]) -> re.Pattern:
    """
    Build a robust regex that matches any keyword as a substring-ish token.
    We intentionally keep this forgiving for Telegram text.
    """
    kws = [k.strip() for k in keywords if k.strip()]
    # Escape, but allow '+' and '.' literally too
    parts = [re.escape(k) for k in kws]
    if not parts:
        parts = [re.escape(k) for k in DEFAULT_KEYWORDS]
    pat = r"(" + "|".join(parts) + r")"
    return re.compile(pat, flags=re.IGNORECASE)

def msg_url(channel: str, msg_id: int) -> str:
    ch = channel.lstrip("@")
    return f"https://t.me/{ch}/{msg_id}"

def pick_first(d: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default

def load_secrets(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def get_tg_creds(secrets: Dict[str, Any]) -> Tuple[int, str]:
    api_id = pick_first(secrets, ["telegram_api_id", "tg_api_id", "api_id"], None)
    api_hash = pick_first(secrets, ["telegram_api_hash", "tg_api_hash", "api_hash"], None)
    if api_id is None or api_hash is None:
        # allow env var fallback
        api_id = api_id or os.environ.get("TELEGRAM_API_ID")
        api_hash = api_hash or os.environ.get("TELEGRAM_API_HASH")
    if api_id is None or api_hash is None:
        raise SystemExit(
            "Missing Telegram credentials. Add telegram_api_id/telegram_api_hash to secrets.json "
            "or set TELEGRAM_API_ID / TELEGRAM_API_HASH env vars."
        )
    return int(api_id), str(api_hash)

def make_openai_client(base_url: Optional[str], api_key: Optional[str]) -> Any:
    if OpenAI is None:
        raise SystemExit("openai SDK not installed. Install with: pip install openai")
    # OpenAI SDK accepts base_url and api_key (LM Studio is OpenAI-compatible)
    return OpenAI(base_url=base_url, api_key=api_key)

def keyword_match(text: str, rx: re.Pattern) -> Optional[str]:
    if not text:
        return None
    m = rx.search(text)
    return m.group(0) if m else None

def truncate_for_llm(text: str, max_chars: int = 1200) -> str:
    t = norm(text)
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + "…"


# -----------------------------
# Data model
# -----------------------------

@dataclass
class EnergyRow:
    datetime: str
    channel: str
    message_id: int
    url: str
    text: str
    keep_reason: str
    keyword_hit: str
    llm_is_energy: str
    llm_sector: str
    llm_confidence: float
    llm_reason: str


# -----------------------------
# LLM Judge
# -----------------------------

async def llm_judge_energy(
    client: Any,
    model: str,
    text: str,
    timeout_s: int = 60,
) -> Tuple[bool, str, float, str]:
    """
    Returns (is_energy, sector, confidence, reason)
    """
    prompt = (
        "Classify this Telegram post.\n"
        "POST:\n"
        f"{truncate_for_llm(text)}\n"
    )
    # NOTE: do NOT pass response_format; LM Studio rejects json_object.
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=120,
        messages=[
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        timeout=timeout_s,
    )
    content = (resp.choices[0].message.content or "").strip()

    # Extract JSON object if wrapped
    m = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    jtxt = m.group(0) if m else ""

    if not jtxt:
        return (False, "other", 0.0, "Non-JSON output")

    try:
        obj = json.loads(jtxt)
    except Exception:
        return (False, "other", 0.0, "Parse failure")

    is_energy = bool(obj.get("is_energy", False))
    sector = str(obj.get("sector", "other") or "other").strip().lower()
    conf = float(obj.get("confidence", 0.0) or 0.0)
    reason = norm(str(obj.get("reason", "")))[:180] or "No reason"

    # normalize
    if sector not in {"oil", "gas", "power", "renewables", "nuclear", "coal", "carbon", "other"}:
        sector = "other"
    conf = max(0.0, min(1.0, conf))

    return (is_energy, sector, conf, reason)


# -----------------------------
# Main fetch loop
# -----------------------------

async def fetch_energy(
    base_dir: Path,
    channels: List[str],
    since_dt: datetime,
    kw_rx: re.Pattern,
    state_path: Path,
    use_llm: bool,
    llm_client: Any,
    llm_model: str,
    max_llm_calls: int,
    min_llm_confidence: float,
) -> List[EnergyRow]:
    secrets = load_json(base_dir / "config" / "secrets.json")
    api_id, api_hash = get_tg_creds(secrets)

    session_name = str(base_dir / "config" / "telethon_energy.session")
    client = TelegramClient(session_name, api_id, api_hash)

    state = load_json(state_path)
    state.setdefault("channels", {})

    kept: List[EnergyRow] = []
    llm_calls_used = 0

    await client.start()

    for ch in channels:
        ch = ch.strip()
        if not ch:
            continue
        if not ch.startswith("@"):
            ch = "@" + ch

        ch_state = state["channels"].get(ch, {})
        last_id = int(ch_state.get("last_id", 0) or 0)

        fetched = 0
        kept_here = 0

        print(f"Fetching from {ch}...")

        try:
            entity = await client.get_entity(ch)
        except Exception as e:
            print(f"  !! Failed to resolve {ch}: {e}")
            continue

        # Iterate newest->oldest; stop when older than since_dt
        try:
            async for msg in client.iter_messages(entity, limit=5000):
                if not msg:
                    continue
                if not getattr(msg, "date", None):
                    continue
                if msg.date.replace(tzinfo=timezone.utc) < since_dt:
                    break
                if msg.id and msg.id <= last_id:
                    # already processed
                    continue

                fetched += 1
                text = norm(getattr(msg, "message", "") or "")
                if not text:
                    continue

                kw = keyword_match(text, kw_rx)
                keep = False
                keep_reason = ""
                llm_is_energy = ""
                llm_sector = ""
                llm_conf = 0.0
                llm_reason = ""

                if kw:
                    keep = True
                    keep_reason = "keyword"
                elif use_llm and llm_calls_used < max_llm_calls:
                    # LLM only for borderline (no keyword hit)
                    try:
                        is_e, sector, conf, reason = await llm_judge_energy(
                            llm_client, llm_model, text
                        )
                        llm_calls_used += 1
                        llm_is_energy = "true" if is_e else "false"
                        llm_sector = sector
                        llm_conf = conf
                        llm_reason = reason
                        if is_e and conf >= min_llm_confidence:
                            keep = True
                            keep_reason = "llm"
                    except Exception as e:
                        # conservative default: do not keep on LLM failure
                        llm_is_energy = "error"
                        llm_reason = f"LLM error: {e}"

                if keep:
                    kept_here += 1
                    kept.append(
                        EnergyRow(
                            datetime=msg.date.astimezone(timezone.utc).isoformat(),
                            channel=ch,
                            message_id=int(msg.id),
                            url=msg_url(ch, int(msg.id)),
                            text=text,
                            keep_reason=keep_reason,
                            keyword_hit=kw or "",
                            llm_is_energy=llm_is_energy,
                            llm_sector=llm_sector,
                            llm_confidence=float(llm_conf),
                            llm_reason=llm_reason,
                        )
                    )

                # update per-channel last_id as we go (monotonic increasing with newest->oldest? not strictly)
                # We'll set it at end using max seen id in this run.
            # end iter
        except FloodWaitError as e:
            print(f"  !! FloodWaitError for {ch}: sleep {e.seconds}s")
            await asyncio.sleep(min(e.seconds, 30))
        except Exception as e:
            print(f"  !! Error fetching {ch}: {e}")

        # Update state: set last_id to max message_id we kept/fetched this run (best effort)
        max_id_seen = last_id
        for r in kept:
            if r.channel == ch:
                max_id_seen = max(max_id_seen, r.message_id)
        state["channels"][ch] = {"last_id": max_id_seen, "updated_utc": utcnow().isoformat()}

        print(f"  -> fetched ~{fetched} msgs since {since_dt.isoformat()} | kept {kept_here}")

    # Save state
    save_json(state_path, state)

    await client.disconnect()
    return kept


# -----------------------------
# CSV Output
# -----------------------------

def write_csv(path: Path, rows: List[EnergyRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else [
        "datetime","channel","message_id","url","text","keep_reason","keyword_hit",
        "llm_is_energy","llm_sector","llm_confidence","llm_reason"
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


# -----------------------------
# CLI
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=int, default=168, help="Lookback window in hours (default 168 = 7 days)")
    ap.add_argument("--channels_file", type=str, default=DEFAULT_CHANNELS_FILE, help="Path to channels list (one per line)")
    ap.add_argument("--keywords_file", type=str, default=DEFAULT_KEYWORDS_FILE, help="Path to keywords list (one per line)")
    ap.add_argument("--out_dir", type=str, default=DEFAULT_OUT_DIR, help="Output directory for CSV")
    ap.add_argument("--state_file", type=str, default=DEFAULT_STATE_FILE, help="Per-channel state JSON path")

    # LLM (optional)
    ap.add_argument("--use_llm", action="store_true", help="Use LLM to judge borderline posts (no keyword hit)")
    ap.add_argument("--base_url", type=str, default=None, help="OpenAI-compatible base_url (LM Studio: http://127.0.0.1:1234/v1)")
    ap.add_argument("--model", type=str, default=None, help="Model name (LM Studio: e.g. llama-3.1-8b-instruct)")
    ap.add_argument("--api_key", type=str, default=None, help="API key (LM Studio: any string like 'lm-studio')")
    ap.add_argument("--secrets", type=str, default="config/secrets.json", help="Secrets JSON (optional; used for defaults)")
    ap.add_argument("--max_llm_calls", type=int, default=600, help="Max LLM calls per run (default 600)")
    ap.add_argument("--min_llm_confidence", type=float, default=0.55, help="Min confidence to keep on LLM match (default 0.55)")

    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    out_dir = (base_dir / args.out_dir).resolve()
    state_path = (base_dir / args.state_file).resolve()

    channels_path = (base_dir / args.channels_file).resolve()
    keywords_path = (base_dir / args.keywords_file).resolve()

    channels = load_lines(channels_path)
    if not channels:
        raise SystemExit(f"No channels found in {channels_path}. Add @channel per line.")

    keywords = load_lines(keywords_path)
    if not keywords:
        keywords = DEFAULT_KEYWORDS

    kw_rx = build_kw_regex(keywords)

    since_dt = utcnow() - timedelta(hours=int(args.hours))

    use_llm = bool(args.use_llm)
    llm_client = None
    llm_model = args.model or ""

    if use_llm:
        secrets = load_secrets(str((base_dir / args.secrets).resolve()))
        base_url = args.base_url or secrets.get("base_url") or secrets.get("lmstudio_base_url")
        api_key = args.api_key or secrets.get("api_key") or secrets.get("lmstudio_api_key") or secrets.get("openai_api_key") or "lm-studio"
        llm_model = llm_model or secrets.get("model") or secrets.get("lmstudio_model") or "llama-3.1-8b-instruct"
        if not base_url:
            # Allow OpenAI default if they want
            base_url = secrets.get("openai_base_url")  # usually None
        llm_client = make_openai_client(base_url=base_url, api_key=api_key)

    print(f"[energy] hours={args.hours} since={since_dt.isoformat()}")
    print(f"[energy] channels_file={channels_path}")
    print(f"[energy] keywords={len(keywords)} use_llm={use_llm}")
    if use_llm:
        print(f"[energy] base_url={args.base_url} model={llm_model} max_llm_calls={args.max_llm_calls}")

    rows = asyncio.run(
        fetch_energy(
            base_dir=base_dir,
            channels=channels,
            since_dt=since_dt,
            kw_rx=kw_rx,
            state_path=state_path,
            use_llm=use_llm,
            llm_client=llm_client,
            llm_model=llm_model,
            max_llm_calls=int(args.max_llm_calls),
            min_llm_confidence=float(args.min_llm_confidence),
        )
    )

    # Sort oldest->newest for nicer downstream reading
    rows.sort(key=lambda r: r.datetime)

    out_path = out_dir / f"filtered_energy_{utcnow().date().isoformat()}.csv"
    write_csv(out_path, rows)

    print(f"STATS: {{'channels': {len(channels)}, 'kept': {len(rows)}}}")
    print(f"✅ Saved {len(rows)} rows → {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())