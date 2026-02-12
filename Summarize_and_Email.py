#!/usr/bin/env python3
# Summarize_and_Email.py
# ==========================================================
# Reads a filtered Telegram CSV and produces a daily email (HTML)
# with the brown/grey scheme.
#
# Supports CSV formats:
#  (A) already has: country/category/english/skip/subject/url/datetime/channel
#  (B) has: datetime/channel/message_id/url/raw/lm_json
#      where lm_json contains {"skip":..., "country":..., "category":..., "subject":..., "english":...}
#
# If country is missing/unknown and OpenAI is configured, this script will
# classify + translate + summarise items so the email still groups by country.

import argparse
import ast
import json
import os
import re
import html
import smtplib
from datetime import datetime, timedelta, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple
from typing import List
import pandas as pd


APPROVED_COUNTRIES = [
    "Russia", "Kazakhstan", "Kyrgyz Republic", "Uzbekistan", "Tajikistan",
    "Turkmenistan", "Azerbaijan", "Georgia", "Armenia", "Ukraine", "Belarus",
]
APPROVED_SET = set(APPROVED_COUNTRIES)

WATCH_KEYWORDS = []

# -----------------------------
# HTML template (brown/grey)
# -----------------------------
HTML_STYLE = """\
<style>
  .t-body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 120%; letter-spacing: .1pt; color: #76777B; }
  .t-accent { font-weight: 700; color: #9F7E56; }
  .t-p { margin: 0 0 12px 0; }
  .t-hr { margin: 18px 0 0 0; border-top: 1px solid #E1E1E1; padding-top: 10px; }
  .t-h2 { margin: 16px 0 8px 0; font-size: 12pt; font-weight: 700; color: #2B2B2B; }
  .t-muted { color: #9AA0A6; }
  .t-li { margin: 0 0 10px 0; }
  a { color: #6B6F76; text-decoration: underline; }
</style>
"""

HTML_WRAPPER_START = """\
<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width" />
{style}
</head>
<body>
<table role="presentation" width="100%" cellpadding="0" cellspacing="0" border="0" style="background:#ffffff;">
  <tr>
    <td align="center">
      <table role="presentation" width="680" cellpadding="0" cellspacing="0" border="0" style="width:680px; max-width:680px;">
        <tr>
          <td class="t-body" style="padding:18px 18px 6px 18px; font-family:Arial,sans-serif; font-size:10.5pt; line-height:120%; letter-spacing:.1pt; color:#76777B;">
"""

HTML_WRAPPER_END = """\
          </td>
        </tr>
      </table>
    </td>
  </tr>
</table>
</body>
</html>
"""

# -----------------------------
# Country normalisation
# -----------------------------
COUNTRY_ALIASES = {
    "rf": "Russia",
    "ru": "Russia",
    "russia": "Russia",
    "russian federation": "Russia",
    "kz": "Kazakhstan",
    "kazakhstan": "Kazakhstan",
    "kg": "Kyrgyz Republic",
    "kyrgyzstan": "Kyrgyz Republic",
    "kyrgyz republic": "Kyrgyz Republic",
    "uz": "Uzbekistan",
    "uzbekistan": "Uzbekistan",
    "tj": "Tajikistan",
    "tajikistan": "Tajikistan",
    "tm": "Turkmenistan",
    "turkmenistan": "Turkmenistan",
    "az": "Azerbaijan",
    "azerbaijan": "Azerbaijan",
    "ge": "Georgia",
    "georgia": "Georgia",
    "am": "Armenia",
    "armenia": "Armenia",
    "ua": "Ukraine",
    "ukraine": "Ukraine",
    "by": "Belarus",
    "belarus": "Belarus",
}

def normalize_country(x: Any) -> str:
    if not isinstance(x, str):
        return "unknown"

    s = x.strip()
    if not s:
        return "unknown"

    # Remove punctuation and normalize spaces
    s_clean = re.sub(r"[^\w\s-]", "", s).strip()
    k = s_clean.lower()

    # Direct match
    if s_clean in APPROVED_SET:
        return s_clean

    # Alias lookup (lowercased)
    if k in COUNTRY_ALIASES:
        return COUNTRY_ALIASES[k]

    # Case-insensitive match against approved list
    for c in APPROVED_COUNTRIES:
        if c.lower() == k:
            return c

    return "unknown"

# -----------------------------
# Utilities
# -----------------------------
def load_secrets(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_channel(ch: Any) -> str:
    if not isinstance(ch, str):
        return "@unknown"
    s = ch.strip()
    s = re.sub(r"^@+", "", s)
    return "@" + s if s else "@unknown"


def strip_code_fences(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s2 = s.strip()
    if s2.startswith("```"):
        s2 = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s2)
        s2 = re.sub(r"\s*```$", "", s2).strip()
    return s2


def extract_first_json_object(s: str) -> Optional[str]:
    if not isinstance(s, str):
        return None
    s = strip_code_fences(s)
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        return s[start:end + 1]
    return None


def safe_parse_lm_json(x: Any) -> Dict[str, Any]:
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

    s0 = strip_code_fences(s)
    try:
        return json.loads(s0)
    except Exception:
        pass

    j = extract_first_json_object(s0)
    if j:
        try:
            return json.loads(j)
        except Exception:
            pass

    try:
        v = ast.literal_eval(s0)
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


def extract_csv_date_from_filename(csv_path: str) -> Optional[str]:
    m = re.search(r"filtered_(\d{4}-\d{2}-\d{2})\.csv$", os.path.basename(csv_path))
    return m.group(1) if m else None


def fmt_dt_utc(dt: pd.Timestamp) -> str:
    try:
        dtu = dt.tz_convert("UTC")
    except Exception:
        dtu = dt
    return dtu.strftime("%Y-%m-%d %H:%M")


def build_subject(prefix: str, csv_date: Optional[str], now_utc: datetime) -> str:
    date_str = csv_date if csv_date else now_utc.strftime("%Y-%m-%d")
    return f"{prefix} – {date_str}"


def html_escape(s: Any) -> str:
    if s is None:
        return ""
    s = str(s)
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
         .replace('"', "&quot;")
    )

def strip_emojis(text: Any) -> str:
    if not isinstance(text, str):
        return ""

    # Remove emojis, symbols, dingbats, variation selectors
    emoji_pattern = re.compile(
        "["
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F680-\U0001F6FF"  # transport & map
        "\U0001F700-\U0001F77F"
        "\U0001F780-\U0001F7FF"
        "\U0001F800-\U0001F8FF"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FAFF"
        "\u2600-\u26FF"          # misc symbols (☢ ⚠ ❗)
        "\u2700-\u27BF"          # dingbats
        "\uFE0F"                 # variation selector
        "]+",
        flags=re.UNICODE,
    )

    return emoji_pattern.sub("", text)

def looks_cyrillic(text: str) -> bool:
    if not text:
        return False
    # if any Cyrillic letters present
    return bool(re.search(r"[А-Яа-яЁё]", text))

def llm_dedupe_items(client, model, rows: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate semantically similar items using LLM event keys.
    rows: already small (e.g. top 3–6 per country)
    """
    if rows.empty or client is None:
        return rows

    event_keys = []
    for _, r in rows.iterrows():
        text = (r.get("headline","") + " — " + r.get("why","")).strip()

        prompt = (
            "Create a short stable identifier for the underlying news EVENT.\n"
            "Different wording or outlets describing the SAME event must map to the SAME key.\n"
            "Do NOT include dates unless the date itself is the event.\n"
            "Return JSON only: {\"event_key\": \"...\"}\n\n"
            f"TEXT:\n{text}\n"
        )

        try:
            resp = client.responses.create(
                model=model,
                input=prompt,
                temperature=0.0,
            )
            out = resp.output_text.strip()
            key = json.loads(out).get("event_key","")
        except Exception:
            key = text[:60].lower()

        event_keys.append(key)

    rows = rows.copy()
    rows["_event_key"] = event_keys

    # Keep first occurrence of each event
    rows = rows.drop_duplicates("_event_key", keep="first")

    return rows.drop(columns=["_event_key"], errors="ignore")


# -----------------------------
# OpenAI (optional)
# -----------------------------
def get_openai_client(api_key: str):
    from openai import OpenAI  # type: ignore
    return OpenAI(api_key=api_key)


from typing import Dict
import json
import re

def _clamp_words(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", (s or "")).strip()
    if not s:
        return ""
    words = s.split(" ")
    if len(words) <= n:
        return s
    return " ".join(words[:n]).strip()

def openai_classify_translate_and_summarise(client, model: str, text: str, url: str) -> Dict[str, str]:
    """
    Returns dict with keys: country, headline, why
    - Translates to clean English if needed
    - Removes emojis/decorative symbols
    - Summary ("why") <= 26 words, purely what the post says (no 'why it matters')
    """

    prompt = (
        "You are preparing a daily news digest.\n"
        "Given ONE Telegram post (may be Russian/other language), do ALL of the following:\n"
        f"- Choose ONE primary country from this list exactly:\n{', '.join(APPROVED_COUNTRIES)}\n"
        "- If genuinely unclear, use 'unknown'.\n"
        "- Translate into clear English if needed.\n"
        "- Remove emojis and decorative symbols.\n"
        "- Write a short headline.\n"
        "- Write a ONE-sentence summary of what the Telegram post says (max 26 words). "
        "Do NOT write 'why it matters' or generic implications.\n"
        "Return ONLY a JSON object with keys: country, headline, why.\n\n"
        f"URL: {url}\n"
        f"TEXT:\n{text}\n"
    )

    # Prefer Responses API with JSON schema when available
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=0.2,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "digest_item",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "country": {"type": "string"},
                            "headline": {"type": "string"},
                            "why": {"type": "string"}
                        },
                        "required": ["country", "headline", "why"],
                        "additionalProperties": False
                    }
                }
            },
        )
        out = (resp.output_text or "").strip()
    except Exception:
        # Fallback to chat.completions (less strict)
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        out = (comp.choices[0].message.content or "").strip()

    out = strip_code_fences(out)
    j = extract_first_json_object(out) or out

    try:
        obj = json.loads(j)

        c = normalize_country(obj.get("country", "unknown"))
        h = strip_emojis(str(obj.get("headline", "")).strip())
        w = strip_emojis(str(obj.get("why", "")).strip())

        # Enforce word limits even if the model drifts
        h = _clamp_words(h, 30) or "Update"
        w = _clamp_words(w, 26) or "See source for details."

        if c not in APPROVED_SET:
            c = "unknown"

        return {"country": c, "headline": h, "why": w}

    except Exception:
        return {"country": "unknown", "headline": "Update", "why": "See source for details."}


# =======================
# Text / summarisation helpers
# =======================

def extract_full_sentence(text: str) -> str:
    if not text:
        return ""
    m = re.search(r"^(.+?[.!?])(\s|$)", text.strip())
    return m.group(1) if m else text.strip()


def fallback_summarise(text: str) -> Tuple[str, str]:
    t = strip_emojis(re.sub(r"\s+", " ", (text or "")).strip())
    if not t:
        return "Update", "See source for details."

    # Split into sentences
    parts = re.split(r"(?<=[.!?])\s+", t)

    # Headline: first sentence trimmed to <= 18 words
    headline = parts[0].strip()
  

    # Summary: prefer second sentence; else reuse trimmed first sentence
    if len(parts) >= 2 and parts[1].strip():
        summary = parts[1].strip()
    else:
        summary = parts[0].strip()

    # Keep summary to ~26 words
    s_words = summary.split()
    if len(s_words) > 26:
        summary = " ".join(s_words[:26]).strip()
        # add a period if missing
        if summary and summary[-1] not in ".!?":
            summary += "."

    return headline, summary


# -----------------------------
# Data normalisation
# -----------------------------
def normalise_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # datetime
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    else:
        df["datetime"] = pd.to_datetime(pd.NaT, utc=True)

    # channel/url/raw defaults
    if "channel" not in df.columns:
        df["channel"] = "@unknown"
    df["channel"] = df["channel"].map(normalize_channel)

    if "url" not in df.columns:
        df["url"] = ""
    if "raw" not in df.columns:
        df["raw"] = ""

    # If structured fields missing, try lm_json
    need_from_lm = any(c not in df.columns for c in ["country", "category", "subject", "english", "skip"])
    if need_from_lm:
        if "lm_json" in df.columns:
            parsed = df["lm_json"].map(safe_parse_lm_json)
        else:
            parsed = [{} for _ in range(len(df))]
        df["__lm"] = parsed

        def lm_get(row, key, default=""):
            d = row.get("__lm") or {}
            v = d.get(key, default)
            return v if v is not None else default

        if "country" not in df.columns:
            df["country"] = df.apply(lambda r: lm_get(r, "country", "unknown"), axis=1)
        if "category" not in df.columns:
            df["category"] = df.apply(lambda r: lm_get(r, "category", ""), axis=1)
        if "subject" not in df.columns:
            df["subject"] = df.apply(lambda r: lm_get(r, "subject", ""), axis=1)
        if "english" not in df.columns:
            df["english"] = df.apply(lambda r: lm_get(r, "english", ""), axis=1)
        if "skip" not in df.columns:
            df["skip"] = df.apply(lambda r: bool(lm_get(r, "skip", False)), axis=1)

        df.drop(columns=["__lm"], inplace=True, errors="ignore")

    # types
    df["country"] = df["country"].fillna("unknown").astype(str)
    df["category"] = df["category"].fillna("").astype(str)
    df["subject"] = df["subject"].fillna("").astype(str)
    df["english"] = df["english"].fillna("").astype(str)
    df["skip"] = df["skip"].fillna(False).astype(bool)

    # NORMALISE COUNTRY (key fix)
    df["country"] = df["country"].map(normalize_country)

    return df


# -----------------------------
# Rendering
# -----------------------------
def render_header(subject_line: str) -> str:
    return (
        f'<p class="t-p" style="margin:0 0 12px 0;">'
        f'<span class="t-accent" style="font-weight:700; color:#9F7E56;">'
        f'{html_escape(subject_line)}'
        f'</span>'
        f'</p>'
        f'<p class="t-hr" style="margin:18px 0 0 0; '
        f'border-top:1px solid #E1E1E1; padding-top:10px;"></p>'
    )

def render_footer(source_csv: str) -> str:
    return (
        f'<p class="t-hr" style="margin:24px 0 10px 0; '
        f'border-top:1px solid #E1E1E1; padding-top:10px;"></p>'
        f'<p class="t-p t-muted" style="margin:0; font-size:12px; color:#9AA0A6;">'
        f'Source CSV: {html_escape(os.path.basename(source_csv))}'
        f'</p>'
    )



def llm_event_dedupe_keep_rows(client, model: str, rows: pd.DataFrame, max_keep: int) -> pd.DataFrame:
    """
    Uses OpenAI to dedupe by underlying EVENT, not text similarity.
    Returns a subset of rows to keep (<= max_keep), preferring most informative items.
    """

    if rows is None or rows.empty or client is None:
        return rows.head(max_keep) if rows is not None else rows

    # Sort newest first so "keep the best" still tends to keep fresh items if ties
    rows = rows.sort_values("datetime", ascending=False).reset_index(drop=False)  # keep original index in "index"
    items = []
    for i, r in rows.iterrows():
        headline = (r.get("headline") or "").strip()
        why = (r.get("why") or "").strip()
        url = (r.get("url") or "").strip()
        ch = (r.get("channel") or "").strip()
        dt = r.get("datetime")
        dt_s = ""
        try:
            dt_s = pd.to_datetime(dt, utc=True).strftime("%Y-%m-%d %H:%M")
        except Exception:
            dt_s = str(dt or "")

        # Short payload: headline + one-sentence summary is usually enough
        items.append({
            "i": i,               # local position
            "dt": dt_s,
            "channel": ch,
            "url": url,
            "headline": headline,
            "summary": why,
        })

        prompt = (
            "You deduplicate a daily news email.\n"
            "Goal: remove items that report the SAME UNDERLYING EVENT, even if wording differs, "
            "names are spelled differently (e.g., Assaubayeva/Asaubaeva), or one item adds extra context.\n\n"
            "Rules:\n"
            "- Group items by underlying event.\n"
            "- For each group, pick ONE best item to KEEP (most informative/complete).\n"
            f"- Keep at most {max_keep} items total.\n"
            "Return ONLY JSON, no markdown.\n\n"
            "JSON schema:\n"
            "{\"keep\": [<i>, ...], \"groups\": [[<i>,<i>...], ...]}\n\n"
            "Items:\n"
            f"{json.dumps(items, ensure_ascii=False)}"
        )

    # Call OpenAI
    try:
        resp = client.responses.create(model=model, input=prompt, temperature=0.1)
        out = (resp.output_text or "").strip()
    except Exception:
        comp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
        )
        out = (comp.choices[0].message.content or "").strip()

    # Parse JSON robustly
    out = out.strip()
    # strip accidental code fences
    out = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", out)
    out = re.sub(r"\s*```$", "", out).strip()

    # extract first {...}
    start, end = out.find("{"), out.rfind("}")
    if start >= 0 and end > start:
        out = out[start:end+1]

    try:
        obj = json.loads(out)
        keep_idxs = obj.get("keep", [])
        keep_idxs = [k for k in keep_idxs if isinstance(k, int) and 0 <= k < len(rows)]
        keep_idxs = list(dict.fromkeys(keep_idxs))  # de-dupe, preserve order
        if not keep_idxs:
            return rows.head(max_keep)
        kept = rows.loc[keep_idxs].copy()

        # enforce max_keep
        kept = kept.head(max_keep)

        # restore original dataframe index order by datetime desc (already)
        kept = kept.sort_values("datetime", ascending=False)
        # drop helper column
        kept = kept.drop(columns=["index"], errors="ignore")
        return kept
    except Exception:
        # Safe fallback
        return rows.head(max_keep).drop(columns=["index"], errors="ignore")

def render_keyword_alerts(rows: pd.DataFrame) -> str:
    title = "Keyword Alerts (AgroTerra / АгроТерра / NCH Capital / Chevron)"
    parts = [(
        '<div class="t-h2" style="margin:16px 0 8px 0; font-size:12pt; font-weight:700; color:#2B2B2B;">'
        f'{html_escape(title)}'
        '</div>'
    )]

    if rows is None or rows.empty:
        parts.append('<p class="t-p" style="margin:0 0 12px 0;">No items matched the watched keywords in the last 24 hours.</p>')
        parts.append('<p class="t-hr" style="margin:18px 0 0 0; border-top:1px solid #E1E1E1; padding-top:10px;"></p>')
        return "\n".join(parts)

    # Figure out which URL column exists
    url_col = None
    for c in ["url", "link", "source_url", "message_url"]:
        if c in rows.columns:
            url_col = c
            break

    # Ensure datetime column is parsed if present
    if "datetime" in rows.columns:
        rows = rows.copy()
        rows["datetime"] = pd.to_datetime(rows["datetime"], errors="coerce", utc=True)

    # Sort newest first when possible
    if "datetime" in rows.columns:
        rows = rows.sort_values("datetime", ascending=False)

    for _, r in rows.iterrows():
        dt = r.get("datetime", pd.NaT)
        dt_s = fmt_dt_utc(dt) if pd.notna(dt) else ""

        url = ""
        if url_col:
            url = str(r.get(url_col, "") or "").strip()

        # Optional: show which keyword(s) matched, if your prefilter writes it
        kw_terms = ""
        if "kw_terms" in rows.columns:
            kw_terms = str(r.get("kw_terms", "") or "").strip()
        elif "matched" in rows.columns:
            kw_terms = str(r.get("matched", "") or "").strip()

        kw_terms = strip_emojis(kw_terms).strip()

        # Build line
        left = f'<span class="t-accent" style="font-weight:700; color:#9F7E56;">{html_escape(dt_s)}</span>' if dt_s else ""
        mid = f'<span class="t-muted" style="color:#9AA0A6;"> • {html_escape(kw_terms)}</span>' if kw_terms else ""
        right = f'<a href="{html_escape(url)}">{html_escape(url)}</a>' if url else '<span class="t-muted" style="color:#9AA0A6;">(no link)</span>'

        parts.append(
            f'<p class="t-p" style="margin:0 0 12px 0;">'
            f'{left}{mid}<span> • </span>{right}'
            f'</p>'
        )

    parts.append('<p class="t-hr" style="margin:18px 0 0 0; border-top:1px solid #E1E1E1; padding-top:10px;"></p>')
    return "\n".join(parts)


def render_country_block(country: str, items: pd.DataFrame, max_items: int) -> str:
    parts = [f'<div class="t-h2" style="margin:16px 0 8px 0; font-size:12pt; font-weight:700; color:#2B2B2B;">{html_escape(country.upper())}</div>']
    if items.empty:
        parts.append('<p class="t-p" style="margin:0 0 12px 0;">No qualifying items in the last 24 hours.</p>')
        parts.append('<p class="t-hr" style="margin:18px 0 0 0; border-top:1px solid #E1E1E1; padding-top:10px;"></p>')
        return "\n".join(parts)

    items = items.sort_values("datetime", ascending=False).head(max_items)

    for _, r in items.iterrows():
        headline = strip_emojis(r.get("headline", "") or "").strip()
        why = strip_emojis(r.get("why", "") or "").strip()
        url = r.get("url", "") or ""
        ch = normalize_channel(r.get("channel", "@unknown"))
        dt = r.get("datetime", pd.NaT)
        dt_s = fmt_dt_utc(dt) if pd.notna(dt) else ""

        parts.append(
            f'<p class="t-p t-li" style="margin:0 0 10px 0;">'
            f'<span class="t-accent" style="font-weight:700; color:#9F7E56;">- {html_escape(headline)}</span>'
            f'<span> — {html_escape(why)}</span> '
            f'(<a href="{html_escape(url)}">link</a>)'
            f'</p>'
        )
        parts.append(
            f'<p class="t-p t-muted" style="margin:0 0 12px 0; color:#9AA0A6;">'
            f'{html_escape(dt_s)} • {html_escape(ch)}'
            f'</p>'
        )

    parts.append('<p class="t-hr" style="margin:18px 0 0 0; border-top:1px solid #E1E1E1; padding-top:10px;"></p>')
    return "\n".join(parts)


# -----------------------------
# Email send / preview
# -----------------------------
def send_email_smtp(secrets: Dict[str, Any], subject: str, html_body: str) -> None:
    email_cfg = secrets.get("email", {})
    required = ["smtp_server", "smtp_port", "username", "password", "recipients", "from"]
    missing = [k for k in required if k not in email_cfg]
    if missing:
        raise KeyError(f"Missing email key(s) in secrets.json under 'email': {', '.join(missing)}")

    smtp_server = email_cfg["smtp_server"]
    smtp_port = int(email_cfg["smtp_port"])
    username = email_cfg["username"]
    password = email_cfg["password"]
    recipients = email_cfg["recipients"]
    from_name = email_cfg["from"]

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = f"{from_name} <{username}>"
    msg["To"] = ", ".join(recipients)

    msg.attach(MIMEText(html_body, "html", "utf-8"))

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(username, password)
        server.sendmail(username, recipients, msg.as_string())


def write_preview(path: str, html_body: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html_body)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize filtered Telegram CSV and email/preview it.")
    parser.add_argument("csv_positional", nargs="?", default=None, help="Path to filtered CSV (optional positional).")
    parser.add_argument("--csv", default=None, help="Path to filtered CSV.")
    parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours (UTC).")
    parser.add_argument("--secrets", default=str(Path("config/secrets.json")), help="Path to secrets.json.")
    parser.add_argument("--model", default=None, help="OpenAI model name (overrides secrets.openai.model).")
    parser.add_argument("--max_per_country", type=int, default=3, help="Max items per country (and Energy/Global blocks).")
    parser.add_argument("--prefix", default="Telegram Daily", help="Email subject prefix")
    parser.add_argument("--include_unknown", action="store_true", help="Include items with country=unknown (and Energy/Global blocks).")
    args = parser.parse_args()

    # -----------------------------
    # Resolve CSV path
    # -----------------------------
    csv_path = args.csv or args.csv_positional
    if not csv_path:
        base = Path("data/filtered")
        candidates = sorted(base.glob("filtered_*.csv"))
        if not candidates:
            raise FileNotFoundError("No CSV provided and no data/filtered/filtered_*.csv found.")
        csv_path = str(candidates[-1])

    # -----------------------------
    # Load secrets
    # -----------------------------
    secrets = load_secrets(args.secrets)

    # -----------------------------
    # OpenAI optional (only for summarise/translate + event dedupe)
    # -----------------------------
    api_key = secrets.get("openai", {}).get("api_key")
    model = args.model or secrets.get("openai", {}).get("model") or "gpt-4o-mini"
    client = None
    if api_key:
        try:
            client = get_openai_client(api_key)
        except Exception:
            client = None

    # -----------------------------
    # Time window
    # -----------------------------
    now_utc = datetime.now(timezone.utc)
    cutoff = now_utc - timedelta(hours=args.hours)

    # -----------------------------
    # Read + normalise
    # -----------------------------
    df = pd.read_csv(csv_path)
    df = normalise_dataframe(df)

    # If the CSV has no country column (e.g., energy keyword-only feed), default to unknown
    if "country" not in df.columns:
        df["country"] = "unknown"
    df["country"] = df["country"].fillna("unknown").astype(str).map(normalize_country)

    # Ensure skip exists
    if "skip" not in df.columns:
        df["skip"] = False
    df["skip"] = df["skip"].fillna(False).astype(bool)

    # Ensure english/raw exist
    if "english" not in df.columns:
        df["english"] = ""
    if "raw" not in df.columns:
        # energy feeds often use "text"
        df["raw"] = df["text"] if "text" in df.columns else ""

    # Window + skip filter
    df = df[pd.notna(df["datetime"])].copy()
    df = df[df["datetime"] >= cutoff].copy()
    df = df[df["skip"] == False].copy()

    # If not including unknown, drop them now
    if not args.include_unknown:
        df = df[df["country"] != "unknown"].copy()

    # -----------------------------
    # Load precomputed keyword alerts (optional)
    # -----------------------------
    alerts_path = secrets.get("alerts_csv")  # e.g. "data/alerts/alerts_latest.csv"
    df_kw = pd.DataFrame(columns=["datetime", "channel", "message_id", "url", "kw_terms", "snippet"])

    if alerts_path and Path(alerts_path).exists():
        try:
            df_kw = pd.read_csv(alerts_path)
        except pd.errors.EmptyDataError:
            df_kw = pd.DataFrame(columns=["datetime", "channel", "message_id", "url", "kw_terms", "snippet"])

        if "datetime" in df_kw.columns:
            df_kw["datetime"] = pd.to_datetime(df_kw["datetime"], errors="coerce", utc=True)

    # -----------------------------
    # Build per-row headline/why and (if needed) country via OpenAI
    # -----------------------------
    headlines: List[str] = []
    whys: List[str] = []
    countries: List[str] = []

    for _, r in df.iterrows():
        english = (r.get("english", "") or "").strip()

        raw = r.get("raw", None)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            raw = r.get("text", None)
        if raw is None or (isinstance(raw, float) and pd.isna(raw)):
            raw = r.get("original", "")
        raw = str(raw or "").strip()

        url = (r.get("url", "") or "").strip()

        text_for_processing = english if english else raw
        text_for_processing = strip_emojis(text_for_processing)

        country = normalize_country(r.get("country", "unknown"))

        needs_llm = (
            country == "unknown"
            or not english
            or looks_cyrillic(text_for_processing)
        )

        if client and needs_llm:
            obj = openai_classify_translate_and_summarise(client, model, text_for_processing, url)
            country = normalize_country(obj.get("country", "unknown"))
            headline = extract_full_sentence(obj.get("headline", "Update"))
            why = obj.get("why", "See source for details.")
        else:
            headline, why = fallback_summarise(text_for_processing)
            headline = extract_full_sentence(headline)

        headline = strip_emojis(headline).strip()
        why = strip_emojis(why).strip()

        if country not in APPROVED_SET:
            country = "unknown"

        countries.append(country)
        headlines.append(headline)
        whys.append(why)

    df["country"] = countries
    df["headline"] = headlines
    df["why"] = whys

    # -----------------------------
    # Subject date: prefer filename date so it matches the CSV
    # (supports filtered_YYYY-MM-DD.csv and filtered_energy_YYYY-MM-DD.csv)
    # -----------------------------
    csv_date = extract_csv_date_from_filename(csv_path)
    if csv_date is None:
        m = re.search(r"(?:filtered_energy)_(\d{4}-\d{2}-\d{2})\.csv$", os.path.basename(csv_path))
        csv_date = m.group(1) if m else None

    subject_line = build_subject(args.prefix, csv_date, now_utc)

    # -----------------------------
    # Render HTML
    # -----------------------------
    chunks: List[str] = []
    chunks.append(render_header(subject_line))

    # Keyword alerts: skip for Energy emails
    if not args.prefix.lower().startswith("energy"):
        chunks.append(render_keyword_alerts(df_kw))

    # Country blocks
    for c in APPROVED_COUNTRIES:
        country_items = df[df["country"] == c].sort_values("datetime", ascending=False).head(10)
        country_items = llm_event_dedupe_keep_rows(client, model, country_items, max_keep=args.max_per_country)
        chunks.append(render_country_block(c, country_items, args.max_per_country))

    # Energy/global fallback: include unknown
    if args.include_unknown:
        unknown_items = df[df["country"] == "unknown"].sort_values("datetime", ascending=False).head(25)
        unknown_items = llm_event_dedupe_keep_rows(client, model, unknown_items, max_keep=args.max_per_country)
        if not unknown_items.empty:
            chunks.append(render_country_block("Energy / Global", unknown_items, args.max_per_country))

    chunks.append(render_footer(csv_path))

    html = HTML_WRAPPER_START.format(style=HTML_STYLE) + "\n".join(chunks) + HTML_WRAPPER_END

    # -----------------------------
    # Send / preview
    # -----------------------------
    mail_mode = str(secrets.get("mail_mode", "preview")).lower().strip()
    if mail_mode == "smtp":
        send_email_smtp(secrets, subject=subject_line, html_body=html)
        print("✅ Email sent.")
    else:
        out_path = secrets.get("preview_out", str(Path("out_email_preview.html").absolute()))
        write_preview(out_path, html)
        print("⚠️ mail_mode is not 'smtp' so no email was sent.")
        print(f"✅ Wrote preview HTML to: {out_path}")
        print(f"  open {out_path}")


if __name__ == "__main__":
    main()