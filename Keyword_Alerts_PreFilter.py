#!/usr/bin/env python3
"""
Keyword_Alerts_PreFilter.py
----------------------------------------------------------
Scans Telegram messages (raw + english if present) for a watchlist and writes
an alerts CSV.

Watchlist (current):
- AgroTerra (+ common variants)
- АгроТерра (+ common variants)
- NCH Capital (+ NCH variants)
- Chevron

Matching rules:
- Case-insensitive
- Word-ish boundaries to reduce substring false positives
- Normalizes common zero-width/NBSP/dash characters before matching
"""

import argparse
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pandas as pd


# Canonical terms you want to *report* in kw_terms
WATCH_TERMS = [
    "AgroTerra",
    "АгроТерра",
    "NCH Capital",
    "Chevron",
]

# Boundary templates (avoid matching inside longer alphanumeric runs)
LATIN_BOUNDARY = r"(?<![A-Za-z0-9]){term}(?![A-Za-z0-9])"
CYRILLIC_BOUNDARY = r"(?<![А-Яа-яЁё0-9]){term}(?![А-Яа-яЁё0-9])"


def normalize_text(s: str) -> str:
    """Normalize weird whitespace/dashes/zero-width chars that often appear in Telegram text."""
    if not s:
        return ""
    s = str(s)

    # Common invisible chars
    s = s.replace("\u200b", "")   # zero-width space
    s = s.replace("\ufeff", "")   # BOM
    s = s.replace("\u2060", "")   # word joiner

    # NBSP / thin space etc -> normal space
    s = s.replace("\u00a0", " ")
    s = s.replace("\u2009", " ")
    s = s.replace("\u202f", " ")

    # Normalize common dashes to hyphen
    for ch in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2212"]:
        s = s.replace(ch, "-")

    # Collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_patterns():
    """
    Returns list of tuples: (canonical_term, compiled_regex)
    Includes practical variants to reduce misses without increasing false positives.
    """
    patterns = []

    # AgroTerra variants: AgroTerra / Agro Terra / Agro-Terra
    agro_variants = [
        r"AgroTerra",
        r"Agro[\s\-]+Terra",
    ]
    # АгроТерра variants: АгроТерра / Агро Терра / Агро-Терра
    agro_cyr_variants = [
        r"АгроТерра",
        r"Агро[\s\-]+Терра",
    ]
    # NCH Capital variants: NCH Capital / NCH-Capital / NCH (optional Capital)
    nch_variants = [
        r"NCH[\s\-]+Capital",
        r"NCH",  # if you only want the full phrase, remove this line
    ]
    # Chevron: Chevron (keep strict)
    chev_variants = [
        r"Chevron",
    ]

    # Helper to compile a list of variants under one canonical term
    def add_term(canonical: str, variant_regexes, is_cyrillic: bool):
        for v in variant_regexes:
            term = v  # already regex
            if is_cyrillic:
                pat = CYRILLIC_BOUNDARY.format(term=term)
            else:
                pat = LATIN_BOUNDARY.format(term=term)
            patterns.append((canonical, re.compile(pat, re.IGNORECASE)))

    add_term("AgroTerra", agro_variants, is_cyrillic=False)
    add_term("АгроТерра", agro_cyr_variants, is_cyrillic=True)
    add_term("NCH Capital", nch_variants, is_cyrillic=False)
    add_term("Chevron", chev_variants, is_cyrillic=False)

    return patterns


from datetime import datetime, timezone, timedelta
import pandas as pd

def scan_df(df: pd.DataFrame, hours: int | None = None) -> pd.DataFrame:
    patterns = build_patterns()

    # Work on a copy to avoid side-effects
    df = df.copy()

    # Ensure required columns exist
    for col in ["datetime", "channel", "message_id", "url", "raw"]:
        if col not in df.columns:
            df[col] = ""

    # Some pipelines also have "english" — include it if available to improve hits
    if "english" not in df.columns:
        df["english"] = ""

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)

    # Optional time window
    if hours is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        df = df[pd.notna(df["datetime"])].copy()
        df = df[df["datetime"] >= cutoff].copy()

    hits = []
    for _, row in df.iterrows():
        raw = normalize_text(row.get("raw", "") or "")
        eng = normalize_text(row.get("english", "") or "")

        # Prefer english if present; still scan both
        haystack = f"{eng}\n{raw}".strip()

        matched_terms = []
        for canonical, rx in patterns:
            if rx.search(haystack):
                matched_terms.append(canonical)

        if matched_terms:
            hits.append(
                {
                    "datetime": row.get("datetime"),
                    "channel": row.get("channel"),
                    "message_id": row.get("message_id"),
                    "url": row.get("url"),
                    "kw_terms": ", ".join(sorted(set(matched_terms))),
                    "snippet": (haystack[:400] + "…") if len(haystack) > 400 else haystack,
                }
            )

    # IMPORTANT: define columns that match what we actually create
    out = pd.DataFrame(
        hits,
        columns=["datetime", "channel", "message_id", "url", "kw_terms", "snippet"]
    )
    
    if not out.empty:
        out = out.sort_values("datetime", ascending=False)

    return out


def explain_match(text: str) -> None:
    patterns = build_patterns()
    t = normalize_text(text)
    print("Testing snippet against watchlist...\n")
    any_hit = False
    for canonical, rx in patterns:
        m = rx.search(t)
        if m:
            any_hit = True
            span = m.span()
            print(f"✅ MATCH: {canonical} at {span} -> {t[span[0]:span[1]]!r}")
    if not any_hit:
        print("✅ No matches.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", help="Input CSV from Telegram fetch stage")
    ap.add_argument("--out_dir", default="data/alerts", help="Directory to write alerts CSVs")
    ap.add_argument("--hours", type=int, default=None, help="Only scan last N hours (UTC). Example: 24")
    ap.add_argument("--write_flagged_source", action="store_true",
                    help="Also write a flagged_source CSV (source rows subset) if rows>0")
    ap.add_argument("--write_latest", action="store_true",
                    help="Also write/overwrite alerts_latest.csv in out_dir")
    ap.add_argument("--test_snippet", default=None,
                    help="Debug: pass a snippet string to see what would match")
    args = ap.parse_args()

    if args.test_snippet:
        explain_match(args.test_snippet)
        return

    if not args.csv:
        raise SystemExit("❌ Please provide --csv (or use --test_snippet).")

    in_path = Path(args.csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)
    alerts = scan_df(df, hours=args.hours)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    alerts_path = out_dir / f"alerts_{stamp}.csv"
    alerts.to_csv(alerts_path, index=False)
    print(f"✅ Wrote alerts CSV: {alerts_path} (rows={len(alerts)})")

    if args.write_latest:
        latest_path = out_dir / "alerts_latest.csv"
        alerts.to_csv(latest_path, index=False)
        print(f"✅ Wrote latest alerts: {latest_path}")

    # If you really want a "source subset" file, reconstruct it from original df using message_id/url
    if args.write_flagged_source and len(alerts) > 0:
        # Keep only columns that exist in the source
        keep_cols = [c for c in df.columns if c in ["datetime", "channel", "message_id", "url", "raw", "english", "lm_json"]]
        flagged = df.copy()

        # Filter by message_id when possible; else by url
        if "message_id" in alerts.columns and "message_id" in flagged.columns:
            flagged = flagged[flagged["message_id"].isin(alerts["message_id"])].copy()
        elif "url" in alerts.columns and "url" in flagged.columns:
            flagged = flagged[flagged["url"].isin(alerts["url"])].copy()

        if keep_cols:
            flagged = flagged[keep_cols]

        flagged_path = out_dir / f"flagged_source_{stamp}.csv"
        flagged.to_csv(flagged_path, index=False)
        print(f"✅ Wrote flagged source CSV: {flagged_path} (rows={len(flagged)})")


if __name__ == "__main__":
    main()