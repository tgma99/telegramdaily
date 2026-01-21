#!/usr/bin/env python3
"""
Deduplicate_AfterFetch.py
----------------------------------------------------------
Run this after Telegram_Fetch_Filter.py.

Input CSV expected (at minimum):
- datetime (UTC-ish parseable)
- channel
- message_id (optional but useful)
- url
- raw
Optional:
- english
- lm_json
- country (if already present)

Output:
- deduped CSV (same columns as input; drops internal _text/_norm/_bucket columns)
- duplicates report CSV (ALWAYS written with headers; may be 0 rows)

Dedup logic:
- normalize text (remove URLs, boilerplate, emojis, punctuation; lower; collapse whitespace)
- near dup: Jaccard(token_set) >= jaccard_min AND SequenceMatcher >= sim_threshold
- optional bucketing: global / channel / country
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from pandas.errors import EmptyDataError
from difflib import SequenceMatcher


# -----------------------------
# Normalisation helpers
# -----------------------------
URL_RX = re.compile(r"https?://\S+|t\.me/\S+", re.IGNORECASE)
WS_RX = re.compile(r"\s+")
# Keep letters/numbers in Latin + Cyrillic; replace everything else with space
NON_ALNUM_RX = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")


def safe_parse_lm_json(x: Any) -> Dict[str, Any]:
    """Parse lm_json which might be dict, JSON string, or string containing JSON."""
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
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return {}
        return {}


def normalize_text(s: Any) -> str:
    """
    More aggressive normalisation:
    - drop URLs
    - drop common Telegram boilerplate tails
    - lower
    - strip emojis (astral)
    - keep only Latin/Cyrillic letters + digits
    - collapse whitespace
    """
    if not s:
        return ""
    s = str(s)
    s = s.replace("\u200b", " ").replace("\ufeff", " ")

    # Remove URLs (reposts differ mainly by links)
    s = URL_RX.sub(" ", s)

    # Remove common boilerplate (both EN/RU-ish). Keep it conservative.
    # (This removes "subscribe..." lines and similar tails.)
    s = re.sub(r"\b(subscribe|подписывайтесь|читайте также)\b.*$", " ", s, flags=re.I)

    # Lowercase
    s = s.lower()

    # Remove emojis / astral plane chars
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)

    # Keep letters + digits (Latin + Cyrillic)
    s = NON_ALNUM_RX.sub(" ", s)

    # Collapse whitespace
    s = WS_RX.sub(" ", s).strip()
    return s


def text_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# -----------------------------
# Output schema for dup report
# -----------------------------
DUPS_COLUMNS = [
    "bucket",
    "sim",
    "jaccard",
    "kept_idx",
    "dropped_idx",
    "kept_url",
    "dropped_url",
    "kept_channel",
    "dropped_channel",
    "kept_datetime",
    "dropped_datetime",
    "kept_text",
    "dropped_text",
]


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Deduplicate Telegram messages after fetch/filter stage")
    ap.add_argument("--in_csv", required=True, help="Input filtered CSV")
    ap.add_argument("--out_csv", required=True, help="Output deduplicated CSV")
    ap.add_argument("--dups_report", required=True, help="Duplicates report CSV")
    ap.add_argument("--bucket_by", choices=["global", "country", "channel"], default="global")
    ap.add_argument("--sim_threshold", type=float, default=0.90)
    ap.add_argument("--jaccard_min", type=float, default=0.40)
    ap.add_argument("--prefer_col", choices=["english", "raw", "auto"], default="auto")
    ap.add_argument("--max_pairs_per_bucket", type=int, default=250_000)
    ap.add_argument("--keep_latest", action="store_true", help="If set, keep newest item and drop older duplicates")
    ap.add_argument("--debug_sample", type=int, default=0, help="Limit processing to first N rows (after time sort)")
    ap.add_argument("--debug_top_pairs", type=int, default=0, help="Print top similar pairs (for tuning thresholds)")
    ap.add_argument("--derive_country_from_lm_json", action="store_true",
                    help="If bucket_by=country and 'country' missing, try deriving from lm_json")

    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    out_path = Path(args.out_csv)
    dups_path = Path(args.dups_report)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dups_path.parent.mkdir(parents=True, exist_ok=True)

    # Read input
    try:
        df = pd.read_csv(in_path)
    except EmptyDataError:
        # Write empty outputs with headers
        pd.DataFrame().to_csv(out_path, index=False)
        pd.DataFrame(columns=DUPS_COLUMNS).to_csv(dups_path, index=False)
        print("✅ Empty input → empty outputs written")
        return

    if df.empty:
        df.to_csv(out_path, index=False)
        pd.DataFrame(columns=DUPS_COLUMNS).to_csv(dups_path, index=False)
        print("✅ No rows in input → outputs written")
        return

    # Ensure required columns exist
    for col in ["datetime", "channel", "message_id", "url", "raw"]:
        if col not in df.columns:
            df[col] = ""

    # Optional columns
    if "english" not in df.columns:
        df["english"] = ""
    if "lm_json" not in df.columns:
        df["lm_json"] = ""

    # Datetime
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    # Debug sampling
    if args.debug_sample and len(df) > args.debug_sample:
        df = df.head(args.debug_sample).copy().reset_index(drop=True)

    # Choose text column
    if args.prefer_col == "english":
        df["_text"] = df["english"].fillna("")
    elif args.prefer_col == "raw":
        df["_text"] = df["raw"].fillna("")
    else:
        # auto: prefer english if non-empty else raw (row-wise)
        eng = df["english"].fillna("")
        raw = df["raw"].fillna("")
        df["_text"] = eng.where(eng.str.strip().astype(bool), raw)

    # Normalize ONCE
    df["_norm"] = df["_text"].map(normalize_text)

    # Bucketing
    if args.bucket_by == "global":
        df["_bucket"] = "global"
    elif args.bucket_by == "channel":
        df["_bucket"] = df["channel"].fillna("@unknown").astype(str)
    else:
        # country
        if "country" not in df.columns:
            df["country"] = ""

        if args.derive_country_from_lm_json:
            # Fill missing country from lm_json if possible
            def get_country(row) -> str:
                c = str(row.get("country", "") or "").strip()
                if c:
                    return c
                lm = safe_parse_lm_json(row.get("lm_json"))
                return str(lm.get("country", "") or "").strip()

            df["country"] = df.apply(get_country, axis=1)

        df["_bucket"] = df["country"].fillna("unknown").astype(str)
        df.loc[~df["_bucket"].str.strip().astype(bool), "_bucket"] = "unknown"

    keep = [True] * len(df)
    dups_rows: List[Dict[str, Any]] = []
    debug_pairs: List[Tuple[float, float, str, int, int]] = []

    # Compare within buckets
    for bucket, idxs in df.groupby("_bucket").groups.items():
        idxs = list(idxs)
        n = len(idxs)
        if n < 2:
            continue

        # Guard against O(n^2) explosion
        if n * (n - 1) // 2 > args.max_pairs_per_bucket:
            print(f"⚠️ Skipping bucket '{bucket}' (too many pairs: {n*(n-1)//2})")
            continue

        for ii in range(n):
            i = idxs[ii]
            if not keep[i]:
                continue
            a_norm = df.loc[i, "_norm"]
            if not a_norm:
                continue

            for jj in range(ii + 1, n):
                j = idxs[jj]
                if not keep[j]:
                    continue
                b_norm = df.loc[j, "_norm"]
                if not b_norm:
                    continue

                sim = text_similarity(a_norm, b_norm)
                jac = jaccard_similarity(a_norm, b_norm)

                if args.debug_top_pairs:
                    debug_pairs.append((sim, jac, str(bucket), i, j))

                if sim >= args.sim_threshold and jac >= args.jaccard_min:
                    # Decide which to keep/drop
                    if args.keep_latest:
                        # df is sorted newest first, so i < j implies i is newer
                        kept, dropped = (i, j)
                    else:
                        # Keep the earlier-seen "j" (older) and drop "i" (newer)
                        # (This is your previous behavior; keep_latest flips it.)
                        kept, dropped = (j, i)

                    keep[dropped] = False

                    dups_rows.append(
                        {
                            "bucket": bucket,
                            "sim": sim,
                            "jaccard": jac,
                            "kept_idx": kept,
                            "dropped_idx": dropped,
                            "kept_url": df.loc[kept, "url"],
                            "dropped_url": df.loc[dropped, "url"],
                            "kept_channel": df.loc[kept, "channel"],
                            "dropped_channel": df.loc[dropped, "channel"],
                            "kept_datetime": df.loc[kept, "datetime"],
                            "dropped_datetime": df.loc[dropped, "datetime"],
                            "kept_text": (df.loc[kept, "_norm"] or "")[:250],
                            "dropped_text": (df.loc[dropped, "_norm"] or "")[:250],
                        }
                    )

    # Write outputs
    df_out = df.loc[keep].drop(columns=["_text", "_norm", "_bucket"], errors="ignore")
    df_out.to_csv(out_path, index=False)

    # IMPORTANT: Always write dups report with headers (even if 0 rows)
    dups_df = pd.DataFrame(dups_rows, columns=DUPS_COLUMNS)
    dups_df.to_csv(dups_path, index=False)

    print(f"✅ Deduped: {len(df)} → {len(df_out)} rows")
    print(f"✅ Wrote: {out_path}")
    print(f"✅ Duplicates report: {dups_path} (rows={len(dups_df)})")

    # Debug print
    if args.debug_top_pairs and debug_pairs:
        debug_pairs.sort(key=lambda x: (x[0], x[1]), reverse=True)
        print("\n🔎 Top similar pairs:")
        for sim, jac, bucket, i, j in debug_pairs[: args.debug_top_pairs]:
            print(f"[{bucket}] sim={sim:.3f} jac={jac:.3f}")
            print(" A:", (df.loc[i, "_norm"] or "")[:200])
            print(" B:", (df.loc[j, "_norm"] or "")[:200])


if __name__ == "__main__":
    main()