#!/usr/bin/env python3
"""
Deduplicate_LLM_AfterFetch.py
----------------------------------------------------------
Semantic dedupe using a local LLM via LM Studio OpenAI-compatible endpoint.

Strategy:
- Bucket by country (or global)
- For each bucket, build clusters of "same story" items
- Keep ONE item per cluster (latest by datetime by default)
- Write:
  - deduped CSV
  - duplicates report CSV

Requires:
- LM Studio running with OpenAI-compatible server
  e.g. http://127.0.0.1:1234/v1/chat/completions

Works with input CSV containing at least:
- datetime, url, raw (english optional), country optional (or from lm_json)
"""

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests


# ---------- helpers ----------

URL_RX = re.compile(r"https?://\S+|t\.me/\S+", re.I)
WS_RX = re.compile(r"\s+")

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
    try:
        return json.loads(s)
    except Exception:
        start, end = s.find("{"), s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start:end+1])
            except Exception:
                return {}
        return {}

def normalize_text(s: str) -> str:
    s = str(s or "")
    s = s.replace("\u200b", " ").replace("\ufeff", " ")
    s = URL_RX.sub(" ", s)
    s = s.lower()
    s = re.sub(r"[\U00010000-\U0010ffff]", "", s)  # emojis
    s = re.sub(r"[^0-9a-zа-яё]+", " ", s, flags=re.I)
    s = WS_RX.sub(" ", s).strip()
    return s

def best_text(row: pd.Series) -> str:
    eng = str(row.get("english", "") or "").strip()
    raw = str(row.get("raw", "") or "").strip()
    return eng if eng else raw

def parse_dt(x: Any) -> Optional[pd.Timestamp]:
    try:
        return pd.to_datetime(x, errors="coerce", utc=True)
    except Exception:
        return None

def truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[:n].rstrip() + "…"


# ---------- LM Studio client ----------

def lmstudio_chat(url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return (data.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()

def strip_code_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", s)
        s = re.sub(r"\s*```$", "", s).strip()
    return s

def first_json_object(s: str) -> Optional[str]:
    s = strip_code_fences(s)
    i, j = s.find("{"), s.rfind("}")
    if i >= 0 and j > i:
        return s[i:j+1]
    return None


# ---------- semantic clustering ----------

@dataclass
class Item:
    idx: int
    dt: pd.Timestamp
    url: str
    text: str
    norm: str

def cluster_bucket_llm(
    items: List[Item],
    lm_url: str,
    lm_model: str,
    max_items_per_call: int = 18,
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Returns:
      keep_indices: list of idx to keep
      dups_report_rows: list of dict rows
    """

    # We do incremental clustering:
    # - maintain clusters as lists of indices into `items`
    # - for each new item, ask LLM if it matches an existing cluster (compare to cluster "representative")
    clusters: List[List[int]] = []
    reps: List[int] = []  # index into items list

    dups_rows: List[Dict[str, Any]] = []

    def rep_summary(it: Item) -> str:
        return truncate(it.text, 240)

    for k, it in enumerate(items):
        if not clusters:
            clusters.append([k])
            reps.append(k)
            continue

        # Build a prompt comparing this item to each cluster representative
        # Keep prompt small: show rep snippets.
        rep_blocks = []
        for ci, rk in enumerate(reps):
            rep_blocks.append(f"{ci}: {rep_summary(items[rk])}")
        rep_text = "\n".join(rep_blocks)

        prompt = (
            "You are deduplicating a Telegram news digest.\n"
            "Decide whether NEW_ITEM is the SAME underlying news story as one of the CLUSTER_REPS.\n"
            "Same story means: same event/announcement/update, even if phrased differently.\n"
            "If none match, answer -1.\n\n"
            "Return ONLY JSON: {\"match\": <cluster_index_or_-1>, \"reason\": \"...\"}\n\n"
            f"CLUSTER_REPS:\n{rep_text}\n\n"
            f"NEW_ITEM:\n{rep_summary(it)}\n"
        )

        out = lmstudio_chat(lm_url, lm_model, prompt, temperature=0.0)
        j = first_json_object(out) or out
        match = -1
        reason = ""
        try:
            obj = json.loads(j)
            match = int(obj.get("match", -1))
            reason = str(obj.get("reason", "") or "")
        except Exception:
            match = -1
            reason = "LLM parse failed"

        if 0 <= match < len(clusters):
            # add to cluster; record dup link
            clusters[match].append(k)

            kept_k = reps[match]
            # Keep latest by datetime (swap rep if new is later)
            if it.dt > items[kept_k].dt:
                reps[match] = k
                kept_k = k

            dups_rows.append({
                "kept_url": items[reps[match]].url,
                "dropped_url": it.url,
                "kept_dt": str(items[reps[match]].dt),
                "dropped_dt": str(it.dt),
                "reason": reason[:300],
            })
        else:
            clusters.append([k])
            reps.append(k)

    keep_item_idxs = [items[rk].idx for rk in reps]
    return keep_item_idxs, dups_rows


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dups_report", required=True)
    ap.add_argument("--bucket_by", choices=["global", "country", "channel"], default="country")
    ap.add_argument("--lm_url", default="http://127.0.0.1:1234/v1/chat/completions")
    ap.add_argument("--lm_model", default="local-model")
    ap.add_argument("--max_per_bucket", type=int, default=400, help="Safety cap per bucket")
    args = ap.parse_args()

    in_path = Path(args.in_csv)
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    df = pd.read_csv(in_path)

    # Ensure minimal cols
    for c in ["datetime", "url", "raw"]:
        if c not in df.columns:
            df[c] = ""

    # Derive country from lm_json if needed
    if "country" not in df.columns and "lm_json" in df.columns:
        lm = df["lm_json"].map(safe_parse_lm_json)
        df["country"] = lm.map(lambda d: str(d.get("country", "unknown") or "unknown"))

    if "country" not in df.columns:
        df["country"] = "unknown"

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df[pd.notna(df["datetime"])].copy()
    df = df.sort_values("datetime", ascending=False).reset_index(drop=True)

    # Buckets
    if args.bucket_by == "global":
        df["_bucket"] = "global"
    elif args.bucket_by == "channel":
        df["_bucket"] = df.get("channel", "@unknown").fillna("@unknown")
    else:
        df["_bucket"] = df.get("country", "unknown").fillna("unknown")

    keep_mask = [False] * len(df)
    report_rows: List[Dict[str, Any]] = []

    for bucket, idxs in df.groupby("_bucket").groups.items():
        idxs = list(idxs)
        if not idxs:
            continue

        # Safety cap: only LLM-dedupe most recent N in each bucket
        idxs = idxs[: args.max_per_bucket]

        items: List[Item] = []
        for i in idxs:
            row = df.loc[i]
            txt = best_text(row)
            it = Item(
                idx=int(i),
                dt=row["datetime"],
                url=str(row.get("url", "") or ""),
                text=txt,
                norm=normalize_text(txt),
            )
            if it.norm:
                items.append(it)

        if len(items) <= 1:
            for it in items:
                keep_mask[it.idx] = True
            continue

        keep_idxs, dups = cluster_bucket_llm(items, args.lm_url, args.lm_model)
        for ki in keep_idxs:
            keep_mask[ki] = True
        for d in dups:
            d["bucket"] = bucket
            report_rows.append(d)

    df_out = df.loc[keep_mask].drop(columns=["_bucket"], errors="ignore")
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dups_report).parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(args.out_csv, index=False)

    rep = pd.DataFrame(report_rows)
    if rep.empty:
        # write headers so downstream pd.read_csv never fails
        rep = pd.DataFrame(columns=["bucket","kept_url","dropped_url","kept_dt","dropped_dt","reason"])
    rep.to_csv(args.dups_report, index=False)

    print(f"✅ LLM deduped: {len(df)} → {len(df_out)} rows")
    print(f"✅ Wrote: {args.out_csv}")
    print(f"✅ Dups report: {args.dups_report} (rows={len(report_rows)})")


if __name__ == "__main__":
    main()