#!/usr/bin/env python3
"""
Deduplicate_OpenAI_AfterFetch.py  (rewritten to support LM Studio)

Purpose
- Semantic deduplication of Telegram rows in a CSV using an OpenAI-compatible chat endpoint.
- Works with:
  - OpenAI (api.openai.com) OR
  - LM Studio local server (OpenAI-compatible), e.g. http://127.0.0.1:1234/v1

Typical pipeline
- Run cheap text dedupe first (SequenceMatcher/Jaccard).
- Then run THIS on the already-reduced CSV.

Output
- out_csv: kept rows (canonical)
- dups_report: duplicate decisions with match target + confidence + rationale

Notes
- Set temperature=0 for determinism.
- To use LM Studio: start the server and pass --base_url http://127.0.0.1:1234/v1 and --model <your-model-name>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- Optional dependency: OpenAI python SDK (v1) ---
try:
    from openai import OpenAI
except Exception as e:
    OpenAI = None  # type: ignore


# -----------------------------
# Helpers
# -----------------------------

def load_secrets(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def norm(s: str) -> str:
    s = (s or "").strip()
    s = re.sub(r"\s+", " ", s)
    return s

def to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default

def now_utc_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def pick_first(row: Dict[str, Any], keys: List[str], default: str = "") -> str:
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return str(row[k])
    return default

def parse_datetime(row: Dict[str, Any]) -> str:
    # Best-effort; keep as string for ordering
    return pick_first(row, ["datetime", "date", "time", "timestamp", "created_at"], "")

def compute_bucket(row: Dict[str, Any], bucket_by: str) -> str:
    bucket_by = (bucket_by or "global").lower().strip()
    if bucket_by == "global":
        return "global"
    if bucket_by == "country":
        c = pick_first(row, ["country", "Country", "COUNTRY"], "").strip()
        if c:
            return c.lower()
        ch = pick_first(row, ["channel", "Channel", "source", "Source"], "").strip()
        return f"unknown::{ch.lower()}" if ch else "unknown"
    if bucket_by == "channel":
        ch = pick_first(row, ["channel", "Channel", "source", "Source"], "").strip()
        return ch.lower() if ch else "unknown"
    # fallback: treat as column name
    v = row.get(bucket_by, "")
    v = str(v).strip()
    return v.lower() if v else "unknown"


# -----------------------------
# LLM judge
# -----------------------------

DEDUP_SYSTEM = (
    "You are a strict deduplication judge for a macro-intelligence Telegram feed.\n"
    "Task: decide whether two messages describe the SAME underlying news event.\n"
    "Rules:\n"
    "- SAME event even if wording differs, reposted, translated, minor edits, or different outlets.\n"
    "- NOT the same if the underlying event differs in: actor, location, date, asset, policy decision, or outcome.\n"
    "- If unsure, return NOT_DUP.\n"
    "Return ONLY valid JSON with keys:\n"
    "  decision: one of [DUP, NOT_DUP]\n"
    "  confidence: number 0.0-1.0\n"
    "  reason: short string (<= 25 words)\n"
)

def build_pair_prompt(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    # Choose the best text fields you have
    a_text = pick_first(a, ["text_en", "text", "message", "body", "translated", "summary"], "")
    b_text = pick_first(b, ["text_en", "text", "message", "body", "translated", "summary"], "")
    a_meta = {
        "datetime": parse_datetime(a),
        "channel": pick_first(a, ["channel", "source"], ""),
        "url": pick_first(a, ["url", "link"], ""),
        "country": pick_first(a, ["country"], ""),
    }
    b_meta = {
        "datetime": parse_datetime(b),
        "channel": pick_first(b, ["channel", "source"], ""),
        "url": pick_first(b, ["url", "link"], ""),
        "country": pick_first(b, ["country"], ""),
    }
    return (
        "Message A meta:\n"
        f"{json.dumps(a_meta, ensure_ascii=False)}\n"
        "Message A text:\n"
        f"{a_text}\n\n"
        "Message B meta:\n"
        f"{json.dumps(b_meta, ensure_ascii=False)}\n"
        "Message B text:\n"
        f"{b_text}\n"
    )

def call_llm_judge(
    client: Any,
    model: str,
    base_url: str,
    api_key: str,
    a: Dict[str, Any],
    b: Dict[str, Any],
    timeout_s: int = 60,
) -> Tuple[str, float, str]:
    """
    Returns (decision, confidence, reason)
    decision in {"DUP","NOT_DUP"}
    """
    prompt = build_pair_prompt(a, b)

    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": DEDUP_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=120,
        timeout=timeout_s,
    )

    content = (resp.choices[0].message.content or "").strip()

    # Robust parse: extract first JSON object if present
    m = re.search(r"\{.*?\}", content, flags=re.DOTALL)
    jtxt = m.group(0) if m else ""

    if not jtxt:
        # Fallback if model didn't output JSON
        up = content.upper()
        if "DUP" in up or up.startswith("YES"):
            return ("DUP", 0.5, "Non-JSON output")
        return ("NOT_DUP", 0.0, "Non-JSON output")

    try:
        obj = json.loads(jtxt)
    except Exception:
        return ("NOT_DUP", 0.0, "Parse failure")

    decision = str(obj.get("decision", "NOT_DUP")).strip().upper()
    conf = to_float(obj.get("confidence", 0.0), 0.0)
    reason = norm(str(obj.get("reason", "")))[:180]

    if decision not in ("DUP", "NOT_DUP"):
        decision = "NOT_DUP"
    conf = max(0.0, min(1.0, conf))
    if not reason:
        reason = "No reason"

    return (decision, conf, reason)


# -----------------------------
# Core dedup logic
# -----------------------------

@dataclass
class Decision:
    is_dup: bool
    confidence: float
    reason: str
    match_to_id: str

def choose_id(row: Dict[str, Any]) -> str:
    # Prefer message_id; else stable fallback
    mid = pick_first(row, ["message_id", "id", "msg_id"], "")
    if mid:
        return str(mid)
    # fallback stable-ish
    u = pick_first(row, ["url", "link"], "")
    if u:
        return u
    return f"row_{hash(pick_first(row, ['text_en','text','message'], '')) & 0xffffffff:x}"

def sort_key(row: Dict[str, Any]) -> Tuple[str, str]:
    # Sort by datetime then channel/id for stability
    dt = parse_datetime(row)
    ch = pick_first(row, ["channel", "source"], "")
    rid = choose_id(row)
    return (dt, ch + "|" + rid)

def deduplicate_bucket(
    rows: List[Dict[str, Any]],
    client: Any,
    model: str,
    base_url: str,
    api_key: str,
    min_confidence: float,
    fallback_when_low_conf: bool,
    max_compare_per_row: int = 25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns kept_rows, dups_report_rows
    """
    rows_sorted = sorted(rows, key=sort_key)

    kept: List[Dict[str, Any]] = []
    kept_index: List[Dict[str, Any]] = []  # alias for readability
    dups_report: List[Dict[str, Any]] = []

    for r in rows_sorted:
        r_id = choose_id(r)

        # Compare against recent kept items first (most likely duplicates).
        # We cap comparisons for speed.
        candidates = list(reversed(kept_index))[:max_compare_per_row]

        best: Optional[Decision] = None

        for k in candidates:
            k_id = choose_id(k)

            decision, conf, reason = call_llm_judge(
                client=client,
                model=model,
                base_url=base_url,
                api_key=api_key,
                a=r,
                b=k,
            )

            is_dup = (decision == "DUP")
            if best is None or conf > best.confidence:
                best = Decision(is_dup=is_dup, confidence=conf, reason=reason, match_to_id=k_id)

            # Early stop: high-confidence duplicate
            if is_dup and conf >= max(min_confidence, 0.85):
                break

        if best is None:
            kept.append(r)
            kept_index.append(r)
            continue

        # Decision rule
        if best.is_dup and best.confidence >= min_confidence:
            dups_report.append({
                "datetime": parse_datetime(r),
                "row_id": r_id,
                "dup_of": best.match_to_id,
                "decision": "DUP",
                "confidence": f"{best.confidence:.3f}",
                "reason": best.reason,
                "ts_utc": now_utc_iso(),
            })
        else:
            # If model says DUP but low confidence, optionally keep
            if best.is_dup and best.confidence < min_confidence and fallback_when_low_conf:
                kept.append(r)
                kept_index.append(r)
                dups_report.append({
                    "datetime": parse_datetime(r),
                    "row_id": r_id,
                    "dup_of": best.match_to_id,
                    "decision": "LOWCONF_DUP_KEPT",
                    "confidence": f"{best.confidence:.3f}",
                    "reason": best.reason,
                    "ts_utc": now_utc_iso(),
                })
            else:
                kept.append(r)
                kept_index.append(r)
                dups_report.append({
                    "datetime": parse_datetime(r),
                    "row_id": r_id,
                    "dup_of": best.match_to_id,
                    "decision": "NOT_DUP",
                    "confidence": f"{best.confidence:.3f}",
                    "reason": best.reason,
                    "ts_utc": now_utc_iso(),
                })

    return kept, dups_report


# -----------------------------
# CSV I/O
# -----------------------------

def read_csv(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return [dict(r) for r in reader]

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        # write empty with placeholder header
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(fieldnames or ["empty"])
        return

    if fieldnames is None:
        # keep original column order as best we can
        keys = []
        seen = set()
        for r in rows:
            for k in r.keys():
                if k not in seen:
                    seen.add(k)
                    keys.append(k)
        fieldnames = keys

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})

def write_dups_report(path: str, rows: List[Dict[str, Any]]) -> None:
    fields = ["datetime", "row_id", "dup_of", "decision", "confidence", "reason", "ts_utc"]
    write_csv(path, rows, fieldnames=fields)


# -----------------------------
# Main
# -----------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--dups_report", required=True)

    ap.add_argument("--bucket_by", default="country", help="global|country|channel|<column_name>")
    ap.add_argument("--max_per_bucket", type=int, default=250)

    ap.add_argument("--secrets", default="", help="Path to secrets.json (optional)")
    ap.add_argument("--model", default="", help="Model name. For LM Studio: the loaded model's ID/name.")
    ap.add_argument("--base_url", default="", help="OpenAI-compatible base URL. LM Studio: http://127.0.0.1:1234/v1")
    ap.add_argument("--api_key", default="", help="API key (OpenAI) or dummy (LM Studio can accept anything).")

    ap.add_argument("--min_confidence", type=float, default=0.15)
    ap.add_argument("--fallback_when_low_conf", action="store_true")

    ap.add_argument("--max_compare_per_row", type=int, default=25, help="Cap LLM pairwise checks per row")

    args = ap.parse_args()

    if OpenAI is None:
        print("ERROR: openai python package not available. Install with: pip install openai", file=sys.stderr)
        return 2

    secrets = load_secrets(args.secrets) if args.secrets else {}

    # Resolve endpoint/model
    base_url = (
        args.base_url
        or secrets.get("lmstudio_base_url")
        or secrets.get("openai_base_url")
        or os.environ.get("OPENAI_BASE_URL", "")
        or os.environ.get("LMSTUDIO_BASE_URL", "")
        or ""  # empty means OpenAI default
    )

    api_key = (
        args.api_key
        or secrets.get("openai_api_key")
        or secrets.get("api_key")
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("LMSTUDIO_API_KEY", "")
        or "lm-studio"  # fine for LM Studio
    )

    model = (
        args.model
        or secrets.get("lmstudio_model")
        or secrets.get("openai_model")
        or os.environ.get("LLM_MODEL", "")
    )

    if not model:
        # sensible defaults
        if base_url and "127.0.0.1" in base_url:
            model = "llama-3.1-8b-instruct"  # change to whatever LM Studio shows
        else:
            model = "gpt-4o-mini"

    # Create client
    client = OpenAI(api_key=api_key, base_url=base_url or None)

    print(f"[dedup] in_csv={args.in_csv}")
    print(f"[dedup] out_csv={args.out_csv}")
    print(f"[dedup] dups_report={args.dups_report}")
    print(f"[dedup] base_url={base_url or '(OpenAI default)'}")
    print(f"[dedup] model={model}")
    print(f"[dedup] bucket_by={args.bucket_by} max_per_bucket={args.max_per_bucket}")
    print(f"[dedup] min_confidence={args.min_confidence} fallback_when_low_conf={args.fallback_when_low_conf}")
    print(f"[dedup] max_compare_per_row={args.max_compare_per_row}")

    rows = read_csv(args.in_csv)
    if not rows:
        write_csv(args.out_csv, [])
        write_dups_report(args.dups_report, [])
        print("[dedup] input empty; done.")
        return 0

    # Bucket
    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        b = compute_bucket(r, args.bucket_by)
        buckets.setdefault(b, []).append(r)

    kept_all: List[Dict[str, Any]] = []
    report_all: List[Dict[str, Any]] = []


    for bkey, b_rows in buckets.items():
        if args.max_per_bucket and len(b_rows) > args.max_per_bucket:
            b_rows = sorted(b_rows, key=sort_key)[: args.max_per_bucket]

        kept, rep = deduplicate_bucket(
            rows=b_rows,
            client=client,
            model=model,
            base_url=base_url,
            api_key=api_key,
            min_confidence=args.min_confidence,
            fallback_when_low_conf=args.fallback_when_low_conf,
            max_compare_per_row=args.max_compare_per_row,
        )
        kept_all.extend(kept)
        report_all.extend(rep)
        print(f"[dedup] bucket={bkey} in={len(b_rows)} kept={len(kept)} rep={len(rep)}")

    # Write outputs
    # Preserve input fieldnames if possible
    input_fields = list(rows[0].keys())
    write_csv(args.out_csv, kept_all, fieldnames=input_fields)
    write_dups_report(args.dups_report, report_all)

    print(f"[dedup] DONE: kept={len(kept_all)} report_rows={len(report_all)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())