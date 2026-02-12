#!/usr/bin/env python3
import asyncio
import argparse
from datetime import datetime
from src.config import load_config
from src.fetcher import Fetcher
from src.deduplicator import Deduplicator
from src.summarizer import Summarizer

async def main():
    parser = argparse.ArgumentParser(description="Weekly Telegram Energy Pipeline")
    parser.add_argument("--hours", type=int, default=168, help="Lookback hours (default 7 days)")
    parser.add_argument("--local-llm", action="store_true", help="Use local LLM (LM Studio) where possible")
    parser.add_argument("--dry-run", action="store_true", help="Skip sending email")
    args = parser.parse_args()

    config = load_config()
    print(f"=== Starting Weekly Energy Pipeline ({datetime.now()}) ===")
    
    # 1. Fetch & Filter
    print(">>> Step 1: Fetching & Filtering (Energy)...")
    fetcher = Fetcher(config, topic="energy")
    df = await fetcher.run_pipeline(hours=args.hours, prefer_local_llm=args.local_llm)
    print(f"    Fetched {len(df)} relevant items.")
    
    if df.empty:
        print("No items found. Exiting.")
        return

    # 2. Deduplicate (Text)
    print(">>> Step 2: Text Deduplication...")
    deduper = Deduplicator(config)
    df = deduper.dedup_by_text(df)
    print(f"    After text dedup: {len(df)} items.")

    # 3. Deduplicate (LLM)
    if not df.empty:
        print(">>> Step 3: Semantic Deduplication...")
        df = deduper.dedup_by_llm(df, prefer_local_llm=args.local_llm)
        print(f"    After LLM dedup: {len(df)} items.")

    # 4. Summarize & Email
    print(">>> Step 4: Summarize & Email...")
    summarizer = Summarizer(config)
    subject = f"Telegram Weekly Energy – {datetime.now().strftime('%Y-%m-%d')}"
    html = summarizer.render_html(df, subject_line=subject)
    
    if args.dry_run:
        print("    [Dry Run] Saving email preview to preview_energy.html")
        with open("preview_energy.html", "w", encoding="utf-8") as f:
            f.write(html)
    else:
        summarizer.send_email(html, subject)

    print("=== Pipeline Complete ===")

if __name__ == "__main__":
    asyncio.run(main())
