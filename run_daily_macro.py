#!/usr/bin/env python3
import asyncio
import argparse
from datetime import datetime
from src.config import load_config
from src.fetcher import Fetcher
from src.deduplicator import Deduplicator
from src.summarizer import Summarizer

async def main():
    parser = argparse.ArgumentParser(description="Daily Telegram Macro Pipeline")
    parser.add_argument("--hours", type=int, default=24, help="Lookback hours")
    parser.add_argument("--local-llm", action="store_true", help="Use local LLM (LM Studio) where possible")
    parser.add_argument("--dry-run", action="store_true", help="Skip sending email")
    args = parser.parse_args()

    config = load_config()
    start_time = datetime.now()
    print(f"=== Starting Daily Macro Pipeline ({start_time}) ===")
    
    # 1. Fetch & Filter
    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Step 1: Fetching & Filtering...")
    fetcher = Fetcher(config, topic="macro")
    df = await fetcher.run_pipeline(hours=args.hours, prefer_local_llm=args.local_llm)
    print(f"[{datetime.now().strftime('%H:%M:%S')}]     Fetched {len(df)} relevant items.")
    
    if df.empty:
        print("No items found. Exiting.")
        return

    # 2. Deduplicate (Text)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Step 2: Text Deduplication...")
    deduper = Deduplicator(config)
    df = deduper.dedup_by_text(df)
    print(f"[{datetime.now().strftime('%H:%M:%S')}]     After text dedup: {len(df)} items.")

    # 3. Deduplicate (LLM)
    # Only run if we still have items
    if not df.empty:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Step 3: Semantic Deduplication...")
        print(f"    Processing {len(df)} items against recent history...")
        df = deduper.dedup_by_llm(df, prefer_local_llm=args.local_llm)
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     After LLM dedup: {len(df)} items.")

    # 4. Summarize & Email
    print(f"[{datetime.now().strftime('%H:%M:%S')}] >>> Step 4: Summarize & Email...")
    summarizer = Summarizer(config)
    subject = f"Telegram Daily Macro – {datetime.now().strftime('%Y-%m-%d')}"
    html = summarizer.render_html(df, subject_line=subject)
    
    if args.dry_run:
        print(f"[{datetime.now().strftime('%H:%M:%S')}]     [Dry Run] Saving email preview to preview_macro.html")
        with open("preview_macro.html", "w", encoding="utf-8") as f:
            f.write(html)
    else:
        summarizer.send_email(html, subject)

    end_time = datetime.now()
    print(f"[{end_time.strftime('%H:%M:%S')}] === Pipeline Complete (Duration: {end_time - start_time}) ===")

if __name__ == "__main__":
    asyncio.run(main())
