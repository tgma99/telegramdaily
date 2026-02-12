import re
import pandas as pd
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.config import Config
from src.llm_client import LLMClient
from src.telegram_client import TelegramFetcher

# Regex helpers
URL_RX = re.compile(r"https?://\S+|t\.me/\S+", re.I)
WS_RX = re.compile(r"\s+")
EMOJI_RX = re.compile(
    r"[\U00010000-\U0010ffff]|"  # Emojis and other non-BMP symbols
    r"[\u2600-\u27bf]|"          # Misc symbols/Dingbats
    r"[\u203c-\u204b]|"          # Rare double punctuation (like ⁉️)
    r"[\u2500-\u2b5f]|"          # Geometric shapes, arrows, etc.
    r"[\u3000-\u303f]|"          # CJK punctuation
    r"[\ufe0e\ufe0f]",           # Variation selectors
    re.UNICODE
)

def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Strip zero-width space
    s = s.replace("\u200b", "")
    # Remove emojis and special symbols
    s = EMOJI_RX.sub("", s)
    # Collapse whitespace
    s = WS_RX.sub(" ", s).strip()
    return s

def message_url(channel: str, msg_id: int) -> str:
    return f"https://t.me/{channel.lstrip('@')}/{msg_id}"

class Fetcher:
    def __init__(self, config: Config, topic: str = "macro"):
        self.config = config
        self.topic = topic
        self.llm = LLMClient(config.secrets)
        self.tg = TelegramFetcher(config.secrets)
        
        # Load topic-specific channels
        self.channels = config.channels.get(topic, [])
        if not self.channels:
            print(f"⚠️ Warning: No channels found for topic '{topic}'")

    APPROVED_COUNTRIES = {
        "Russia", "Kazakhstan", "Kyrgyz Republic", "Uzbekistan", "Tajikistan",
        "Turkmenistan", "Azerbaijan", "Georgia", "Armenia", "Ukraine", "Belarus",
        "Mongolia"
    }

    def _build_prompt(self, text: str) -> str:
        # We can make this configurable via prompts.yaml later
        
        if self.topic == "pharma":
            return f"""
You are a classifier for a multinational pharmaceutical company's Regulatory Affairs department.
Output ONLY a single compact JSON object.

Detect Telegram messages relevant to:
- REGULATORY ENVIRONMENT for medicinal products (policy, registration, pricing, reimbursement, import/export, sanctions, tenders).
- DEMAND for medicines and vaccines (epidemics, outbreaks, shortages, treatment protocols).

CRITICAL RULES:
1. Focus ONLY on: {", ".join(sorted(self.APPROVED_COUNTRIES))}, plus Mongolia. (If another country like France or USA is the source, it must involve the listed countries).
2. If NOT related to pharma regulation, access, or demand, return {{"skip": true}}.
3. If about general culture or personal blogs, return {{"skip": true}}.

If relevant:
- country: primary country affected (from the allowed list).
- category: "pharma"
- subject: short comma-separated list of topics (e.g. pricing, shortage, vaccine, tenders).
- english: clear English summary (max 3-4 sentences) focusing on regulatory impact, access, or demand.

Output JSON schema:
{{
  "skip": false,
  "country": "...",
  "category": "pharma",
  "subject": "...",
  "english": "..."
}}

Telegram message:
{text}
""".strip()

        # Default / Macro / Energy Prompt
        return f"""
You are a strict classifier. Output ONLY a single compact JSON object and nothing else.

Decide if the Telegram message is about one of:
- "domestic politics"
- "foreign relations"
- "macroeconomics"
- "energy"

CRITICAL RULES:
1. If the message is NOT primarily about one of these countries: {", ".join(sorted(self.APPROVED_COUNTRIES))}, return {{"skip": true}}.
2. If the message is about culture, language, tourism, or personal blogs (even if from these countries), return {{"skip": true}}. 
3. If the message is about a country NOT in the list (e.g. France, USA, China) and does not involve the listed countries, return {{"skip": true}}.

If relevant:
- category: one of the above keys (lowercase)
- country: ONE primary country from the list above (string).
- subject: comma-separated list of key topics (e.g. GDP, inflation, oil, gas).
- english: clear English translation.

Output JSON schema:
{{
  "skip": false,
  "category": "...",
  "country": "...",
  "subject": "...",
  "english": "..."
}}

Telegram message:
{text}
""".strip()

    async def run_pipeline(self, hours: int = 24, prefer_local_llm: bool = False) -> pd.DataFrame:
        rows = []
        stats = {"fetched": 0, "kept": 0, "skipped": 0, "errors": 0, "topic_mismatch": 0, "country_mismatch": 0}

        async for channel, msgs in self.tg.fetch_all(self.channels, lookback_hours=hours):
            stats["fetched"] += len(msgs)
            
            for msg in reversed(msgs):
                original = normalize_text(getattr(msg, "message", "") or "")
                original = original.replace("\x00", "").strip()
                if not original:
                    continue

                # Skip short URL-only messages
                if len(original) < 40 and URL_RX.search(original):
                    stats["skipped"] += 1
                    continue

                # Classify
                prompt = self._build_prompt(original)
                res = self.llm.classify_and_translate(
                    prompt=prompt,
                    prefer_local=prefer_local_llm
                )

                if res.get("skip") is True:
                    stats["skipped"] += 1
                    continue
                
                # Strict Country Validation
                country = res.get("country", "unknown")
                # Normalize check: strict match against allowed set
                # (You might want to handle partial matches like 'Russia / China' if LLM ignores instruction)
                if country not in self.APPROVED_COUNTRIES:
                     # Check if it's a known alias or if valid country is contained in string?
                     # For valid strictness, we trust the LLM followed "ONE primary country"
                     # But let's be safe against small variances or "Russia, China"
                     found_valid = False
                     for c in self.APPROVED_COUNTRIES:
                         if c.lower() in country.lower():
                             country = c # Normalize to the valid string
                             found_valid = True
                             break
                     
                     if not found_valid:
                         stats["country_mismatch"] += 1
                         print(f"  (skip-country) {channel}/{msg.id} -> {country} (not in allowed list)")
                         continue
                
                # Filter by topic if specific topic is meant to be enforced
                # If self.topic is generic like "macro", we might accept multiple, 
                # but for "energy", we likely only want energy.
                # Assuming simple strict equality for now if topic is set.
                category = res.get("category", "").lower()
                if self.topic and self.topic != "macro" and category != self.topic:
                    # For macro, we might allow politics/foreign relations/macro
                    # For energy, we strictly want energy
                    if self.topic == "energy" and category != "energy":
                         stats["topic_mismatch"] += 1
                         continue

                rows.append({
                    "datetime": msg.date.isoformat(),
                    "channel": channel,
                    "message_id": msg.id,
                    "url": message_url(channel, msg.id),
                    "category": category,
                    # Save the normalized valid country if we fixed it, or original
                    "country": country, 
                    "subject": res.get("subject", ""),
                    "original": original,
                    "english": normalize_text(res.get("english", ""))
                })
                stats["kept"] += 1
                print(f"  (keep) {channel}/{msg.id} -> {country} | {category}")

        print(f"Stats: {stats}")
        
        df = pd.DataFrame(rows)
        if not df.empty:
             df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
             df = df.sort_values(["datetime", "country"], ascending=True)
             
        return df
