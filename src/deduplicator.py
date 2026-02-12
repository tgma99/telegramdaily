import pandas as pd
import re
from difflib import SequenceMatcher
from typing import List, Dict, Any, Tuple
from src.config import Config
from src.llm_client import LLMClient

# Text Normalization Helpers
URL_RX = re.compile(r"https?://\S+|t\.me/\S+", re.I)
WS_RX = re.compile(r"\s+")
NON_ALNUM_RX = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")

def normalize_text(s: str) -> str:
    """Aggressive normalization for text deduplication."""
    if not s:
        return ""
    s = str(s).lower()
    s = URL_RX.sub(" ", s)
    s = NON_ALNUM_RX.sub(" ", s)
    s = WS_RX.sub(" ", s).strip()
    return s

def text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def jaccard_similarity(a: str, b: str) -> float:
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb: 
        return 0.0
    return len(sa & sb) / len(sa | sb)

class Deduplicator:
    def __init__(self, config: Config):
        self.config = config
        self.llm = LLMClient(config.secrets)

    def dedup_by_text(
        self, 
        df: pd.DataFrame, 
        sim_threshold: float = 0.90, 
        jaccard_min: float = 0.40
    ) -> pd.DataFrame:
        """
        Fast pairwise deduplication using text similarity.
        O(N^2) within buckets (country/channel).
        """
        if df.empty:
            return df
            
        df = df.copy()
        df["_norm"] = df["original"].fillna("").map(normalize_text)
        
        # Bucket by country to speed up
        buckets = df.groupby("country")
        drop_indices = set()
        
        for _, group in buckets:
            idxs = group.index.tolist()
            n = len(idxs)
            if n < 2: 
                continue
                
            for i in range(n):
                idx_a = idxs[i]
                if idx_a in drop_indices: 
                    continue
                
                txt_a = df.loc[idx_a, "_norm"]
                
                for j in range(i + 1, n):
                    idx_b = idxs[j]
                    if idx_b in drop_indices: 
                        continue
                        
                    txt_b = df.loc[idx_b, "_norm"]
                    
                    sim = text_similarity(txt_a, txt_b)
                    jac = jaccard_similarity(txt_a, txt_b)
                    
                    if sim >= sim_threshold and jac >= jaccard_min:
                        # Drop the older one (assuming sorted by date desc? No, we sorted asc in fetcher)
                        # Actually fetcher sorts ASC returning oldest first.
                        # So idx_b is newer than idx_a.
                        # We usually want to keep the LATEST one?
                        # Original script kept older (first seen) or switchable. 
                        # Let's keep the NEWEST (idx_b) and drop idx_a? 
                        # Or keep idx_a and drop idx_b (dedup moving forward).
                        # Let's keep the EARLIER one (idx_a) to be stable, unless updated.
                        # Actually original script said "Text dedupe (cheap baseline, keeps history)".
                        # Let's drop idx_b (the later duplicate).
                        drop_indices.add(idx_b)
        
        return df.drop(index=list(drop_indices)).drop(columns=["_norm"], errors="ignore")

    def dedup_by_llm(
        self, 
        df: pd.DataFrame, 
        prefer_local_llm: bool = False
    ) -> pd.DataFrame:
        """
        Semantic deduplication using LLM.
        Compares each row against recent KEPT rows in same bucket.
        """
        if df.empty:
            return df
            
        # Ensure sorted by date ascending (oldest first)
        df = df.sort_values("datetime", ascending=True).reset_index(drop=True)
        
        kept_rows = []
        
        # Group by bucket (country)
        # We process row by row to build up the "kept" history
        # but we can do it per-bucket to separate contexts
        
        # To avoid complex groupby iteration that loses global order or index,
        # let's just iterate all, but maintain a localized buffer for each country.
        
        buffer_by_country = {} # country -> list of recent kept messages
        
        for _, row in df.iterrows():
            country = row.get("country", "unknown")
            text = row.get("english") or row.get("original") or ""
            
            # Get recent candidates
            candidates = buffer_by_country.get(country, [])
            
            # Compare against last N candidates (e.g. 15)
            is_dup = False
            # Check most recent first
            for cand in reversed(candidates[-15:]):
                cand_text = cand.get("english") or cand.get("original") or ""
                
                # PRE-FILTER: If text similarity is very low, skip LLM
                # This drastically reduces calls if the messages are clearly different
                if text_similarity(normalize_text(text), normalize_text(cand_text)) < 0.2:
                    continue

                prompt = f"""
Compare these two Telegram messages.
Do they describe the SAME underlying news event?
(Same actor, location, date, action).
Dup if just rewording or reposting.
Not dup if update or different event.

Message A: {cand_text}
Message B: {text}

Return JSON: {{"decision": "DUP" or "NOT_DUP", "confidence": float}}
""".strip()

                res = self.llm.deduplicate_pair(
                    prompt=prompt,
                    prefer_local=prefer_local_llm
                )
                
                if res.get("decision") == "DUP" and res.get("confidence", 0.0) > 0.6:
                    is_dup = True
                    print(f"  (llm-dup) Dropping {row['channel']}/{row['message_id']} (dup of {cand['channel']}/{cand['message_id']})")
                    break
            
            if not is_dup:
                kept_rows.append(row)
                buffer_by_country.setdefault(country, []).append(row)
                
        return pd.DataFrame(kept_rows)
