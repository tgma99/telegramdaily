import os
import sys
import json
import random
import pandas as pd
import requests
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# ============================
# CONFIG
# ============================

LM_URL = "http://localhost:1234/v1/chat/completions"
MODEL = "gpt-4-turbo"  # or your LM Studio model
TOP_N = 30

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "filtered_translated_macro.csv"
OUTPUT_PDF = f"Filtered_Laws_Appointments_{datetime.now().strftime('%y%m%d')}.pdf"


# ============================
# SCORING PROMPT
# ============================

SYSTEM_PROMPT = """
You are scoring Telegram news items.

For each item, give a score from 0 to 10 based on political/legal significance.

Scoring rules:

+2 law, decree, bill, regulation, court decision
+2 president / PM / minister / parliament / senior officials
+2 appointments, dismissals, resignations, elections
+2 policy changes, reforms, strategies, government programs
+1 agencies, ministries, regulators, state-owned companies
+1 scandals, investigations, corruption, arrests of officials
+1 hints of future political or policy changes

Score MUST be an integer 0–10.

Output JSON:
{
  "score": <integer>,
  "short_reason": "why",
  "clean_english": "one-sentence summary"
}
"""


# ============================
# LLM QUERY FUNCTION
# ============================

def query_lmstudio(prompt_text: str):
    """Send prompt to local LM Studio model."""
    temperature_value = round(random.uniform(0.25, 0.45), 2)

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        "temperature": temperature_value,
        "max_tokens": 200
    }

    r = requests.post(LM_URL, json=payload)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ============================
# SCORING LOGIC
# ============================

def score_messages(df):
    results = []

    for i, row in df.iterrows():
        raw_text = str(row.get("english", row.get("message_text", ""))).strip()
        if not raw_text:
            continue

        try:
            raw_reply = query_lmstudio(raw_text)
            data = json.loads(raw_reply)

            # Ensure score exists and is valid
            score = int(data.get("score", 0))

        except Exception as e:
            print(f"[!] Row {i}: JSON parse error → default score 0")
            print(f"    Raw output: {raw_reply if 'raw_reply' in locals() else 'NO OUTPUT'}")
            score = 0
            data = {"clean_english": raw_text, "short_reason": "LLM parse error"}

        results.append({
            "score": score,
            "clean_english": data.get("clean_english", raw_text),
            "reason": data.get("short_reason", "No reason given"),
            "source_channel": row.get("channel", ""),
            "timestamp": row.get("date", "")
        })

    return pd.DataFrame(results)


# ============================
# PDF CREATION
# ============================

def save_pdf(df, filename):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(filename, pagesize=A4)
    story = []

    story.append(Paragraph("<b>Ranked Political / Legal News</b>", styles["Title"]))
    story.append(Spacer(1, 16))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles["Normal"]))
    story.append(Spacer(1, 24))

    for _, row in df.iterrows():
        story.append(Paragraph(f"<b>[Score {row['score']}]</b> {row['timestamp']} — {row['source_channel']}", styles["Heading4"]))
        story.append(Paragraph(row["clean_english"], styles["BodyText"]))
        story.append(Paragraph(f"<i>{row['reason']}</i>", styles["Italic"]))
        story.append(Spacer(1, 12))

    doc.build(story)
    print(f"📄 Saved PDF: {filename}")


# ============================
# MAIN
# ============================

if __name__ == "__main__":
    print(f"Loading {INPUT_FILE} …")
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df)} messages → Scoring…")
    scored_df = score_messages(df)

    if scored_df.empty:
        print("⚠️ No rows scored (LLM failure) → generating fallback PDF")
        fallback = pd.DataFrame([{
            "score": 0,
            "clean_english": "No political or legal items identified.",
            "reason": "",
            "source_channel": "",
            "timestamp": ""
        }])
        save_pdf(fallback, OUTPUT_PDF)
        sys.exit(0)

    print("Sorting and keeping top messages…")
    top_df = scored_df.sort_values(by="score", ascending=False).head(TOP_N)

    save_pdf(top_df, OUTPUT_PDF)

    print("Done.")