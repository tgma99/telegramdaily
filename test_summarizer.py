#!/usr/bin/env python3
import json
import pandas as pd
from pathlib import Path
from openai import OpenAI

# -------------------------------
# Load config and OpenAI key
# -------------------------------
ROOT = Path(__file__).resolve().parent
cfg_path = ROOT / "config" / "secrets.json"
cfg = json.loads(cfg_path.read_text())

client = OpenAI(api_key=cfg["openai"]["api_key"])

# -------------------------------
# Define TARGET COUNTRIES
# -------------------------------
TARGET_COUNTRIES = {
    "russia",
    "belarus",
    "kazakhstan",
    "kyrgyz republic",
    "tajikistan",
    "turkmenistan",
    "mongolia",
    "armenia",
    "azerbaijan",
    "georgia",
    "ukraine",
}

# -------------------------------
# Country normaliser for testing
# -------------------------------
def get_country(raw):
    """
    Test version of your real normaliser.
    """
    if not isinstance(raw, str):
        return None

    parts = [p.strip().lower() for p in raw.split(",")]

    for p in parts:
        if p in TARGET_COUNTRIES:
            return p
    return None


# -------------------------------
# Test DataFrame (fake LM output)
# -------------------------------
df = pd.DataFrame([
    {
        "country": "Armenia",
        "english": "Prime Minister met US officials to discuss regional security.",
        "url": "https://t.me/test/111",
    },
    {
        "country": "Armenia,USA",   # should normalise to 'armenia'
        "english": "Inflation decreased to 2.3%, improving investment outlook.",
        "url": "https://t.me/test/112",
    },
    {
        "country": "Germany",       # should be ignored
        "english": "Factory orders declined again.",
        "url": "https://t.me/test/113",
    },
    {
        "country": "Azerbaijan",
        "english": "Peace negotiations with Armenia show progress.",
        "url": "https://t.me/test/200",
    },
])

# Build fake LM JSON structure
df["lm"] = df.apply(lambda row: {
    "country": row["country"],
    "english": row["english"]
}, axis=1)

# Normalise country names
df["country_norm"] = df["lm"].apply(lambda lm: get_country(lm["country"]))

# Keep only target countries
df = df[df["country_norm"].notnull()].copy()

print("\nCountries detected:", df["country_norm"].unique().tolist())


# -------------------------------
# Helper: attach URLs to summary lines
# -------------------------------
def attach_urls_to_summary(summary_text, items):
    """
    Take arbitrary model output (lines) and:
    - turn each non-empty line into a bullet (if needed)
    - append one URL per bullet from items (in order)
    """
    lines_out = []
    remaining_items = list(items)  # shallow copy

    for line in summary_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue

        # If it doesn't look like a bullet, make it one
        if not stripped[0] in "-•*":
            stripped = "- " + stripped

        # Attach URL if we still have one
        if remaining_items:
            url = remaining_items.pop(0)["url"]
            if url:
                stripped = stripped + f" ({url})"

        lines_out.append(stripped)

    # Fallback: if model returned nothing useful, just use original items
    if not lines_out and items:
        for it in items:
            lines_out.append(f"- {it['text']} ({it['url']})")

    return "\n".join(lines_out)


# -------------------------------
# Summarisation Function (test)
# -------------------------------
def summarize_country_for_test(country, sub_df):
    """
    Test version of your summariser:
    - calls OpenAI
    - prints RAW summary
    - returns processed bullet list with URLs
    """
    items = []
    for _, row in sub_df.iterrows():
        items.append({"text": row["english"], "url": row["url"]})

    text_block = "\n".join(f"- {item['text']}" for item in items)

    prompt = f"""
Summarize the key news about {country}.
Write 3–5 short bullet points.
Each point should be on its own line.
Do NOT include URLs or sources.

News items:
{text_block}
"""

    resp = client.chat.completions.create(
        model="o3-mini",
        messages=[{"role": "user", "content": prompt}],
    )

    summary_text = resp.choices[0].message.content

    print(f"\nRAW SUMMARY FOR {country.upper()}:\n{summary_text}\n")

    processed = attach_urls_to_summary(summary_text, items)
    return processed


# -------------------------------
# Run the test
# -------------------------------
print("\n=== TEST OUTPUT ===\n")

for c in df["country_norm"].unique():
    sub = df[df["country_norm"] == c]

    header = f"**{c.upper()}**"
    print(header)

    summary = summarize_country_for_test(c, sub)
    print(summary)
    print()