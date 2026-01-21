import sys
import re
from difflib import SequenceMatcher
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm

APPROVED = {
    "Russia", "Kazakhstan", "Kyrgyz Republic", "Uzbekistan", "Tajikistan",
    "Turkmenistan", "Azerbaijan", "Georgia", "Armenia", "Ukraine", "Belarus"
}

def normalize_country(c):
    if not isinstance(c, str):
        return "unknown"
    c2 = c.strip().lower()

    mapping = {
        r"russia|rf|ru|россия": "Russia",
        r"kazakh.?|kazakhstan|kz": "Kazakhstan",
        r"kyrgyz.?|kirghiz.?|kg": "Kyrgyz Republic",
        r"uzbek.?|uzbekistan|uz": "Uzbekistan",
        r"tajik.?|tj": "Tajikistan",
        r"turkmen.?|tm": "Turkmenistan",
        r"azerbai.?|az": "Azerbaijan",
        r"georgia|sakartvelo|geo": "Georgia",
        r"armenia|am": "Armenia",
        r"ukraine|ukr|ua": "Ukraine",
        r"belarus|by": "Belarus",
    }

    for pattern, val in mapping.items():
        if re.search(pattern, c2):
            return val

    return "unknown"


def is_similar(a, b, threshold=0.90):
    return SequenceMatcher(None, a, b).ratio() >= threshold


def remove_duplicates(df):
    seen = []
    rows = []

    for _, r in df.iterrows():
        text = str(r["english"]).strip()
        if not text:
            continue
        if any(text == s or is_similar(text, s) for s in seen):
            continue

        seen.append(text)
        rows.append(r)

    return pd.DataFrame(rows)


def generate_pdf(input_csv, output_pdf):
    df = pd.read_csv(input_csv)

    # Filter to last 24 hours
    if "datetime" in df.columns:
        df["ts"] = pd.to_datetime(df["datetime"], errors="coerce")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=1)
        df = df[df["ts"] >= cutoff]
    else:
        print("WARNING: no datetime column in CSV")

    # Normalize country
    if "country" not in df.columns:
        df["country"] = "unknown"
    df["country"] = df["country"].apply(normalize_country)

    # Ensure english exists
    if "english" not in df.columns:
        df["english"] = ""

    # Remove duplicates
    df = remove_duplicates(df)

    # Keep only approved countries
    df = df[df["country"].isin(APPROVED)]

    # Parse datetime into date + time
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%Y-%m-%d")
        df["time"] = pd.to_datetime(df["datetime"], errors="coerce").dt.strftime("%H:%M")
    else:
        df["date"] = ""
        df["time"] = ""

    styles = getSampleStyleSheet()
    story = []

    # Title
    story.append(Paragraph("<b>Macro-Advisory Daily Telegram Summary</b>", styles["Title"]))
    story.append(Spacer(1, 20))

    # Sort countries alphabetically
    sorted_countries = sorted(APPROVED)
    first_country = True

    for country in sorted_countries:
        subset = df[df["country"] == country]
        if subset.empty:
            continue

        # Page breaks between countries
        if not first_country:
            story.append(PageBreak())
        first_country = False

        story.append(Paragraph(f"<b>{country}</b>", styles["Heading2"]))
        story.append(Spacer(1, 14))

        for _, row in subset.iterrows():
            english = row.get("english", "")
            dt = row.get("datetime", "")
            channel = row.get("channel", "").lstrip("@")
            msg_id = row.get("message_id", "")
            url = row.get("url", "")

            # Parse datetime
            try:
                ts = pd.to_datetime(dt)
                date_str = ts.strftime("%Y-%m-%d")
                time_str = ts.strftime("%H:%M")
            except:
                date_str = ""
                time_str = ""

            # URL fallback
            if not isinstance(url, str) or not url.startswith("http"):
                if channel and msg_id:
                    url = f"https://t.me/{channel}/{msg_id}"
                else:
                    url = ""

            # Article text first
            story.append(Paragraph(english, styles["BodyText"]))
            story.append(Spacer(1, 6))

            # Metadata after the article
            meta = []
            if date_str or time_str:
                meta.append(f"{date_str} {time_str}")
            if channel:
                meta.append(f"@{channel}")
            if url:
                meta.append(f'<a href="{url}">{url}</a>')

            meta_html = " — ".join(meta)
            story.append(Paragraph(f'<font size="9" color="grey">{meta_html}</font>', styles["BodyText"]))
            story.append(Spacer(1, 16))

    # Build PDF
    doc = SimpleDocTemplate(output_pdf, pagesize=A4)
    doc.build(story)

    print("PDF saved:", output_pdf)



# MAIN --------------------------------------------------------------------
if __name__ == "__main__":
    csv_in = sys.argv[1]
    pdf_out = sys.argv[2]
    generate_pdf(csv_in, pdf_out)
    