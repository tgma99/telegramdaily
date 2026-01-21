#!/usr/bin/env python3
"""
Test harness for Summarize_and_Email.py

- Creates a dummy filtered_YYYY-MM-DD.csv in data/filtered
- Mocks OpenAI chat completion and send_email
- Runs main() and prints the email body
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd

import Summarize_and_Email as SAE  # make sure this is in the same directory or on PYTHONPATH


def create_dummy_csv():
    """Create a small dummy filtered_YYYY-MM-DD.csv to drive the test."""
    today_str = datetime.now().strftime("%Y-%m-%d")
    csv_name = f"filtered_{today_str}.csv"
    csv_path = SAE.INPUT_DIR / csv_name

    SAE.INPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "datetime": f"{today_str} 08:30",
            "channel": "@russian_news",
            "url": "https://example.com/russia1",
            "original": "О компании АгроТерра и сельском хозяйстве.",
            "english": "Discussion of Agroterra and agriculture in Russia.",
            "raw": "Agroterra in Russia news",
            "lm_json": json.dumps({
                "skip": False,
                "country": "Russia",
                "english": "Agroterra in Russia news"
            }),
        },
        {
            "datetime": f"{today_str} 09:15",
            "channel": "@kazakh_news",
            "url": "https://example.com/kazakhstan1",
            "original": "GM инвестирует в Казахстан.",
            "english": "GM invests in Kazakhstan car plant.",
            "raw": "GM invests in Kazakhstan",
            "lm_json": json.dumps({
                "skip": False,
                "country": "Kazakhstan",
                "english": "GM invests in Kazakhstan"
            }),
        },
        {
            "datetime": f"{today_str} 10:00",
            "channel": "@ukraine_news",
            "url": "https://example.com/ukraine1",
            "original": "Макроэкономический обзор Украины.",
            "english": "Macroeconomic overview of Ukraine.",
            "raw": "Macroeconomic overview of Ukraine",
            "lm_json": json.dumps({
                "skip": False,
                "country": "Ukraine",
                "english": "Macroeconomic overview of Ukraine"
            }),
        },
        {
            # This one should be skipped by the LM flag
            "datetime": f"{today_str} 11:00",
            "channel": "@misc_channel",
            "url": "https://example.com/skipme",
            "original": "Нерелевантная новость.",
            "english": "Irrelevant news.",
            "raw": "Irrelevant news",
            "lm_json": json.dumps({
                "skip": True,
                "country": "Russia",
                "english": "Irrelevant news"
            }),
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    print(f"✅ Dummy CSV written to: {csv_path}")


def mock_openai():
    """Mock SAE.client.chat.completions.create to avoid real OpenAI calls."""

    def fake_create(*args, **kwargs):
        # You can inspect kwargs["messages"][0]["content"] if you want
        class Msg:
            pass

        class Choice:
            pass

        class Resp:
            pass

        msg = Msg()
        # Simple fixed dummy content – enough to exercise attach_urls_to_summary
        msg.content = "• Dummy bullet 1\n• Dummy bullet 2\n• Dummy bullet 3"

        choice = Choice()
        choice.message = msg

        resp = Resp()
        resp.choices = [choice]
        return resp

    # Overwrite the method used in summarize_country
    SAE.client.chat.completions.create = fake_create
    print("✅ Mocked OpenAI client.chat.completions.create")


def mock_send_email():
    """Mock SAE.send_email so no real email is sent."""

    def fake_send_email(subject, body):
        print("\n========== FAKE send_email CALLED ==========")
        print("Subject:", subject)
        print("Body preview (first 2000 chars):")
        print(body[:2000])
        print("============== END FAKE EMAIL ==============\n")

    SAE.send_email = fake_send_email
    print("✅ Mocked send_email")


def main():
    # Ensure dummy input exists
    create_dummy_csv()

    # Patch OpenAI and email
    mock_openai()
    mock_send_email()

    # Run the real main() from Summarize_and_Email
    print("🚀 Running Summarize_and_Email.main() with dummy data...")
    SAE.main()


if __name__ == "__main__":
    main()