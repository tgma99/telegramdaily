import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import datetime, timezone
import pandas as pd
from typing import List, Dict, Any, Optional
from src.config import Config

# HTML Styles
HTML_STYLE = """
<style>
  .t-body { font-family: Arial, sans-serif; font-size: 10.5pt; line-height: 120%; letter-spacing: .1pt; color: #76777B; }
  .t-accent { font-weight: 700; color: #9F7E56; }
  .t-p { margin: 0 0 12px 0; }
  .t-hr { margin: 18px 0 0 0; border-top: 1px solid #E1E1E1; padding-top: 10px; }
  .t-h2 { margin: 16px 0 8px 0; font-size: 12pt; font-weight: 700; color: #2B2B2B; }
  .t-muted { color: #9AA0A6; }
  .t-li { margin: 0 0 10px 0; }
  a { color: #6B6F76; text-decoration: underline; }
</style>
"""

class Summarizer:
    def __init__(self, config: Config):
        self.config = config

    def render_html(self, df: pd.DataFrame, subject_line: str) -> str:
        if df.empty:
             return "<html><body><p>No items today.</p></body></html>"

        # Sort by Country then Time
        df["country"] = df["country"].fillna("Other").str.strip().str.title()
        df.loc[df["country"].str.lower() == "unknown", "country"] = "Other"
        
        # Create explicit sort key: "Other" goes last (z prefix), others sort normally
        df["_sort"] = df["country"].apply(lambda c: "zzzz" if c == "Other" else c)
        
        df = df.sort_values(["_sort", "datetime"], ascending=[True, False])
        
        parts = []
        parts.append(f"<html><head><meta charset='utf-8'>{HTML_STYLE}</head><body class='t-body'>")
        parts.append(f"<div style='width:680px; padding:18px;'>")
        
        # Header
        parts.append(f"<h1 class='t-h2'>{subject_line}</h1>")
        parts.append(f"<p class='t-muted'>{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}</p>")
        parts.append("<div class='t-hr'></div>")

        # Group by Country
        for country, group in df.groupby("country", sort=False):
            parts.append(f"<div class='t-h2'>{country.upper()}</div>")
            
            # Limit items per country (e.g. 5)
            for _, row in group.head(5).iterrows():
                headline = row.get("subject", "") or "Update"
                headline = headline.strip().capitalize()
                summary = row.get("english", "") or row.get("original", "")
                # Truncate summary if too long for "why" style
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                
                url = row.get("url", "#")
                
                parts.append(f"<p class='t-p t-li'>")
                parts.append(f"<span class='t-accent'>{headline}</span>: ")
                parts.append(f"{summary} ")
                parts.append(f"(<a href='{url}'>link</a>)")
                parts.append(f"</p>")
            
            parts.append("<div class='t-hr'></div>")
            
        parts.append("</div></body></html>")
        return "\n".join(parts)

    def send_email(self, html_body: str, subject: str) -> None:
        cfg = self.config.secrets.email_config
        if not cfg:
            print("⚠️ Email config missing, skipping send.")
            return

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = f"{cfg.get('from', 'Telegram Daily')} <{cfg.get('username')}>"
        msg["To"] = ", ".join(cfg.get("recipients", []))
        
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        try:
            with smtplib.SMTP(cfg["smtp_server"], int(cfg["smtp_port"])) as server:
                server.starttls()
                server.login(cfg["username"], cfg["password"])
                server.sendmail(cfg["username"], cfg["recipients"], msg.as_string())
            print(f"✅ Email sent: {subject}")
        except Exception as e:
            print(f"❌ Failed to send email: {e}")
