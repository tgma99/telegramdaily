import os
import sys
import smtplib
from email.message import EmailMessage

def send_pdf(pdf_path):

    if not os.path.isfile(pdf_path):
        print(f"ERROR: PDF does not exist: {pdf_path}")
        sys.exit(1)

    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    email_user = os.getenv("EMAIL_USER")
    email_pass = os.getenv("EMAIL_PASS")
    email_to = os.getenv("EMAIL_TO")

    if not all([smtp_server, smtp_port, email_user, email_pass, email_to]):
        print("ERROR: Missing email environment variables.")
        sys.exit(1)

    msg = EmailMessage()
    msg["Subject"] = "Macro-Advisory Daily Telegram Summary"
    msg["From"] = email_user
    msg["To"] = email_to
    msg.set_content("Daily Telegram summary attached.\n")

    with open(pdf_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="application",
            subtype="pdf",
            filename=os.path.basename(pdf_path),
        )

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(email_user, email_pass)
            server.send_message(msg)
            print("Email sent successfully.")
    except Exception as e:
        print("ERROR sending email:")
        print(e)
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python email_pdf.py /path/to/file.pdf")
        sys.exit(1)

    send_pdf(sys.argv[1])
    