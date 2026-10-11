from __future__ import annotations

import os
import smtplib
from email.message import EmailMessage
from datetime import datetime, timezone

from .config import get_setting


def email_mode() -> str:
    return (get_setting("EMAIL_DELIVERY_MODE", "mock") or "mock").lower()


def send_email(to_email: str, subject: str, body: str) -> tuple[bool, str]:
    if email_mode() != "smtp":
        return True, "Mock send completed. No external email was sent."

    required = ["SMTP_HOST", "SMTP_PORT", "SMTP_USERNAME", "SMTP_PASSWORD", "SMTP_FROM"]
    values = {k: get_setting(k) for k in required}
    missing = [k for k, v in values.items() if not v]
    if missing:
        return False, "Missing SMTP settings: " + ", ".join(missing)

    msg = EmailMessage()
    msg["From"] = values["SMTP_FROM"]
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)
    try:
        with smtplib.SMTP(values["SMTP_HOST"], int(values["SMTP_PORT"]), timeout=20) as smtp:
            smtp.starttls()
            smtp.login(values["SMTP_USERNAME"], values["SMTP_PASSWORD"])
            smtp.send_message(msg)
        return True, f"Email sent at {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    except Exception as exc:
        return False, f"Email send failed: {exc}"
