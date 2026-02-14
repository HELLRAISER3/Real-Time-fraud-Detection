from logger.log import logging
import os
from typing import Optional


def alert_model_failure(message: str):
    logging.error(f"MODEL FAILURE: {message}")
    _send_email(subject="Model Failure Alert", message=message)


def alert_high_error_rate(error_rate: float):
    """Alert when error rate is too high"""
    logging.warning(f"HIGH ERROR RATE: {error_rate:.1f}%")
    _send_email(
        subject="High Error Rate Alert",
        message=f"Error rate is {error_rate:.1f}% (threshold: 5%)"
    )


def _send_email(subject: str, message: str):
    if not os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
        return
    try:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg.set_content(message)
        msg['Subject'] = subject
        msg['From'] = os.getenv("ALERT_EMAIL_FROM", "alerts@fraud-detection.local")
        msg['To'] = os.getenv("ALERT_EMAIL_TO", "admin@fraud-detection.local")

        smtp_host = os.getenv("ALERT_EMAIL_SMTP_HOST", "localhost")
        smtp_port = int(os.getenv("ALERT_EMAIL_SMTP_PORT", "587"))
        smtp_password = os.getenv("ALERT_EMAIL_PASSWORD", "")

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            if smtp_password:
                server.login(msg['From'], smtp_password)
            server.send_message(msg)

        logging.info(f"Email alert sent: {subject}")
    except Exception as e:
        logging.error(f"Failed to send email alert: {e}")
