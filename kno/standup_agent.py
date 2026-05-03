"""
Daily Standup Agent for kno.ai
-------------------------------
Aggregates Jira, Gmail, and Confluence activity from the last 24 hours and
formats it into a concise morning standup summary.

Usage:
    python -m kno.standup_agent --now                        # run once, print to terminal
    python -m kno.standup_agent --now --email me@example.com # run once + send email
    python -m kno.standup_agent --hour 8 --minute 30 --email me@example.com  # schedule
"""

import argparse
import asyncio
import base64
import datetime
import os
import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import google.auth
import schedule
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types
from googleapiclient.discovery import build

from kno.atlassian_connector import (
    get_jira_issue,
    get_confluence_page,
    search_confluence_pages,
    search_jira_issues,
)
from kno.agent import search_gmail  # reuse the tool defined in the main agent

_GMAIL_SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.send",
]

def _gmail_send_service():
    creds, _ = google.auth.default(scopes=_GMAIL_SCOPES)
    return build("gmail", "v1", credentials=creds)


# ── Agent definition ──────────────────────────────────────────────────────────

STANDUP_INSTRUCTIONS = """
You are a daily standup assistant for a software engineering team.

When asked for a standup summary, you MUST call these tools in order:

1. search_jira_issues("status = \\"In Progress\\" ORDER BY updated DESC")
   → lists every ticket currently in progress

2. search_jira_issues("status = \\"Done\\" AND updated >= -1d ORDER BY updated DESC")
   → lists tickets completed in the last 24 hours

3. search_gmail("is:unread newer_than:1d")
   → fetches unread emails from the last 24 hours

4. search_confluence_pages("lastModified >= now(-1d)")
   → finds Confluence pages updated yesterday

After calling all four tools, format a standup briefing with exactly these sections:

✅ DONE YESTERDAY
  • List each completed ticket as: KEY: Summary
  • If none, write "Nothing completed yet — keep pushing! 💪"

🔄 IN PROGRESS
  • List each in-progress ticket as: KEY: Summary
  • If a ticket has been In Progress for more than 2 days, flag it with ⚠️

📧 EMAILS TO ACTION
  • Summarise each unread email in one line: Sender → subject/topic
  • If none, write "Inbox zero! 🎉"

📋 CONFLUENCE UPDATES
  • List any pages updated yesterday as: Page Title (Space)
  • If none, write "No doc updates."

🚧 BLOCKERS / NEEDS ATTENTION
  • Highlight any In Progress tickets with no recent activity (>2 days unchanged)
  • Mention any emails that look urgent or require a reply
  • If nothing stands out, write "No blockers identified."

Keep the tone friendly and concise — this is a morning briefing, not a report.
Start with: "Good morning! Here's your standup for [today's date]:"
""".strip()

standup_agent = Agent(
    name="standup_agent",
    model="gemini-2.5-flash",
    description="Generates a daily standup summary from Jira, Gmail, and Confluence.",
    instruction=STANDUP_INSTRUCTIONS,
    tools=[
        search_jira_issues,
        get_jira_issue,
        search_gmail,
        search_confluence_pages,
        get_confluence_page,
    ],
)


# ── Email delivery ───────────────────────────────────────────────────────────

# Map emoji section headers → background colour for the HTML email
_SECTION_COLOURS = {
    "✅": "#d4edda",   # green  — Done Yesterday
    "🔄": "#d1ecf1",   # blue   — In Progress
    "📧": "#fff3cd",   # yellow — Emails to Action
    "📋": "#e2e3e5",   # grey   — Confluence Updates
    "🚧": "#f8d7da",   # red    — Blockers
}


def _text_to_html(text: str, today: str) -> str:
    """Convert the plain-text standup summary into a styled HTML email."""
    lines = text.splitlines()
    html_lines = [
        "<html><body style='font-family:Arial,sans-serif;max-width:680px;"
        "margin:auto;padding:24px;color:#333'>",
        f"<h2 style='color:#1a73e8'>🌅 Daily Standup — {today}</h2>",
    ]

    current_colour = "#ffffff"
    in_section = False

    for line in lines:
        # Detect section header lines (start with an emoji key)
        header_emoji = next((e for e in _SECTION_COLOURS if line.startswith(e)), None)
        if header_emoji:
            if in_section:
                html_lines.append("</div>")
            current_colour = _SECTION_COLOURS[header_emoji]
            safe = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(
                f"<div style='background:{current_colour};border-radius:8px;"
                f"padding:12px 16px;margin:12px 0'>"
                f"<strong style='font-size:15px'>{safe}</strong>"
            )
            in_section = True
        elif line.strip().startswith("•"):
            safe = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f"<div style='padding:2px 0 2px 8px'>{safe}</div>")
        elif line.strip().startswith("Good morning"):
            safe = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f"<p style='font-size:15px;color:#555'>{safe}</p>")
        elif line.strip():
            safe = line.replace("<", "&lt;").replace(">", "&gt;")
            html_lines.append(f"<div style='padding:2px 0 2px 8px;color:#555'>{safe}</div>")

    if in_section:
        html_lines.append("</div>")

    html_lines.append(
        "<p style='font-size:12px;color:#999;margin-top:24px;border-top:1px solid #eee;"
        "padding-top:12px'>Sent by kno.ai standup agent</p>"
        "</body></html>"
    )
    return "\n".join(html_lines)


def send_standup_email(summary_text: str, to_email: str, today: str) -> None:
    """Send the standup summary as a styled HTML email via the Gmail API."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🌅 Daily Standup — {today}"
    msg["From"]    = "me"
    msg["To"]      = to_email

    msg.attach(MIMEText(summary_text, "plain"))
    msg.attach(MIMEText(_text_to_html(summary_text, today), "html"))

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    try:
        service = _gmail_send_service()
        service.users().messages().send(userId="me", body={"raw": raw}).execute()
        print(f"📨  Standup emailed to {to_email}")
    except Exception as e:
        print(f"⚠️  Email send failed: {e}")


# ── Runner ────────────────────────────────────────────────────────────────────

async def _run_standup_async(to_email: str | None = None, max_retries: int = 3) -> None:
    """Async core of the standup run, with exponential-backoff retry on 429."""
    today = datetime.date.today().strftime("%A, %B %-d %Y")
    print(f"\n{'─' * 60}")
    print(f"  🌅  Daily Standup  —  {today}")
    print(f"{'─' * 60}\n")

    runner = InMemoryRunner(agent=standup_agent, app_name="standup")
    session = await runner.session_service.create_session(
        app_name="standup",
        user_id="user",
    )

    message = (
        f"Generate my daily standup summary for today ({today}). "
        "Call all four tools (Jira in-progress, Jira done, Gmail unread, Confluence updates) "
        "before writing the summary."
    )

    for attempt in range(1, max_retries + 1):
        try:
            response_text = ""
            async for event in runner.run_async(
                user_id="user",
                session_id=session.id,
                new_message=genai_types.Content(
                    role="user",
                    parts=[genai_types.Part(text=message)],
                ),
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    response_text = "".join(
                        p.text for p in event.content.parts if hasattr(p, "text")
                    )

            print(response_text or "⚠️  No response received from the standup agent.")
            print(f"\n{'─' * 60}\n")

            # Send email if requested
            if response_text and to_email:
                send_standup_email(response_text, to_email, today)

            return  # success — exit retry loop

        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if is_rate_limit and attempt < max_retries:
                wait = 30 * attempt  # 30s → 60s → 90s
                print(f"⚠️  Rate limit hit (attempt {attempt}/{max_retries}). "
                      f"Retrying in {wait}s…")
                await asyncio.sleep(wait)
                session = await runner.session_service.create_session(
                    app_name="standup",
                    user_id="user",
                )
            else:
                print(f"❌  Standup failed after {attempt} attempt(s): {e}")
                return


def run_standup(to_email: str | None = None) -> None:
    """Run the standup agent once, print to terminal, and optionally email the summary."""
    asyncio.run(_run_standup_async(to_email=to_email))


# ── Scheduler ─────────────────────────────────────────────────────────────────

def schedule_standup(hour: int = 9, minute: int = 0, to_email: str | None = None) -> None:
    """Schedule run_standup() to fire every day at hour:minute (24-hour clock)."""
    time_str = f"{hour:02d}:{minute:02d}"
    schedule.every().day.at(time_str).do(run_standup, to_email=to_email)
    dest = f" → emailing {to_email}" if to_email else " → terminal only"
    print(f"⏰  Standup scheduled daily at {time_str}{dest}. Waiting…  (Ctrl-C to stop)\n")
    try:
        while True:
            schedule.run_pending()
            time.sleep(30)
    except KeyboardInterrupt:
        print("\nScheduler stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="kno.ai Daily Standup Agent")
    parser.add_argument(
        "--now",
        action="store_true",
        help="Run the standup once immediately and exit (skip scheduling).",
    )
    parser.add_argument(
        "--hour",   type=int, default=9,  help="Hour to run the daily standup (default: 9)"
    )
    parser.add_argument(
        "--minute", type=int, default=0,  help="Minute to run the daily standup (default: 0)"
    )
    parser.add_argument(
        "--email",
        type=str,
        default=None,
        metavar="ADDRESS",
        help="Email address to send the standup to (in addition to terminal output).",
    )
    args = parser.parse_args()

    # Always run immediately on startup so you get a standup right away
    run_standup(to_email=args.email)

    if not args.now:
        schedule_standup(hour=args.hour, minute=args.minute, to_email=args.email)
