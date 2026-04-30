"""
Daily Standup Agent for kno.ai
-------------------------------
Aggregates Jira, Gmail, and Confluence activity from the last 24 hours and
formats it into a concise morning standup summary.

Usage:
    python -m kno.standup_agent          # run once immediately, then schedule at 9am
    python -m kno.standup_agent --now    # run once and exit (useful for testing)
"""

import argparse
import asyncio
import datetime
import time

import schedule
from google.adk.agents.llm_agent import Agent
from google.adk.runners import InMemoryRunner
from google.genai import types as genai_types

from kno.atlassian_connector import (
    get_jira_issue,
    get_confluence_page,
    search_confluence_pages,
    search_jira_issues,
)
from kno.agent import search_gmail  # reuse the tool defined in the main agent


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


# ── Runner ────────────────────────────────────────────────────────────────────

async def _run_standup_async() -> None:
    """Async core of the standup run."""
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


def run_standup() -> None:
    """Run the standup agent once and print the summary to the terminal."""
    asyncio.run(_run_standup_async())


# ── Scheduler ─────────────────────────────────────────────────────────────────

def schedule_standup(hour: int = 9, minute: int = 0) -> None:
    """Schedule run_standup() to fire every day at hour:minute (24-hour clock)."""
    time_str = f"{hour:02d}:{minute:02d}"
    schedule.every().day.at(time_str).do(run_standup)
    print(f"⏰  Standup scheduled daily at {time_str}. Waiting…  (Ctrl-C to stop)\n")
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
    args = parser.parse_args()

    # Always run immediately on startup so you get a standup right away
    run_standup()

    if not args.now:
        schedule_standup(hour=args.hour, minute=args.minute)
