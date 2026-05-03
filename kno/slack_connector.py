"""
Slack connector for kno.ai
--------------------------
Posts messages and standup summaries to Slack channels.

Config via env vars or direct args:
    SLACK_BOT_TOKEN   — xoxb-... bot token
    SLACK_CHANNEL     — default channel (e.g. #all-knoaiworkspace)
"""

import os
from dotenv import load_dotenv
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv('/Users/dhanapal/kno-ai/kno/.env')

_TOKEN   = os.environ.get("SLACK_BOT_TOKEN", "")
_CHANNEL = os.environ.get("SLACK_CHANNEL", "#all-knoaiworkspace")


def _client() -> WebClient:
    return WebClient(token=_TOKEN)


# ── Plain post ────────────────────────────────────────────────────────────────

def post_slack_message(text: str, channel: str | None = None) -> dict:
    """Post a plain-text message to a Slack channel.

    Args:
        text:    The message text to post.
        channel: Channel name or ID (default: SLACK_CHANNEL env var).

    Returns:
        Status dict with ok/error.
    """
    target = channel or _CHANNEL
    try:
        resp = _client().chat_postMessage(channel=target, text=text)
        return {"status": "success", "channel": target, "ts": resp["ts"]}
    except SlackApiError as e:
        return {"status": "error", "message": str(e)}


# ── Standup post (rich Block Kit formatting) ──────────────────────────────────

def _standup_blocks(summary_text: str, today: str) -> list:
    """Convert plain-text standup summary into Slack Block Kit blocks."""

    SECTION_ICONS = {
        "✅": ("✅ *DONE YESTERDAY*", "#2eb886"),
        "🔄": ("🔄 *IN PROGRESS*",    "#36a64f"),
        "📧": ("📧 *EMAILS TO ACTION*","#daa038"),
        "📋": ("📋 *CONFLUENCE UPDATES*","#888888"),
        "🚧": ("🚧 *BLOCKERS / NEEDS ATTENTION*","#e01e5a"),
    }

    blocks = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": f"🌅 Daily Standup — {today}", "emoji": True},
        },
        {"type": "divider"},
    ]

    lines = summary_text.splitlines()
    current_lines: list[str] = []
    current_colour = "#1a73e8"

    def _flush(colour: str):
        if current_lines:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": "\n".join(current_lines)},
            })
            current_lines.clear()

    for line in lines:
        emoji_key = next((e for e in SECTION_ICONS if line.strip().startswith(e)), None)
        if emoji_key:
            _flush(current_colour)
            label, current_colour = SECTION_ICONS[emoji_key]
            blocks.append({"type": "divider"})
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": label},
            })
        elif line.strip().startswith("•"):
            current_lines.append(f"  {line.strip()}")
        elif line.strip().startswith("Good morning"):
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"_{line.strip()}_"},
            })
        elif line.strip():
            current_lines.append(line.strip())

    _flush(current_colour)
    blocks.append({"type": "divider"})
    blocks.append({
        "type": "context",
        "elements": [{"type": "mrkdwn", "text": "_Sent by kno.ai standup agent_"}],
    })
    return blocks


def post_standup_to_slack(summary_text: str, today: str, channel: str | None = None) -> None:
    """Post a formatted standup summary to a Slack channel using Block Kit.

    Args:
        summary_text: Plain-text standup summary from the agent.
        today:        Date string for the header, e.g. "Monday, May 4 2026".
        channel:      Target channel (default: SLACK_CHANNEL env var).
    """
    target = channel or _CHANNEL
    try:
        _client().chat_postMessage(
            channel=target,
            text=f"🌅 Daily Standup — {today}",   # fallback for notifications
            blocks=_standup_blocks(summary_text, today),
        )
        print(f"💬  Standup posted to Slack {target}")
    except SlackApiError as e:
        print(f"⚠️  Slack post failed: {e}")
