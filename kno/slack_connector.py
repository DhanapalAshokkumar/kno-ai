"""
Slack connector for kno.ai
--------------------------
Read messages from and post messages to Slack channels.

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


# ── User name cache (avoid repeated API calls) ────────────────────────────────
_user_cache: dict[str, str] = {}

def _resolve_sender(cli: WebClient, user_id: str) -> str:
    """Resolve a Slack user ID to a display name, with caching."""
    if not user_id:
        return "unknown"
    if user_id in _user_cache:
        return _user_cache[user_id]
    try:
        info = cli.users_info(user=user_id)
        name = (
            info["user"].get("real_name")
            or info["user"].get("name")
            or user_id
        )
    except SlackApiError:
        name = user_id
    _user_cache[user_id] = name
    return name


def _channel_id(cli: WebClient, channel_name: str) -> str | None:
    """Resolve a channel name to its ID."""
    name = channel_name.lstrip("#")
    resp = cli.conversations_list(types="public_channel", limit=200)
    for ch in resp.get("channels", []):
        if ch["name"] == name:
            return ch["id"]
    return None


def _format_ts(ts: str) -> str:
    """Convert a Slack timestamp to a readable datetime string."""
    import datetime
    try:
        return datetime.datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


# ── Read messages ─────────────────────────────────────────────────────────────

def get_slack_channel_messages(channel: str | None = None, limit: int = 20) -> dict:
    """Get the most recent messages from a Slack channel.

    Args:
        channel: Channel name, e.g. 'all-knoaiworkspace' or '#general'.
                 Defaults to SLACK_CHANNEL env var.
        limit:   Number of recent messages to fetch (default: 20).

    Returns:
        Recent messages with sender name, text, and timestamp.
    """
    try:
        cli = _client()
        target = (channel or _CHANNEL).lstrip("#")
        ch_id = _channel_id(cli, target)
        if not ch_id:
            return {"status": "error", "message": f"Channel #{target} not found"}

        history = cli.conversations_history(channel=ch_id, limit=limit)
        messages = [
            {
                "sender":    _resolve_sender(cli, m.get("user", "")),
                "text":      m.get("text", "")[:500],
                "timestamp": _format_ts(m.get("ts", "")),
            }
            for m in history.get("messages", [])
            if m.get("text", "").strip()
        ]
        if not messages:
            return {"status": "no_results", "message": f"No messages in #{target}"}
        return {"status": "success", "channel": target, "count": len(messages), "messages": messages}

    except SlackApiError as e:
        return {"status": "error", "message": str(e)}


def search_slack_messages(query: str, channel: str | None = None, limit: int = 50) -> dict:
    """Search recent Slack messages in a channel by keyword.

    Args:
        query:   Keyword or phrase to search for. Use '*' to return all messages.
        channel: Channel name to search (default: SLACK_CHANNEL env var).
                 Use 'all' to search across all accessible channels.
        limit:   Max messages to scan per channel (default: 50).

    Returns:
        Matching messages with channel, sender name, text, and timestamp.
    """
    try:
        cli = _client()
        target = (channel or _CHANNEL).lstrip("#")

        all_channels = cli.conversations_list(
            types="public_channel", limit=200
        ).get("channels", [])

        if target == "all":
            channels_to_search = all_channels
        else:
            channels_to_search = [c for c in all_channels if c["name"] == target]
            if not channels_to_search:
                return {"status": "error", "message": f"Channel #{target} not found"}

        results = []
        wildcard = query.strip() == "*"
        for ch in channels_to_search:
            try:
                history = cli.conversations_history(channel=ch["id"], limit=limit)
                for msg in history.get("messages", []):
                    text = msg.get("text", "")
                    if not text.strip():
                        continue
                    if wildcard or query.lower() in text.lower():
                        results.append({
                            "channel":   ch["name"],
                            "sender":    _resolve_sender(cli, msg.get("user", "")),
                            "text":      text[:500],
                            "timestamp": _format_ts(msg.get("ts", "")),
                        })
            except SlackApiError:
                continue

        if not results:
            return {"status": "no_results", "message": f"No Slack messages found for: {query}"}
        return {"status": "success", "count": len(results), "messages": results}

    except SlackApiError as e:
        return {"status": "error", "message": str(e)}


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
