"""
Citation formatter for kno.ai.
Converts raw tool results into numbered citations with source metadata.
"""
from typing import Any
from datetime import datetime


def format_citations(sources: list[dict]) -> tuple[dict[int, str], str]:
    """Convert a list of source dicts into inline citation markers + references block.

    Args:
        sources: List of dicts with keys: title, url, author, date, source, text

    Returns:
        Tuple of:
          - markers: {1: "[1]", 2: "[2]", ...}  — for insertion into prose
          - references: formatted references block as a string
    """
    if not sources:
        return {}, ""

    markers = {}
    ref_lines = ["\n---\n**Sources**\n"]

    for i, src in enumerate(sources, 1):
        markers[i] = f"[{i}]"

        source_type = src.get("source", "").capitalize()
        title   = src.get("title", "Unknown")
        url     = src.get("url", "")
        author  = src.get("author", "")
        date    = src.get("date", "")
        excerpt = src.get("text", src.get("excerpt", ""))[:200].strip()

        # Format date nicely if it's ISO
        try:
            date = datetime.fromisoformat(date.replace("Z", "+00:00")).strftime("%b %d, %Y")
        except Exception:
            pass

        line = f"**[{i}] {source_type}: {title}**"
        if author:
            line += f"  \nAuthor: {author}"
        if date:
            line += f" · Updated: {date}"
        if url:
            line += f"  \n[{url}]({url})"
        if excerpt:
            clean = excerpt.replace("\n", " ").strip()
            line += f'  \n> "{clean}..."'

        ref_lines.append(line)

    return markers, "\n\n".join(ref_lines)


def annotate_response(response: str, sources: list[dict]) -> str:
    """Append a references block to an agent response.

    If the response already contains [1], [2] markers, just add the
    references block. Otherwise, append sources at the end.

    Args:
        response: The agent's response text
        sources:  List of source dicts

    Returns:
        Response with references block appended.
    """
    if not sources:
        return response

    _, refs = format_citations(sources)
    return response.rstrip() + "\n" + refs


def sources_from_gmail(threads: list[dict]) -> list[dict]:
    """Convert Gmail thread results to citation sources."""
    return [
        {
            "source": "Gmail",
            "title": t.get("subject", "(no subject)"),
            "url": f"https://mail.google.com/mail/u/0/#search/rfc822msgid/{t.get('thread_id','')}",
            "author": t.get("from", ""),
            "date": t.get("date", ""),
            "text": (t.get("body", "") or "")[:200],
        }
        for t in (threads or [])
    ]


def sources_from_drive(files: list[dict]) -> list[dict]:
    """Convert Drive file results to citation sources."""
    return [
        {
            "source": "Google Drive",
            "title": f.get("name", "Untitled"),
            "url": f.get("url", f.get("webViewLink", "")),
            "author": f.get("owner", ""),
            "date": f.get("modified", f.get("modifiedTime", "")),
            "text": f.get("snippet", ""),
        }
        for f in (files or [])
    ]


def sources_from_slack(messages: list[dict]) -> list[dict]:
    """Convert Slack search results to citation sources."""
    from datetime import datetime
    results = []
    for m in (messages or []):
        ts = m.get("timestamp", "")
        # Convert Slack epoch timestamp to a readable date
        try:
            date_str = datetime.fromtimestamp(float(ts)).strftime("%b %d, %Y")
        except Exception:
            date_str = ts
        results.append({
            "source": "Slack",
            "title": f"#{m.get('channel', 'unknown')} — {m.get('text', '')[:60]}",
            "url": m.get("permalink", ""),
            "author": m.get("author", m.get("user", "")),
            "date": date_str,
            "text": m.get("text", "")[:200],
        })
    return results


def sources_from_confluence(pages: list[dict]) -> list[dict]:
    """Convert Confluence page results to citation sources."""
    return [
        {
            "source": "Confluence",
            "title": p.get("title", "Untitled"),
            "url": p.get("url", ""),
            "author": "",
            "date": "",
            "text": p.get("content", "")[:200],
        }
        for p in (pages or [])
    ]


def sources_from_jira(issues: list[dict]) -> list[dict]:
    """Convert Jira issue results to citation sources."""
    return [
        {
            "source": "Jira",
            "title": f"{i.get('key','')} — {i.get('summary','')}",
            "url": i.get("url", ""),
            "author": i.get("assignee", ""),
            "date": "",
            "text": i.get("summary", ""),
        }
        for i in (issues or [])
    ]


def sources_from_github(items: list[dict], kind: str = "Issue") -> list[dict]:
    """Convert GitHub issue/PR results to citation sources."""
    return [
        {
            "source": f"GitHub {kind}",
            "title": f"#{i.get('number','')} {i.get('title','')}",
            "url": i.get("url", ""),
            "author": i.get("author", ""),
            "date": "",
            "text": i.get("title", ""),
        }
        for i in (items or [])
    ]


def sources_from_zoho(items: list[dict], kind: str = "Deal") -> list[dict]:
    """Convert Zoho CRM results to citation sources."""
    return [
        {
            "source": f"Zoho CRM {kind}",
            "title": i.get("deal_name") or i.get("first_name", "") + " " + i.get("last_name", ""),
            "url": "",
            "author": "",
            "date": i.get("closing_date", ""),
            "text": f"Stage: {i.get('stage','')} | Amount: ${i.get('amount','')}",
        }
        for i in (items or [])
    ]
