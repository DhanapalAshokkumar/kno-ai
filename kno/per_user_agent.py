"""
Multi-tenant agent runner.
Each user's query executes with ONLY their own credentials — never another user's.
Vertex AI Session Service gives persistent sessions across Cloud Run instances.
Vertex AI Memory Bank gives long-term memory across conversations.
"""
import os
import base64
from datetime import datetime, timezone, timedelta
from typing import Optional, AsyncGenerator

from google.adk.agents.llm_agent import Agent
from google.adk.runners import Runner
from google.adk.sessions import VertexAiSessionService, InMemorySessionService
from google.adk.memory import VertexAiMemoryBankService, InMemoryMemoryService
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
from google.adk.tools.load_memory_tool import LoadMemoryTool
from google.genai import types as genai_types
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

from kno.user_store import get_app_credentials

import logging
logger = logging.getLogger(__name__)


# ── Shared date helper ────────────────────────────────────────────────────────

def _cutoff_dt(days_ago: Optional[int]) -> Optional[datetime]:
    """Return a UTC datetime N days in the past, or None."""
    if days_ago is None:
        return None
    return datetime.now(timezone.utc) - timedelta(days=days_ago)


def _gmail_after(days_ago: Optional[int]) -> str:
    """Return a Gmail 'after:YYYY/MM/DD' token, or empty string."""
    if days_ago is None:
        return ""
    dt = _cutoff_dt(days_ago)
    return f" after:{dt.strftime('%Y/%m/%d')}"


def _ts_after(days_ago: Optional[int]) -> Optional[float]:
    """Return a Unix timestamp cutoff, or None."""
    if days_ago is None:
        return None
    return _cutoff_dt(days_ago).timestamp()

# ── Vertex AI config ──────────────────────────────────────────────────────────
_GCP_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_GCP_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_USE_VERTEX   = os.environ.get("GOOGLE_GENAI_USE_VERTEXAI", "0") == "1"

# ── Service config ────────────────────────────────────────────────────────────
# AGENT_ENGINE_ID is required for VertexAiMemoryBankService (long-term memory).
# It's optional for VertexAiSessionService (persistent sessions still work without it).
# Set AGENT_ENGINE_ID env var once a Vertex AI Agent Engine has been created.
_AGENT_ENGINE_ID = os.environ.get("AGENT_ENGINE_ID", "")

# Build session + memory services once at module load
# (Cloud Run keeps the module warm between requests)
def _make_services():
    if _USE_VERTEX:
        # Persistent sessions across Cloud Run instances — no engine ID needed
        session_svc = VertexAiSessionService(
            project=_GCP_PROJECT,
            location=_GCP_LOCATION,
        )
        # Long-term memory across sessions requires an Agent Engine
        if _AGENT_ENGINE_ID:
            memory_svc = VertexAiMemoryBankService(
                project=_GCP_PROJECT,
                location=_GCP_LOCATION,
                agent_engine_id=_AGENT_ENGINE_ID,
            )
        else:
            # Fallback: per-session memory only (no cross-session recall)
            memory_svc = InMemoryMemoryService()
    else:
        # Local dev fallback — in-memory, no GCP needed
        session_svc = InMemorySessionService()
        memory_svc  = InMemoryMemoryService()
    return session_svc, memory_svc

_session_service, _memory_service = _make_services()

SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",   # read + send + labels (superset of readonly)
    "https://www.googleapis.com/auth/drive.readonly",
]

_INSTRUCTION = """You are kno, an AI assistant for company knowledge.
Help employees find information from their connected tools quickly and accurately.

## Citations — REQUIRED for every response
- Every factual claim MUST have a citation marker like [1], [2].
- After your answer, ALWAYS include a **Sources** section.
- Format EXACTLY like this — no raw JSON, no data dumps:

  **Sources**
  [1] Gmail: Subject of email — From: sender@example.com | Date: May 8, 2026
  [2] Zoho CRM: Deal Name — Stage: Negotiation | Amount: $70,000 | Closing: Apr 29 | [link](url)
  [3] Confluence: Page Title — Space: ENG | Updated: May 1, 2026 | [link](url)
  [4] Jira: KEY-123 — Summary of issue | Status: In Progress | [link](url)
  [5] GitHub: owner/repo-name — Description of repo | Updated: May 8, 2026 | [link](url)
  [6] Slack: #channel-name — snippet of message | Author: username | Date: May 8, 2026 | [link](url)
  [7] Google Drive: Document Title — Owner: name | Updated: May 1, 2026 | [link](url)

- Keep source lines SHORT and human-readable. Never paste raw JSON or full email bodies.
- For GitHub repos: each repo in the list has its OWN "source_number" field — use THAT number as the citation marker. Never reuse the same number for multiple repos.
- If no source found: write *No source found.*

## Search strategy
- Always search before saying you don't know.
- For documents/knowledge base:
  * User asks about a TOPIC ("find docs about onboarding", "deployment guide") → call search_knowledge_base(query="...")
  * User asks by DATE only ("updated last 30 days", "recent docs", "this month") → call browse_knowledge_base(days_ago=30) ← NO keyword needed, call immediately
  * User asks by AUTHOR only ("docs by Dhanapal", "written by Alice") → call browse_knowledge_base(author="Dhanapal") ← NO keyword needed, call immediately
  * User asks by SOURCE only ("all Confluence pages") → call browse_knowledge_base(source_type="confluence") ← call immediately
  * Combined date+author or date+source → call browse_knowledge_base(days_ago=30, author="Alice")
  RULE: NEVER ask the user for a keyword when they specify a date, author, or source filter. Call browse_knowledge_base immediately.
- For emails: use search_gmail. Show subject, sender, date — NOT full body.
- For files: use search_drive. It automatically reads PDFs, images, Slides, and Sheets via Gemini multimodal — the full extracted content is in the "content" field of each result. No second tool call needed. Always mention the file type in citations: "[1] Google Drive (PDF): Annual Report 2025". For a named file the user asks about directly, you can also call read_drive_file_multimodal(file_id, file_name, mime_type) if you already have the file_id.
- For team chat:
  * "What's been discussed in Slack?" → call get_slack_activity()  ← no parameters, call immediately
  * "What's happening in Slack?" → call get_slack_activity()
  * "Summarise #general" → call get_slack_activity()
  * "Search Slack for budget" → call search_slack_messages(query="budget")
  RULE: For any open-ended Slack question, call get_slack_activity() immediately — zero parameters, no clarifying question needed.
- For tasks/bugs: use search_jira_issues. To CREATE a ticket: create_jira_issue(summary, project, issue_type, description). To UPDATE: update_jira_issue(issue_key, status/comment/assignee).
- For code/PRs: use list_github_repos, search_github_issues, or get_github_pull_requests.
- For CRM: use search_zoho_contacts or search_zoho_deals. To UPDATE a deal: update_zoho_deal(deal_id, stage/amount/closing_date). To LOG activity: create_zoho_activity(deal_id, subject, activity_type).

## Write / action tools — use when the user asks you to DO something
- "Send an email to Alice about X" → send_gmail(to, subject, body)
- "Post to #general that X" → post_slack_message(channel, text)
- "Create a Jira ticket for X" → create_jira_issue(summary, project, issue_type, description)
- "Move ENG-42 to Done" → update_jira_issue(issue_key="ENG-42", status="Done")
- "Add a comment to ENG-42" → update_jira_issue(issue_key="ENG-42", comment="...")
- "Log a follow-up call on the TechCo deal" → first call search_zoho_deals(name="TechCo") to get deal_id, then create_zoho_activity(deal_id, subject, activity_type="Call")
- "Update the Acme deal stage to Closed Won" → first call search_zoho_deals(name="Acme") to get deal_id, then update_zoho_deal(deal_id, stage="Closed Won")
RULE: Always confirm the action with the user BEFORE calling a write tool, unless they explicitly said "do it" or "go ahead". Show what you're about to do: "I'll create a Jira ticket: [summary] in project ENG — shall I proceed?"

## Formatting
- Be concise. Use bullet points. Employees are busy.
- For emails: show subject + sender + 1-sentence summary ONLY — never paste full body.
- For deals: show name, stage, amount, closing date — nothing else.
- Keep total response under 400 words unless the user asks for detail.

## Metadata filters — use these whenever the user implies time, person, or source
Every search tool accepts optional filter parameters. Apply them proactively:

| User says | Filter to use |
|---|---|
| "recently", "last week", "this week" | days_ago=7 |
| "last month", "past month" | days_ago=30 |
| "today", "yesterday" | days_ago=1 |
| "last 90 days", "this quarter" | days_ago=90 |
| "by Alice", "from Bob", "Alice's" | author="Alice" |
| "in Confluence", "Confluence pages" | source_type="confluence" (KB only) |
| "in Gmail" / "emails" | use search_gmail directly |

Rules:
- ALWAYS add days_ago when user says "recent", "latest", "last N days/weeks/months".
- ALWAYS add author when user names a person in context of finding content.
- For KB only: add source_type to limit to one source ("confluence", "github", "drive", etc.).
- Do NOT invent filters the user didn't imply.
- KB filter-only queries → use browse_knowledge_base (not search_knowledge_base). NEVER ask for a keyword when the user gives a date, author, or source.

## Memory
- Use load_memory when user references past conversations ("last week", "that deal").
- If a tool returns "not connected": tell user to go to Settings → Connect Apps."""


# ── Per-user Google services ──────────────────────────────────────────────────

def _google_creds(creds_data: dict) -> Credentials:
    c = Credentials(
        token=None,
        refresh_token=creds_data["refresh_token"],
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=SCOPES,
    )
    c.refresh(Request())
    return c


# ── Tool factories (scoped to one user) ──────────────────────────────────────

def _make_gmail_tool(email: str):
    creds_data = get_app_credentials(email, "gmail")

    def search_gmail(query: str, max_results: int = 5,
                     days_ago: int = None, author: str = None) -> dict:
        """Search Gmail for emails and threads matching a query.

        Args:
            query: Gmail search query, e.g. 'budget approval'
            max_results: Max threads to return (default 5)
            days_ago: Only return emails from the last N days (e.g. 7, 30)
            author: Filter by sender name or email, e.g. 'alice@company.com'
        """
        if not creds_data:
            return {"status": "error", "message": "Gmail not connected — go to Settings to connect your Gmail."}
        try:
            svc = build("gmail", "v1", credentials=_google_creds(creds_data))
            # Build Gmail query string with optional filters
            full_query = query
            full_query += _gmail_after(days_ago)
            if author:
                full_query += f" from:{author}"
            resp = svc.users().threads().list(userId="me", q=full_query, maxResults=max_results).execute()
            threads = resp.get("threads", [])
            if not threads:
                return {"status": "no_results", "message": f"No emails found for: {query}"}
            results = []
            for t in threads:
                detail = svc.users().threads().get(userId="me", id=t["id"], format="full").execute()
                msgs = detail.get("messages", [])
                headers, body_parts = {}, []
                for msg in msgs:
                    payload = msg.get("payload", {})
                    if not headers:
                        headers = {h["name"]: h["value"] for h in payload.get("headers", [])}

                    def _extract(p):
                        if p.get("mimeType") == "text/plain":
                            d = p.get("body", {}).get("data", "")
                            return base64.urlsafe_b64decode(d + "==").decode("utf-8", errors="replace") if d else ""
                        for part in p.get("parts", []):
                            t = _extract(part)
                            if t: return t
                        return ""

                    text = _extract(payload)
                    if text.strip():
                        body_parts.append(text.strip())
                results.append({
                    "thread_id": t["id"],
                    "subject": headers.get("Subject", "(no subject)"),
                    "from": headers.get("From", "unknown"),
                    "date": headers.get("Date", "unknown"),
                    "snippet": "\n---\n".join(body_parts)[:300],  # short snippet only
                })
            return {"status": "success", "count": len(results), "threads": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def send_gmail(to: str, subject: str, body: str,
                   cc: str = None, reply_to_thread_id: str = None) -> dict:
        """Send an email via Gmail.

        Args:
            to:                 Recipient email address, e.g. 'alice@company.com'
            subject:            Email subject line
            body:               Plain-text email body
            cc:                 Optional CC email address
            reply_to_thread_id: Optional Gmail thread ID to reply into
        """
        if not creds_data:
            return {"status": "error", "message": "Gmail not connected — go to Settings."}
        try:
            import email.mime.text
            import email.mime.multipart

            svc = build("gmail", "v1", credentials=_google_creds(creds_data))

            msg = email.mime.multipart.MIMEMultipart()
            msg["To"]      = to
            msg["Subject"] = subject
            if cc:
                msg["Cc"] = cc
            msg.attach(email.mime.text.MIMEText(body, "plain"))

            raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
            payload: dict = {"raw": raw}
            if reply_to_thread_id:
                payload["threadId"] = reply_to_thread_id

            sent = svc.users().messages().send(userId="me", body=payload).execute()
            return {
                "status":    "sent",
                "message_id": sent.get("id"),
                "thread_id":  sent.get("threadId"),
                "to":         to,
                "subject":    subject,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [search_gmail, send_gmail]


def _make_drive_tools(email: str) -> list:
    """Return two Drive tools: search_drive and read_drive_file_multimodal."""
    creds_data = get_app_credentials(email, "gmail")  # Gmail and Drive share OAuth

    # ── MIME type helpers ─────────────────────────────────────────────────────
    _GOOGLE_EXPORT = {
        "application/vnd.google-apps.document":     ("text/plain",              "txt"),
        "application/vnd.google-apps.spreadsheet":  ("text/csv",                "csv"),
        "application/vnd.google-apps.presentation": ("application/pdf",         "pdf"),
        "application/vnd.google-apps.drawing":      ("image/png",               "png"),
    }
    _MULTIMODAL_MIME = {
        "application/pdf",
        "image/jpeg", "image/jpg", "image/png", "image/gif",
        "image/webp", "image/heic", "image/heif",
    }
    _LABEL = {
        "application/pdf":                           "PDF",
        "application/vnd.google-apps.spreadsheet":  "Sheet",
        "application/vnd.google-apps.presentation": "Slides",
        "application/vnd.google-apps.document":     "Doc",
        "image/jpeg": "Image", "image/jpg": "Image",
        "image/png":  "Image", "image/gif": "Image",
    }

    def _type_label(mime: str) -> str:
        for k, v in _LABEL.items():
            if k in mime:
                return v
        return "File"

    def _is_multimodal_candidate(mime: str) -> bool:
        """True for file types Gemini can read natively via inline_data."""
        if mime in _MULTIMODAL_MIME:
            return True
        if "google-apps.presentation" in mime or "google-apps.spreadsheet" in mime:
            return True   # will be exported to PDF/CSV first
        return False

    # ── Shared read logic (used by both tools) ───────────────────────────────

    def _extract_file_content(svc, file_id: str, file_name: str,
                               mime_type: str) -> str:
        """Download a Drive file and extract its text content.

        For PDFs and images: uses Gemini multimodal (inline_data).
        For Sheets: exports as CSV.
        For Docs/Slides: exports as plain text / PDF then reads via Gemini.
        Returns extracted text (up to 6000 chars) or empty string on failure.
        """
        import io
        import re as _re
        import vertexai
        from vertexai.generative_models import GenerativeModel, Part
        from googleapiclient.http import MediaIoBaseDownload

        actual_mime = mime_type

        if mime_type in _GOOGLE_EXPORT:
            export_mime, _ = _GOOGLE_EXPORT[mime_type]
            actual_mime    = export_mime
            req = svc.files().export_media(fileId=file_id, mimeType=export_mime)
        else:
            req = svc.files().get_media(fileId=file_id)

        buf = io.BytesIO()
        dl  = MediaIoBaseDownload(buf, req)
        done = False
        while not done:
            _, done = dl.next_chunk()
        raw_bytes = buf.getvalue()

        # Plain-text fast paths (no Gemini needed)
        if actual_mime in ("text/plain", "text/csv"):
            text = raw_bytes.decode("utf-8", errors="replace")
            return _re.sub(r"\s+", " ", text).strip()[:6000]

        # Gemini multimodal for PDF / images / exported Slides-as-PDF
        vertexai.init(project=_GCP_PROJECT, location=_GCP_LOCATION)
        model     = GenerativeModel("gemini-2.5-flash")
        file_part = Part.from_data(data=raw_bytes, mime_type=actual_mime)
        response  = model.generate_content([
            file_part,
            "Extract ALL text content from this file. "
            "Return the full text preserving headings and structure. "
            "Include all tables, figures, dates, and numbers. "
            "Do not summarise — extract everything verbatim.",
        ])
        return response.text.strip()[:8000]

    # ── Tool 1: search_drive ──────────────────────────────────────────────────

    def search_drive(query: str, max_results: int = 5,
                     days_ago: int = None, author: str = None) -> dict:
        """Search Google Drive for files matching a query.

        Automatically reads PDFs, images, Slides, and Sheets using Gemini
        multimodal and returns their full extracted content alongside metadata.

        Args:
            query:       Keyword search, e.g. 'Q3 roadmap' or a file name like 'Resume.pdf'
            max_results: Max files to return (default 5)
            days_ago:    Only return files modified in the last N days
            author:      Filter by owner display name, e.g. 'Alice Smith'
        """
        if not creds_data:
            return {"status": "error",
                    "message": "Google Drive not connected — go to Settings to connect your Google account."}
        try:
            import re as _re
            svc = build("drive", "v3", credentials=_google_creds(creds_data))
            drive_q = f"fullText contains '{query}' and trashed=false"
            if days_ago:
                cutoff = _cutoff_dt(days_ago).strftime("%Y-%m-%dT%H:%M:%S")
                drive_q += f" and modifiedTime > '{cutoff}'"
            resp = svc.files().list(
                q=drive_q,
                pageSize=max_results,
                fields="files(id,name,mimeType,modifiedTime,webViewLink,owners)",
            ).execute()
            files = resp.get("files", [])
            if not files:
                return {"status": "no_results", "message": f"No files found for: {query}"}

            results = []
            for f in files:
                mime = f.get("mimeType", "")
                file_entry = {
                    "id":       f["id"],
                    "name":     f.get("name"),
                    "mime_type": mime,
                    "type":     _type_label(mime),
                    "modified": f.get("modifiedTime"),
                    "url":      f.get("webViewLink"),
                    "owner":    (f.get("owners") or [{}])[0].get("displayName", ""),
                    "content":  "",
                }

                # Auto-extract content for every supported file type
                if _is_multimodal_candidate(mime) or "google-apps.document" in mime:
                    try:
                        content = _extract_file_content(svc, f["id"], f.get("name",""), mime)
                        file_entry["content"] = content
                    except Exception as ex:
                        logger.error("Auto-read failed for %s (%s): %s", f.get("name"), mime, ex)
                        file_entry["content"] = ""

                results.append(file_entry)

            # Post-filter by author/owner
            if author:
                al = author.lower()
                results = [r for r in results if al in r.get("owner", "").lower()]

            if not results:
                return {"status": "no_results", "message": f"No files found for: {query}"}
            return {"status": "success", "count": len(results), "files": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Tool 2: read_drive_file_multimodal ────────────────────────────────────
    # Kept as an explicit tool for when the user names a specific file directly.

    def read_drive_file_multimodal(file_id: str, file_name: str,
                                    mime_type: str) -> dict:
        """Read a specific Drive file using Gemini multimodal.

        Use when the user asks about a named file (e.g. 'read Resume.pdf').
        Works for PDFs, images, Slides, Sheets, and scanned documents.

        Args:
            file_id:   Drive file ID (get from search_drive if unknown)
            file_name: File name for the citation
            mime_type: MIME type, e.g. 'application/pdf'
        """
        if not creds_data:
            return {"status": "error", "message": "Google Drive not connected."}
        try:
            svc = build("drive", "v3", credentials=_google_creds(creds_data))
            content = _extract_file_content(svc, file_id, file_name, mime_type)
            return {
                "status":    "success",
                "file_name": file_name,
                "mime_type": mime_type,
                "type":      _type_label(mime_type),
                "content":   content,
            }
        except Exception as e:
            logger.error("read_drive_file_multimodal failed for %s: %s", file_name, e)
            return {"status": "error", "message": str(e)}

    return [search_drive, read_drive_file_multimodal]


def _make_slack_tools(email: str):
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    creds_data = get_app_credentials(email, "slack")

    # Shared user-ID resolver (cached per tool-factory call)
    _slack_user_cache: dict = {}

    def _resolve_slack_user(cli, uid: str) -> str:
        if not uid or uid.startswith("B"):
            return uid
        if uid in _slack_user_cache:
            return _slack_user_cache[uid]
        try:
            info = cli.users_info(user=uid)
            name = (info.get("user") or {}).get("real_name") or \
                   (info.get("user") or {}).get("name") or uid
            _slack_user_cache[uid] = name
            return name
        except Exception:
            return uid

    def get_slack_activity(days_ago: int = None, author: str = None) -> dict:
        """Return recent messages from the most active Slack channels.

        Call this for ANY question about recent Slack activity, e.g.:
        "What has been discussed in Slack?", "What's happening in Slack?",
        "Any recent updates?", "Summarise Slack", "What's been said in #general?"

        Args:
            days_ago: Only return messages from the last N days (e.g. 7)
            author: Filter by author display name, e.g. 'Alice'
        """
        if not creds_data:
            return {"status": "error", "message": "Slack not connected — go to Settings to connect Slack."}
        try:
            token = creds_data.get("user_token") or creds_data.get("bot_token", "")
            if not token:
                return {"status": "error", "message": "Slack token missing — reconnect Slack in Settings."}
            cli = WebClient(token=token)
            ts_cutoff = _ts_after(days_ago)

            try:
                ch_resp = cli.conversations_list(
                    types="public_channel,private_channel", limit=200, exclude_archived=True)
            except SlackApiError:
                ch_resp = cli.conversations_list(
                    types="public_channel", limit=200, exclude_archived=True)

            all_channels = ch_resp.get("channels", [])
            all_channels.sort(key=lambda c: c.get("num_members", 0), reverse=True)

            results = []
            for ch in all_channels[:6]:
                try:
                    kwargs = {"channel": ch["id"], "limit": 20}
                    if ts_cutoff:
                        kwargs["oldest"] = str(ts_cutoff)
                    hist = cli.conversations_history(**kwargs)
                    for msg in hist.get("messages", []):
                        text = msg.get("text", "").strip()
                        if not text or msg.get("subtype"):
                            continue
                        ts_raw      = msg.get("ts", "")
                        uid         = msg.get("user", msg.get("username", ""))
                        author_name = _resolve_slack_user(cli, uid) if uid else "unknown"
                        if author and author.lower() not in author_name.lower():
                            continue
                        results.append({
                            "channel":   ch.get("name", "unknown"),
                            "author":    author_name,
                            "text":      text[:400],
                            "timestamp": ts_raw,
                            "permalink": f"https://slack.com/archives/{ch['id']}/p{ts_raw.replace('.','')}" if ts_raw else "",
                        })
                except SlackApiError:
                    continue

            if not results:
                return {"status": "no_results", "message": "No recent Slack messages found."}
            return {"status": "success", "count": len(results), "messages": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_slack_messages(query: str, count: int = 10,
                              days_ago: int = None, author: str = None) -> dict:
        """Search Slack for messages containing a specific keyword or phrase.

        Use this ONLY when the user names a specific topic to look for,
        e.g. "search Slack for budget approval" or "find Slack messages about the outage".
        For general recent activity use get_slack_activity instead.

        Args:
            query: Keyword or phrase to search for (required)
            count: Max messages to return (default 10)
            days_ago: Only return messages from the last N days
            author: Filter by author display name, e.g. 'Alice'
        """
        if not creds_data:
            return {"status": "error", "message": "Slack not connected — go to Settings to connect Slack."}
        try:
            token = creds_data.get("user_token") or creds_data.get("bot_token", "")
            if not token:
                return {"status": "error", "message": "Slack token missing — reconnect Slack in Settings."}
            cli = WebClient(token=token)
            # Slack Search API supports "after:YYYY-MM-DD" in query string
            full_query = query
            if days_ago:
                cutoff_date = _cutoff_dt(days_ago).strftime("%Y-%m-%d")
                full_query += f" after:{cutoff_date}"
            if author:
                full_query += f" from:{author}"
            resp = cli.search_messages(query=full_query, count=count, sort="timestamp", sort_dir="desc")
            matches = resp.get("messages", {}).get("matches", [])
            if not matches:
                return {"status": "no_results", "message": f"No Slack messages found for: {query}"}

            return {"status": "success", "count": len(matches), "messages": [
                {
                    "channel": m.get("channel", {}).get("name", "unknown"),
                    "author": _resolve_slack_user(cli, m.get("user", m.get("username", ""))),
                    "text": m.get("text", "")[:400],
                    "timestamp": m.get("ts", ""),
                    "permalink": m.get("permalink", ""),
                }
                for m in matches
            ]}
        except SlackApiError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def post_slack_message(channel: str, text: str,
                            thread_ts: str = None) -> dict:
        """Post a message to a Slack channel or thread.

        Args:
            channel:   Channel name (e.g. 'general') or ID (e.g. 'C012AB3CD')
            text:      Message text — supports Slack mrkdwn formatting
            thread_ts: Optional timestamp of parent message to reply in-thread
        """
        if not creds_data:
            return {"status": "error", "message": "Slack not connected — go to Settings."}
        try:
            cli = WebClient(token=creds_data.get("user_token") or creds_data.get("bot_token", ""))
            # Normalise channel name — strip leading # if present
            ch = channel.lstrip("#")
            kwargs: dict = {"channel": ch, "text": text}
            if thread_ts:
                kwargs["thread_ts"] = thread_ts
            resp = cli.chat_postMessage(**kwargs)
            return {
                "status":     "sent",
                "channel":    resp.get("channel"),
                "ts":         resp.get("ts"),
                "message":    text[:100],
            }
        except SlackApiError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [get_slack_activity, search_slack_messages, post_slack_message]


def _make_github_tools(email: str):
    import requests as req

    creds_data = get_app_credentials(email, "github")

    def _headers():
        token = creds_data.get("token", "") if creds_data else ""
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def search_github_issues(query: str, repo: str = None, state: str = "open",
                             days_ago: int = None, author: str = None) -> dict:
        """Search GitHub issues by keyword.

        Args:
            query: Text to search in issue titles/bodies
            repo: 'owner/repo' or bare repo name
            state: 'open', 'closed', or 'all'
            days_ago: Only return issues updated in the last N days
            author: Filter by issue author (GitHub username)
        """
        if not creds_data:
            return {"status": "error", "message": "GitHub not connected — go to Settings to connect GitHub."}
        try:
            owner = creds_data.get("owner", "")
            q = f"{query} repo:{owner}/{repo or owner} is:issue state:{state}"
            if days_ago:
                cutoff = _cutoff_dt(days_ago).strftime("%Y-%m-%d")
                q += f" updated:>{cutoff}"
            if author:
                q += f" author:{author}"
            r = req.get("https://api.github.com/search/issues",
                        headers=_headers(), params={"q": q, "per_page": 10})
            items = r.json().get("items", [])
            if not items:
                return {"status": "no_results", "message": f"No issues found for: {query}"}
            return {"status": "success", "count": len(items), "issues": [
                {"number": i["number"], "title": i["title"],
                 "state": i["state"], "url": i["html_url"]}
                for i in items
            ]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_github_repos() -> dict:
        """List all GitHub repositories for the connected owner/org.

        Returns:
            All repos with name, description, URL, and source_number.
            IMPORTANT: cite each repo using its own source_number field as the
            citation marker [N] — every repo is a distinct source with a unique number.
        """
        if not creds_data:
            return {"status": "error", "message": "GitHub not connected — go to Settings to connect GitHub."}
        try:
            owner = creds_data.get("owner", "")
            # Try org repos first, fall back to user repos
            r = req.get(f"https://api.github.com/orgs/{owner}/repos",
                        headers=_headers(), params={"per_page": 30, "sort": "updated"})
            if r.status_code == 404:
                r = req.get(f"https://api.github.com/users/{owner}/repos",
                            headers=_headers(), params={"per_page": 30, "sort": "updated"})
            repos = r.json() if r.ok else []
            if not repos:
                return {"status": "no_results", "message": f"No repos found for {owner}"}
            return {
                "status": "success",
                "count": len(repos),
                "citation_note": "Each repo below has a unique source_number — use it as [N] in citations.",
                "repos": [
                    {
                        "source_number": idx + 1,
                        "name": rp["name"],
                        "full_name": rp["full_name"],
                        "description": rp.get("description", ""),
                        "url": rp["html_url"],
                        "updated": rp.get("updated_at", ""),
                    }
                    for idx, rp in enumerate(repos)
                ],
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_github_pull_requests(repo: str = None, state: str = "open",
                                  days_ago: int = None, author: str = None) -> dict:
        """List pull requests for a GitHub repo.

        Args:
            repo: bare repo name (e.g. 'kno-ai') or 'owner/repo'. If omitted,
                  lists PRs across ALL repos for the connected owner.
            state: 'open', 'closed', or 'all'
            days_ago: Only return PRs updated in the last N days
            author: Filter by PR author (GitHub username)
        """
        if not creds_data:
            return {"status": "error", "message": "GitHub not connected — go to Settings to connect GitHub."}
        try:
            owner = creds_data.get("owner", "")
            if repo:
                repos_to_check = [f"{owner}/{repo}" if "/" not in repo else repo]
            else:
                # Get all repos and check each for PRs
                r = req.get(f"https://api.github.com/users/{owner}/repos",
                            headers=_headers(), params={"per_page": 20, "sort": "updated"})
                if not r.ok:
                    r = req.get(f"https://api.github.com/orgs/{owner}/repos",
                                headers=_headers(), params={"per_page": 20, "sort": "updated"})
                repos_to_check = [rp["full_name"] for rp in (r.json() if r.ok else [])]

            ts_cutoff = _cutoff_dt(days_ago) if days_ago else None
            all_prs = []
            for full_repo in repos_to_check[:10]:  # cap at 10 repos
                r = req.get(f"https://api.github.com/repos/{full_repo}/pulls",
                            headers=_headers(), params={"state": state, "per_page": 20})
                if r.ok:
                    for p in r.json():
                        # days_ago filter: check updated_at
                        if ts_cutoff:
                            updated = p.get("updated_at", "")
                            try:
                                pr_dt = datetime.strptime(updated, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                                if pr_dt < ts_cutoff:
                                    continue
                            except ValueError:
                                pass
                        # author filter
                        pr_author = p["user"]["login"]
                        if author and author.lower() not in pr_author.lower():
                            continue
                        all_prs.append({
                            "repo": full_repo,
                            "number": p["number"], "title": p["title"],
                            "author": pr_author, "url": p["html_url"],
                            "updated": p.get("updated_at", ""),
                        })
            if not all_prs:
                return {"status": "no_results", "message": f"No {state} PRs found"}
            return {"status": "success", "count": len(all_prs), "pull_requests": all_prs}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [list_github_repos, search_github_issues, get_github_pull_requests]


def _make_jira_tools(email: str):
    """Jira/Confluence tools scoped to the user's own Atlassian credentials."""
    import requests as req

    creds_data = get_app_credentials(email, "jira")

    def _auth():
        if not creds_data:
            return None
        return (creds_data["email"], creds_data["api_token"])

    def _base(path: str) -> str:
        site = creds_data["site"] if creds_data else ""
        return f"https://{site}{path}"

    def search_jira_issues(query: str, project: str = None, status: str = None,
                           max_results: int = 10, days_ago: int = None,
                           author: str = None) -> dict:
        """Search Jira issues by text, project, and status.

        Args:
            query: Text to search in issue summaries and descriptions
            project: Jira project key, e.g. 'ENG' or 'KNO'
            status: Issue status filter, e.g. 'In Progress', 'Done', 'To Do'
            max_results: Max issues to return (default 10)
            days_ago: Only return issues updated in the last N days
            author: Filter by assignee or reporter display name
        """
        if not creds_data:
            return {"status": "error", "message": "Jira not connected — go to Settings to connect Jira."}
        try:
            proj = project or creds_data.get("jira_project", "")
            jql_parts = [f'text ~ "{query}"']
            if proj:
                jql_parts.append(f'project = "{proj}"')
            if status:
                jql_parts.append(f'status = "{status}"')
            if days_ago:
                jql_parts.append(f'updated >= "-{days_ago}d"')
            if author:
                jql_parts.append(f'(assignee = "{author}" OR reporter = "{author}")')
            jql = " AND ".join(jql_parts) + " ORDER BY updated DESC"

            r = req.post(
                _base("/rest/api/3/search/jql"),
                auth=_auth(),
                json={"jql": jql, "maxResults": max_results,
                      "fields": ["summary", "status", "assignee", "priority", "created", "updated"]},
            )
            if not r.ok:
                return {"status": "error", "message": r.text}
            issues = r.json().get("issues", [])
            if not issues:
                return {"status": "no_results", "message": f"No Jira issues found for: {query}"}
            return {"status": "success", "count": len(issues), "issues": [
                {
                    "key": i["key"],
                    "summary": i["fields"]["summary"],
                    "status": i["fields"]["status"]["name"],
                    "assignee": (i["fields"].get("assignee") or {}).get("displayName", "unassigned"),
                    "url": f"https://{creds_data['site']}/browse/{i['key']}",
                }
                for i in issues
            ]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_jira_issue(issue_key: str) -> dict:
        """Get full details of a Jira issue including description and comments.

        Args:
            issue_key: Jira issue key, e.g. 'ENG-123'
        """
        if not creds_data:
            return {"status": "error", "message": "Jira not connected — go to Settings to connect Jira."}
        try:
            r = req.get(_base(f"/rest/api/3/issue/{issue_key}"), auth=_auth())
            if not r.ok:
                return {"status": "error", "message": r.text}
            data = r.json()
            fields = data.get("fields", {})

            # Get comments
            comments_r = req.get(_base(f"/rest/api/3/issue/{issue_key}/comment"), auth=_auth())
            comments = []
            if comments_r.ok:
                for c in comments_r.json().get("comments", [])[:5]:
                    body = c.get("body", {})
                    text = ""
                    if isinstance(body, dict):
                        for block in body.get("content", []):
                            for inner in block.get("content", []):
                                text += inner.get("text", "")
                    comments.append({
                        "author": (c.get("author") or {}).get("displayName", "unknown"),
                        "text": text[:300],
                    })

            return {
                "status": "success",
                "key": issue_key,
                "summary": fields.get("summary"),
                "status": fields.get("status", {}).get("name"),
                "assignee": (fields.get("assignee") or {}).get("displayName", "unassigned"),
                "url": f"https://{creds_data['site']}/browse/{issue_key}",
                "comments": comments,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_confluence_pages(query: str, space: str = None, max_results: int = 5,
                                days_ago: int = None, author: str = None) -> dict:
        """Search Confluence knowledge base pages.

        Args:
            query: Text to search for
            space: Confluence space key to restrict search (optional)
            max_results: Max pages to return (default 5)
            days_ago: Only return pages last modified within N days
            author: Filter by contributor/author display name
        """
        if not creds_data:
            return {"status": "error", "message": "Confluence not connected — go to Settings to connect Jira/Confluence."}
        try:
            sp = space or creds_data.get("confluence_space", "")
            cql = f'type=page AND text ~ "{query}"'
            if sp:
                cql += f' AND space = "{sp}"'
            if days_ago:
                cutoff = _cutoff_dt(days_ago).strftime("%Y-%m-%d")
                cql += f' AND lastModified >= "{cutoff}"'
            if author:
                cql += f' AND contributor.fullname = "{author}"'
            params = {"cql": cql, "limit": max_results, "expand": "version"}
            r = req.get(_base("/wiki/rest/api/content/search"), auth=_auth(), params=params)
            if not r.ok:
                return {"status": "error", "message": r.text}
            results = r.json().get("results", [])
            if not results:
                return {"status": "no_results", "message": f"No Confluence pages found for: {query}"}
            return {"status": "success", "count": len(results), "pages": [
                {"id": p["id"], "title": p["title"],
                 "url": f"https://{creds_data['site']}/wiki{p['_links'].get('webui', '')}"}
                for p in results
            ]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_confluence_page(page_id: str) -> dict:
        """Get the full content of a Confluence page.

        Args:
            page_id: Confluence page ID (from search_confluence_pages results)
        """
        if not creds_data:
            return {"status": "error", "message": "Confluence not connected — go to Settings to connect Jira/Confluence."}
        try:
            r = req.get(_base(f"/wiki/rest/api/content/{page_id}"),
                        auth=_auth(), params={"expand": "body.storage,version"})
            if not r.ok:
                return {"status": "error", "message": r.text}
            data = r.json()
            raw_html = data.get("body", {}).get("storage", {}).get("value", "")
            # Strip HTML tags for readable text
            import re
            text = re.sub(r"<[^>]+>", " ", raw_html)
            text = re.sub(r"\s+", " ", text).strip()[:5000]
            return {
                "status": "success",
                "title": data.get("title"),
                "url": f"https://{creds_data['site']}/wiki{data.get('_links', {}).get('webui', '')}",
                "content": text,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_jira_issue(summary: str, project: str = None,
                          issue_type: str = "Task", description: str = "",
                          assignee: str = None, priority: str = None,
                          labels: list = None) -> dict:
        """Create a new Jira issue.

        Args:
            summary:    Issue title/summary (required)
            project:    Jira project key, e.g. 'ENG'. Defaults to user's saved project.
            issue_type: 'Task', 'Bug', 'Story', 'Epic' (default: Task)
            description: Issue description in plain text
            assignee:   Jira account ID or display name to assign to
            priority:   'Highest', 'High', 'Medium', 'Low', 'Lowest'
            labels:     List of label strings, e.g. ['follow-up', 'crm']
        """
        if not creds_data:
            return {"status": "error", "message": "Jira not connected — go to Settings."}
        try:
            proj = project or creds_data.get("jira_project", "")

            # If no project given or the given one is invalid, auto-discover
            def _list_projects():
                pr = req.get(_base("/rest/api/3/project/search"), auth=_auth(),
                             params={"maxResults": 50})
                if pr.ok:
                    return [p["key"] for p in pr.json().get("values", [])]
                return []

            if not proj:
                available = _list_projects()
                if len(available) == 1:
                    proj = available[0]   # auto-pick if there's exactly one
                else:
                    return {"status": "error",
                            "message": f"No Jira project specified. Available projects: {available}. "
                                        "Pass project='KEY' with one of those keys."}

            fields: dict = {
                "project":   {"key": proj},
                "summary":   summary,
                "issuetype": {"name": issue_type},
            }
            if description:
                fields["description"] = {
                    "type":    "doc",
                    "version": 1,
                    "content": [{"type": "paragraph", "content": [{"type": "text", "text": description}]}],
                }
            if priority:
                fields["priority"] = {"name": priority}
            if labels:
                fields["labels"] = labels

            r = req.post(_base("/rest/api/3/issue"), auth=_auth(), json={"fields": fields})
            if not r.ok:
                # If project key invalid, return available projects to help the user
                if r.status_code in (400, 404):
                    available = _list_projects()
                    return {"status": "error",
                            "message": f"Project '{proj}' not found. "
                                        f"Available projects: {available}. "
                                        "Retry with the correct project key."}
                return {"status": "error", "message": r.text[:300]}

            data = r.json()
            issue_key = data.get("key", "")
            return {
                "status":    "created",
                "key":       issue_key,
                "id":        data.get("id"),
                "url":       f"https://{creds_data['site']}/browse/{issue_key}",
                "summary":   summary,
                "project":   proj,
                "issue_type": issue_type,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def update_jira_issue(issue_key: str, summary: str = None,
                          status: str = None, assignee: str = None,
                          comment: str = None, priority: str = None) -> dict:
        """Update an existing Jira issue — change status, add a comment, reassign, etc.

        Args:
            issue_key: Jira issue key, e.g. 'ENG-42'
            summary:   New summary/title (optional)
            status:    Transition to this status, e.g. 'In Progress', 'Done'
            assignee:  Jira account ID to assign to
            comment:   Add a comment to the issue
            priority:  New priority: 'Highest', 'High', 'Medium', 'Low'
        """
        if not creds_data:
            return {"status": "error", "message": "Jira not connected."}
        try:
            results = []

            # Update fields (summary, assignee, priority)
            update_fields: dict = {}
            if summary:
                update_fields["summary"] = summary
            if priority:
                update_fields["priority"] = {"name": priority}
            if assignee:
                update_fields["assignee"] = {"id": assignee}
            if update_fields:
                r = req.put(_base(f"/rest/api/3/issue/{issue_key}"),
                            auth=_auth(), json={"fields": update_fields})
                results.append({"field_update": r.status_code})

            # Status transition
            if status:
                trans_r = req.get(_base(f"/rest/api/3/issue/{issue_key}/transitions"), auth=_auth())
                transitions = trans_r.json().get("transitions", []) if trans_r.ok else []
                # Exact match first, then fuzzy (contains)
                match = next((t for t in transitions
                              if t["name"].lower() == status.lower()), None)
                if not match:
                    match = next((t for t in transitions
                                  if status.lower() in t["name"].lower()
                                  or t["name"].lower() in status.lower()), None)
                if match:
                    tr = req.post(_base(f"/rest/api/3/issue/{issue_key}/transitions"),
                                  auth=_auth(), json={"transition": {"id": match["id"]}})
                    results.append({"transition": tr.status_code, "to": match["name"]})
                else:
                    available = [t["name"] for t in transitions]
                    results.append({"transition_error": f"Status '{status}' not found. "
                                    f"Available transitions: {available}. Use one of these exact names."})

            # Add comment
            if comment:
                cr = req.post(_base(f"/rest/api/3/issue/{issue_key}/comment"),
                              auth=_auth(), json={
                                  "body": {
                                      "type": "doc", "version": 1,
                                      "content": [{"type": "paragraph", "content": [{"type": "text", "text": comment}]}],
                                  }
                              })
                results.append({"comment": cr.status_code})

            return {
                "status":    "updated",
                "key":       issue_key,
                "url":       f"https://{creds_data['site']}/browse/{issue_key}",
                "actions":   results,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [search_jira_issues, get_jira_issue, search_confluence_pages,
            get_confluence_page, create_jira_issue, update_jira_issue]


def _make_zoho_tools(email: str):
    """Zoho CRM tools scoped to the user's own credentials."""
    import requests as req

    creds_data = get_app_credentials(email, "zoho")
    _token_store = {"access_token": ""}

    def _refresh():
        if not creds_data:
            return ""
        # Use stored accounts domain (default zoho.in); handles .com, .eu, etc.
        accts_domain = creds_data.get("api_domain", "https://accounts.zoho.in")
        r = req.post(
            f"{accts_domain}/oauth/v2/token",
            data={
                "refresh_token": creds_data["refresh_token"],
                "client_id":     creds_data["client_id"],
                "client_secret": creds_data["client_secret"],
                "grant_type":    "refresh_token",
            },
        )
        token = r.json().get("access_token", "")
        if not token:
            import logging as _log
            _log.getLogger(__name__).error(
                "Zoho token refresh failed: %s", r.text[:200])
        _token_store["access_token"] = token
        return token

    def _headers():
        if not _token_store["access_token"]:
            _refresh()
        return {"Authorization": f"Zoho-oauthtoken {_token_store['access_token']}"}

    def _get(url, params=None):
        resp = req.get(url, headers=_headers(), params=params)
        if resp.status_code == 401:
            _refresh()
            resp = req.get(url, headers=_headers(), params=params)
        return resp

    # Use API domain from stored creds (set at connect-time); fall back to .in
    _api_crm  = (creds_data or {}).get("api_domain_crm", "https://www.zohoapis.in")
    BASE      = f"{_api_crm}/crm/v2"
    # Org ID stored at connect-time — used to build deep links without extra API calls
    _org_id   = (creds_data or {}).get("org_id", "")
    _web_base = _api_crm.replace("www.zohoapis", "crm.zoho").replace("/crm/v2", "")

    def _deal_url(deal_id: str) -> str:
        if _org_id:
            return f"{_web_base}/crm/org{_org_id}/tab/Potentials/{deal_id}"
        return f"{_web_base}/crm/tab/Potentials/{deal_id}"

    def _contact_url(contact_id: str) -> str:
        if _org_id:
            return f"{_web_base}/crm/org{_org_id}/tab/Contacts/{contact_id}"
        return f"{_web_base}/crm/tab/Contacts/{contact_id}"

    def search_zoho_contacts(query: str) -> dict:
        """Search Zoho CRM contacts by name or email.

        Args:
            query: Name or email to search for, e.g. 'Alice' or 'alice@acme.com'
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings to connect Zoho."}
        try:
            resp = _get(f"{BASE}/Contacts/search",
                        params={"word": query, "fields": "First_Name,Last_Name,Email,Phone"})
            if resp.status_code == 204:
                return {"status": "no_results", "message": f"No contacts found for: {query}"}
            if not resp.ok:
                return {"status": "error", "message": resp.text}
            contacts = [
                {
                    "id":         c.get("id"),
                    "first_name": c.get("First_Name", ""),
                    "last_name":  c.get("Last_Name", ""),
                    "email":      c.get("Email", ""),
                    "phone":      c.get("Phone", ""),
                    "url":        _contact_url(c.get("id", "")),
                }
                for c in resp.json().get("data", [])
            ]
            return {"status": "success", "count": len(contacts), "contacts": contacts}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_zoho_deals(stage: str = None, days_ago: int = None,
                          owner: str = None, name: str = None) -> dict:
        """List Zoho CRM deals, optionally filtered by name, stage, recency, or owner.

        Args:
            name:  Search by deal name (substring, case-insensitive), e.g. 'Acme'
            stage: Deal stage to filter by, e.g. 'Qualification', 'Closed Won'.
                   Pass None to return all deals.
            days_ago: Only return deals modified in the last N days
            owner: Filter by deal owner/rep name, e.g. 'Alice'
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings to connect Zoho."}
        try:
            fields = "Deal_Name,Amount,Stage,Closing_Date,Modified_Time,Owner"
            if name:
                # Zoho word_match search: searches across all text fields
                resp = _get(f"{BASE}/Deals/search",
                            params={"word": name, "fields": fields})
            elif stage:
                resp = _get(f"{BASE}/Deals/search",
                            params={"criteria": f"Stage:equals:{stage}", "fields": fields})
            else:
                resp = _get(f"{BASE}/Deals", params={"fields": fields})
            if resp.status_code == 204:
                return {"status": "no_results", "message": f"No deals found" + (f" in stage: {stage}" if stage else "")}
            if not resp.ok:
                return {"status": "error", "message": f"Zoho API error {resp.status_code}: {resp.text[:300]}"}

            ts_cutoff = _cutoff_dt(days_ago)
            deals = []
            for d in resp.json().get("data", []):
                # days_ago filter
                if ts_cutoff:
                    mod = d.get("Modified_Time", "")
                    try:
                        mod_dt = datetime.strptime(mod[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
                        if mod_dt < ts_cutoff:
                            continue
                    except (ValueError, TypeError):
                        pass
                # owner filter
                deal_owner = (d.get("Owner") or {}).get("name", "")
                if owner and owner.lower() not in deal_owner.lower():
                    continue
                deals.append({
                    "id":           d.get("id"),
                    "deal_name":    d.get("Deal_Name", ""),
                    "amount":       d.get("Amount"),
                    "stage":        d.get("Stage", ""),
                    "closing_date": d.get("Closing_Date", ""),
                    "owner":        deal_owner,
                    "url":          _deal_url(d.get("id", "")),
                })
            if not deals:
                return {"status": "no_results", "message": "No deals matched the filters."}
            return {"status": "success", "count": len(deals), "deals": deals}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_zoho_contact(contact_id: str) -> dict:
        """Get full details of a single Zoho CRM contact by ID.

        Args:
            contact_id: The Zoho CRM contact ID (from search_zoho_contacts results)
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings to connect Zoho."}
        try:
            resp = _get(f"{BASE}/Contacts/{contact_id}")
            if resp.status_code == 204:
                return {"status": "no_results", "message": f"Contact not found: {contact_id}"}
            if not resp.ok:
                return {"status": "error", "message": resp.text}
            data = resp.json().get("data", [])
            if not data:
                return {"status": "no_results", "message": f"Contact not found: {contact_id}"}
            return {"status": "success", "contact": data[0]}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _post(url, data):
        resp = req.post(url, headers=_headers(), json=data)
        if resp.status_code == 401:
            _refresh()
            resp = req.post(url, headers=_headers(), json=data)
        return resp

    def _put(url, data):
        resp = req.put(url, headers=_headers(), json=data)
        if resp.status_code == 401:
            _refresh()
            resp = req.put(url, headers=_headers(), json=data)
        return resp

    def update_zoho_deal(deal_id: str, stage: str = None,
                         amount: float = None, closing_date: str = None,
                         description: str = None) -> dict:
        """Update fields on an existing Zoho CRM deal.

        Args:
            deal_id:      Zoho deal record ID (from search_zoho_deals results)
            stage:        New pipeline stage, e.g. 'Proposal/Price Quote', 'Closed Won'
            amount:       Updated deal amount as a number, e.g. 95000
            closing_date: New closing date in YYYY-MM-DD format
            description:  Notes to add to the deal description
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings."}
        try:
            update: dict = {"id": deal_id}
            if stage:
                update["Stage"] = stage
            if amount is not None:
                update["Amount"] = amount
            if closing_date:
                update["Closing_Date"] = closing_date
            if description:
                update["Description"] = description

            r = _put(f"{BASE}/Deals", {"data": [update]})
            if not r.ok:
                return {"status": "error", "message": r.text[:300]}

            resp_data = r.json().get("data", [{}])[0]
            return {
                "status":  "updated",
                "deal_id": deal_id,
                "url":     _deal_url(deal_id),
                "code":    resp_data.get("code"),
                "message": resp_data.get("message", ""),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def create_zoho_activity(deal_id: str, subject: str,
                              activity_type: str = "Call",
                              description: str = "", due_date: str = None) -> dict:
        """Log a follow-up activity (call, task, or meeting) as a Note on a Zoho deal.

        Creates a CRM Note linked to the deal — works with standard CRM OAuth scopes.
        The note captures the activity type, subject, date, and any description.

        Args:
            deal_id:       Zoho deal record ID to attach the activity to
            subject:       Activity subject/title
            activity_type: 'Call', 'Task', 'Meeting', or 'Email' (default: Call)
            description:   Notes about the activity
            due_date:      Due date in YYYY-MM-DD format (defaults to today)
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings."}
        try:
            from datetime import date
            today = due_date or date.today().isoformat()

            # Zoho's /Calls, /Tasks, /Events require additional OAuth scopes
            # (ZohoCRM.modules.calls.ALL, etc.) that may not be on the user's token.
            # /Notes is covered by standard deal/contact scopes and serves the same
            # purpose as a logged activity — visible on the deal timeline.
            note_content = f"[{activity_type.upper()}] {subject}"
            if description:
                note_content += f"\n\n{description}"
            note_content += f"\n\nDue: {today}"

            payload = {
                "Note_Title":   subject,
                "Note_Content": note_content,
                "Parent_Id":    {"id": deal_id},
                "$se_module":   "Deals",
            }

            r = _post(f"{BASE}/Notes", {"data": [payload]})
            if not r.ok:
                return {
                    "status":  "error",
                    "message": f"Zoho Notes API {r.status_code}: {r.text[:300]}",
                }

            resp_data   = r.json().get("data", [{}])[0]
            note_id     = resp_data.get("details", {}).get("id", "")
            return {
                "status":      "created",
                "activity_id": note_id,
                "deal_id":     deal_id,
                "subject":     subject,
                "type":        activity_type,
                "module":      "Notes",
                "url":         _deal_url(deal_id),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [search_zoho_contacts, search_zoho_deals, get_zoho_contact,
            update_zoho_deal, create_zoho_activity]


# ── Agent runner ──────────────────────────────────────────────────────────────

def _make_rag_tools() -> list:
    """Return two knowledge-base tools:
      - search_knowledge_base  → topic/semantic search (requires a query keyword)
      - browse_knowledge_base  → filter-only browsing by date/author/source (no keyword needed)
    """
    try:
        from kno.rag_connector import search_knowledge_base as _rag_search

        def _format(results: list, label: str) -> dict:
            if not results:
                return {"status": "no_results",
                        "message": f"No knowledge base articles found for: {label}"}
            return {
                "status": "success",
                "count": len(results),
                "passages": [
                    {
                        "title":   r["title"],
                        "excerpt": r["text"][:400],
                        "url":     r["url"],
                        "author":  r["author"],
                        "date":    r["date"],
                        "source":  r["source"],
                    }
                    for r in results
                ],
            }

        def search_knowledge_base(query: str,
                                   days_ago: int = None,
                                   author: str = None,
                                   source_type: str = None) -> dict:
            """Search the company knowledge base by TOPIC using semantic similarity.

            Use when the user names a specific subject, e.g. "onboarding", "deployment guide",
            "API rate limits". Returns the most relevant passages with source citations.

            Args:
                query:       Required topic or question, e.g. 'how do we onboard engineers'
                days_ago:    Also restrict to docs modified within the last N days
                author:      Also restrict to docs by this author
                source_type: Also restrict to one source ('confluence', 'github', etc.)
            """
            results = _rag_search(query, top_k=5,
                                  days_ago=days_ago, author=author,
                                  source_type=source_type)
            return _format(results, query)

        def browse_knowledge_base(days_ago: int = None,
                                   author: str = None,
                                   source_type: str = None) -> dict:
            """Browse the knowledge base by date, author, or source — NO keyword needed.

            Call this immediately (without asking for a keyword) whenever the user asks:
              - "show recent docs" / "updated last 30 days" / "this month"
              - "docs written by Alice" / "pages by Dhanapal"
              - "all Confluence pages"
              - any combination of the above

            Results are sorted newest-first.

            Args:
                days_ago:    Return docs modified/ingested within the last N days (e.g. 30, 90)
                author:      Filter by author name — case-insensitive substring match
                source_type: Filter by source — 'confluence', 'github', 'drive', etc.
            """
            results = _rag_search("", top_k=10,
                                  days_ago=days_ago, author=author,
                                  source_type=source_type)
            label = f"filters: days_ago={days_ago}, author={author}, source_type={source_type}"
            return _format(results, label)

        return [search_knowledge_base, browse_knowledge_base]
    except Exception:
        return []


def _build_tools(email: str) -> list:
    tools = [
        # Memory tools come first so they're always available
        PreloadMemoryTool(),
        LoadMemoryTool(),
    ]
    # RAG knowledge base — two tools: search (topic) + browse (filter-only)
    tools += _make_rag_tools()

    tools += _make_gmail_tool(email)     # [search_gmail, send_gmail]
    tools += _make_drive_tools(email)    # [search_drive, read_drive_file_multimodal]
    tools += _make_slack_tools(email)    # [get_slack_activity, search_slack_messages, post_slack_message]
    tools += _make_github_tools(email)   # [list_github_repos, search_github_issues, get_github_pull_requests]
    tools += _make_jira_tools(email)     # [search_jira_issues, get_jira_issue, search_confluence_pages, get_confluence_page, create_jira_issue, update_jira_issue]
    tools += _make_zoho_tools(email)     # [search_zoho_contacts, search_zoho_deals, get_zoho_contact, update_zoho_deal, create_zoho_activity]
    return tools


async def run_user_query(email: str, message: str, session_id: str = None) -> str:
    """Run a query scoped entirely to one user's credentials.

    Args:
        email:      The authenticated user's email (used as user_id).
        message:    The user's query text.
        session_id: Optional — resume an existing session. If None, creates a new one.

    Returns:
        Tuple of (response_text, session_id).
    """
    tools = _build_tools(email)

    agent = Agent(
        model="gemini-2.5-flash",
        name="kno_agent",
        description="kno.ai — AI assistant for company knowledge",
        instruction=_INSTRUCTION,
        tools=tools,
    )

    # ADK App.name must start with a letter — prefix numeric engine IDs.
    app_name = f"kno-{_AGENT_ENGINE_ID}" if (_USE_VERTEX and _AGENT_ENGINE_ID) else "kno"

    runner = Runner(
        agent=agent,
        app_name=app_name,
        session_service=_session_service,
        memory_service=_memory_service,
    )

    # Reuse existing session or create a fresh one
    if session_id:
        try:
            session = await _session_service.get_session(
                app_name=app_name, user_id=email, session_id=session_id
            )
        except Exception:
            session = None
    else:
        session = None

    if session is None:
        session = await _session_service.create_session(
            app_name=app_name, user_id=email
        )

    response_text = ""
    async for event in runner.run_async(
        user_id=email,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=message)],
        ),
    ):
        if event.is_final_response() and event.content and event.content.parts:
            response_text = "".join(p.text for p in event.content.parts if hasattr(p, "text"))

    return response_text, session.id
