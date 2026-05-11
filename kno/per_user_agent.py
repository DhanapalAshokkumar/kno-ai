"""
Multi-tenant agent runner.
Each user's query executes with ONLY their own credentials — never another user's.
Vertex AI Session Service gives persistent sessions across Cloud Run instances.
Vertex AI Memory Bank gives long-term memory across conversations.
"""
import os
import base64
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
    "https://www.googleapis.com/auth/gmail.readonly",
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
  [2] Zoho CRM: Deal Name — Stage: Negotiation | Amount: $70,000 | Closing: Apr 29
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
- For documents/knowledge base: use search_knowledge_base first.
- For emails: use search_gmail. Show subject, sender, date — NOT full body.
- For files: use search_drive.
- For team chat: NEVER ask the user for a Slack search keyword. If the user asks about recent Slack activity, discussions, or "what's happening in Slack" WITHOUT specifying a topic, IMMEDIATELY call get_recent_slack_messages() — do not ask for clarification. Only use search_slack_messages when the user explicitly names a topic (e.g. "search Slack for budget").
- For tasks/bugs: use search_jira_issues.
- For code/PRs: use list_github_repos, search_github_issues, or get_github_pull_requests.
- For CRM: use search_zoho_contacts or search_zoho_deals.

## Formatting
- Be concise. Use bullet points. Employees are busy.
- For emails: show subject + sender + 1-sentence summary ONLY — never paste full body.
- For deals: show name, stage, amount, closing date — nothing else.
- Keep total response under 400 words unless the user asks for detail.

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

    def search_gmail(query: str, max_results: int = 5) -> dict:
        """Search Gmail for emails and threads matching a query.

        Args:
            query: Gmail search query, e.g. 'from:boss@company.com budget'
            max_results: Max threads to return (default 5)
        """
        if not creds_data:
            return {"status": "error", "message": "Gmail not connected — go to Settings to connect your Gmail."}
        try:
            svc = build("gmail", "v1", credentials=_google_creds(creds_data))
            resp = svc.users().threads().list(userId="me", q=query, maxResults=max_results).execute()
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

    return search_gmail


def _make_drive_tool(email: str):
    creds_data = get_app_credentials(email, "gmail")  # Gmail and Drive share OAuth

    def search_drive(query: str, max_results: int = 5) -> dict:
        """Search Google Drive for files matching a query.

        Args:
            query: Keyword search, e.g. 'Q3 roadmap deck'
            max_results: Max files to return (default 5)
        """
        if not creds_data:
            return {"status": "error", "message": "Google Drive not connected — go to Settings to connect your Google account."}
        try:
            svc = build("drive", "v3", credentials=_google_creds(creds_data))
            resp = svc.files().list(
                q=f"fullText contains '{query}' and trashed=false",
                pageSize=max_results,
                fields="files(id,name,mimeType,modifiedTime,webViewLink,owners)",
            ).execute()
            files = resp.get("files", [])
            if not files:
                return {"status": "no_results", "message": f"No files found for: {query}"}

            results = []
            for f in files:
                file_entry = {
                    "id": f["id"],
                    "name": f.get("name"),
                    "modified": f.get("modifiedTime"),
                    "url": f.get("webViewLink"),
                    "owner": (f.get("owners") or [{}])[0].get("displayName", ""),
                    "snippet": "",
                }
                # Try to export a plain-text snippet for Docs/Sheets/Slides
                mime = f.get("mimeType", "")
                if "google-apps.document" in mime or "google-apps.presentation" in mime or "google-apps.spreadsheet" in mime:
                    try:
                        export_resp = svc.files().export(
                            fileId=f["id"], mimeType="text/plain"
                        ).execute()
                        if isinstance(export_resp, bytes):
                            text = export_resp.decode("utf-8", errors="replace")
                        else:
                            text = str(export_resp)
                        import re as _re
                        text = _re.sub(r"\s+", " ", text).strip()
                        file_entry["snippet"] = text[:400]
                    except Exception:
                        pass
                results.append(file_entry)

            return {"status": "success", "count": len(results), "files": results}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return search_drive


def _make_slack_tools(email: str):
    from slack_sdk import WebClient
    from slack_sdk.errors import SlackApiError

    creds_data = get_app_credentials(email, "slack")

    def search_slack_messages(query: str, count: int = 10) -> dict:
        """Search all Slack messages and channels for a keyword using full-text search.

        Uses the Slack Search API — no need for the bot to be invited to channels.
        Returns matching messages with channel name, author, and timestamp.

        Args:
            query: Keyword or phrase to search for, e.g. 'deployment issue' or 'budget approval'
            count: Max messages to return (default 10)
        """
        if not creds_data:
            return {"status": "error", "message": "Slack not connected — go to Settings to connect Slack."}
        try:
            # Prefer user token (has search:read scope) over bot token
            user_token = creds_data.get("user_token", "")
            bot_token  = creds_data.get("bot_token", "")
            token = user_token or bot_token
            if not token:
                return {"status": "error", "message": "Slack token missing — reconnect Slack in Settings."}

            cli = WebClient(token=token)
            resp = cli.search_messages(query=query, count=count, sort="timestamp", sort_dir="desc")
            matches = resp.get("messages", {}).get("matches", [])
            if not matches:
                return {"status": "no_results", "message": f"No Slack messages found for: {query}"}

            results = []
            for m in matches:
                results.append({
                    "channel": m.get("channel", {}).get("name", "unknown"),
                    "author": m.get("username", m.get("user", "unknown")),
                    "text": m.get("text", "")[:400],
                    "timestamp": m.get("ts", ""),
                    "permalink": m.get("permalink", ""),
                })
            return {"status": "success", "count": len(results), "messages": results}
        except SlackApiError as e:
            # Fall back to bot-token channel scan if search:read not granted
            if "missing_scope" in str(e) or "not_allowed_token_type" in str(e):
                return _fallback_slack_scan(creds_data, query)
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_recent_slack_messages(channels: int = 5, messages_per_channel: int = 8) -> dict:
        """Fetch the most recent messages from Slack without needing a keyword.

        Use this when the user asks "what's happening in Slack?", "any recent updates?",
        or "what has the team been discussing?" — i.e. browsing without a specific topic.

        Args:
            channels: Number of most-active public channels to check (default 5)
            messages_per_channel: Messages per channel (default 8)
        """
        if not creds_data:
            return {"status": "error", "message": "Slack not connected — go to Settings to connect Slack."}
        try:
            # Prefer user token, fall back to bot token
            user_token = creds_data.get("user_token", "")
            bot_token  = creds_data.get("bot_token", "")
            token = user_token or bot_token
            if not token:
                return {"status": "error", "message": "Slack token missing — reconnect Slack in Settings."}

            cli = WebClient(token=token)
            # List public channels sorted by member count (most active first)
            ch_resp = cli.conversations_list(
                types="public_channel,private_channel",
                limit=200,
                exclude_archived=True,
            )
            all_channels = ch_resp.get("channels", [])
            # Sort by member count descending, take top N
            all_channels.sort(key=lambda c: c.get("num_members", 0), reverse=True)
            top_channels = all_channels[:channels]

            results = []
            for ch in top_channels:
                try:
                    hist = cli.conversations_history(channel=ch["id"], limit=messages_per_channel)
                    for msg in hist.get("messages", []):
                        text = msg.get("text", "").strip()
                        if not text or msg.get("subtype"):  # skip system messages
                            continue
                        # Build permalink if possible
                        team_id = creds_data.get("team_id", "")
                        ts_raw  = msg.get("ts", "")
                        permalink = (
                            f"https://slack.com/archives/{ch['id']}/p{ts_raw.replace('.', '')}"
                            if ts_raw else ""
                        )
                        results.append({
                            "channel": ch.get("name", "unknown"),
                            "author": msg.get("user", msg.get("username", "unknown")),
                            "text": text[:400],
                            "timestamp": ts_raw,
                            "permalink": permalink,
                        })
                except SlackApiError:
                    continue  # skip channels the token can't access

            if not results:
                return {"status": "no_results", "message": "No recent Slack messages found."}
            return {"status": "success", "count": len(results), "messages": results}
        except SlackApiError as e:
            return {"status": "error", "message": str(e)}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def _fallback_slack_scan(creds_data: dict, query: str) -> dict:
        """Fallback: manually scan public channels the bot is in."""
        try:
            cli = WebClient(token=creds_data.get("bot_token", ""))
            channels = cli.conversations_list(types="public_channel", limit=50).get("channels", [])
            results = []
            for ch in channels:
                try:
                    history = cli.conversations_history(channel=ch["id"], limit=30)
                    for msg in history.get("messages", []):
                        text = msg.get("text", "")
                        if query.lower() in text.lower():
                            results.append({
                                "channel": ch["name"],
                                "text": text[:400],
                                "timestamp": msg.get("ts", ""),
                                "permalink": "",
                            })
                except SlackApiError:
                    continue
            return {"status": "success", "count": len(results), "messages": results} if results \
                else {"status": "no_results", "message": f"No Slack messages found for: {query}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    return [search_slack_messages, get_recent_slack_messages]


def _make_github_tools(email: str):
    import requests as req

    creds_data = get_app_credentials(email, "github")

    def _headers():
        token = creds_data.get("token", "") if creds_data else ""
        return {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

    def search_github_issues(query: str, repo: str = None, state: str = "open") -> dict:
        """Search GitHub issues by keyword.

        Args:
            query: Text to search in issue titles/bodies
            repo: 'owner/repo' or bare repo name
            state: 'open', 'closed', or 'all'
        """
        if not creds_data:
            return {"status": "error", "message": "GitHub not connected — go to Settings to connect GitHub."}
        try:
            owner = creds_data.get("owner", "")
            q = f"{query} repo:{owner}/{repo or owner} is:issue state:{state}"
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

    def get_github_pull_requests(repo: str = None, state: str = "open") -> dict:
        """List pull requests for a GitHub repo.

        Args:
            repo: bare repo name (e.g. 'kno-ai') or 'owner/repo'. If omitted,
                  lists PRs across ALL repos for the connected owner.
            state: 'open', 'closed', or 'all'
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

            all_prs = []
            for full_repo in repos_to_check[:10]:  # cap at 10 repos
                r = req.get(f"https://api.github.com/repos/{full_repo}/pulls",
                            headers=_headers(), params={"state": state, "per_page": 10})
                if r.ok:
                    for p in r.json():
                        all_prs.append({
                            "repo": full_repo,
                            "number": p["number"], "title": p["title"],
                            "author": p["user"]["login"], "url": p["html_url"]
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

    def search_jira_issues(query: str, project: str = None, status: str = None, max_results: int = 10) -> dict:
        """Search Jira issues by text, project, and status.

        Args:
            query: Text to search in issue summaries and descriptions
            project: Jira project key, e.g. 'ENG' or 'KNO'
            status: Issue status filter, e.g. 'In Progress', 'Done', 'To Do'
            max_results: Max issues to return (default 10)
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

    def search_confluence_pages(query: str, space: str = None, max_results: int = 5) -> dict:
        """Search Confluence knowledge base pages.

        Args:
            query: Text to search for
            space: Confluence space key to restrict search (optional)
            max_results: Max pages to return (default 5)
        """
        if not creds_data:
            return {"status": "error", "message": "Confluence not connected — go to Settings to connect Jira/Confluence."}
        try:
            sp = space or creds_data.get("confluence_space", "")
            params = {"cql": f'type=page AND text ~ "{query}"' + (f' AND space = "{sp}"' if sp else ""),
                      "limit": max_results, "expand": "version"}
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

    return [search_jira_issues, get_jira_issue, search_confluence_pages, get_confluence_page]


def _make_zoho_tools(email: str):
    """Zoho CRM tools scoped to the user's own credentials."""
    import requests as req

    creds_data = get_app_credentials(email, "zoho")
    _token_store = {"access_token": ""}

    def _refresh():
        if not creds_data:
            return ""
        r = req.post(
            "https://accounts.zoho.in/oauth/v2/token",
            data={
                "refresh_token": creds_data["refresh_token"],
                "client_id": creds_data["client_id"],
                "client_secret": creds_data["client_secret"],
                "grant_type": "refresh_token",
            },
        )
        token = r.json().get("access_token", "")
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

    BASE = "https://www.zohoapis.in/crm/v2"

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
                {"id": c.get("id"), "first_name": c.get("First_Name", ""),
                 "last_name": c.get("Last_Name", ""), "email": c.get("Email", ""),
                 "phone": c.get("Phone", "")}
                for c in resp.json().get("data", [])
            ]
            return {"status": "success", "count": len(contacts), "contacts": contacts}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def search_zoho_deals(stage: str = None) -> dict:
        """List Zoho CRM deals, optionally filtered by pipeline stage.

        Args:
            stage: Deal stage to filter by, e.g. 'Qualification', 'Closed Won'.
                   Pass None to return all deals.
        """
        if not creds_data:
            return {"status": "error", "message": "Zoho CRM not connected — go to Settings to connect Zoho."}
        try:
            fields = "Deal_Name,Amount,Stage,Closing_Date"
            if stage:
                resp = _get(f"{BASE}/Deals/search",
                            params={"criteria": f"Stage:equals:{stage}", "fields": fields})
            else:
                resp = _get(f"{BASE}/Deals", params={"fields": fields})
            if resp.status_code == 204:
                return {"status": "no_results", "message": f"No deals found" + (f" in stage: {stage}" if stage else "")}
            if not resp.ok:
                return {"status": "error", "message": resp.text}
            deals = [
                {"id": d.get("id"), "deal_name": d.get("Deal_Name", ""),
                 "amount": d.get("Amount"), "stage": d.get("Stage", ""),
                 "closing_date": d.get("Closing_Date", "")}
                for d in resp.json().get("data", [])
            ]
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

    return [search_zoho_contacts, search_zoho_deals, get_zoho_contact]


# ── Agent runner ──────────────────────────────────────────────────────────────

def _make_rag_tool():
    """Wrap the RAG knowledge base search as an ADK-compatible tool."""
    try:
        from kno.rag_connector import search_knowledge_base as _rag_search

        def search_knowledge_base(query: str) -> dict:
            """Search the company knowledge base (Confluence + Drive documents) for information.

            Use this FIRST for any question about company processes, policies, how-tos,
            product specs, or internal documentation. Returns passages with source citations.

            Args:
                query: Natural language search query, e.g. 'onboarding process for engineers'
            """
            results = _rag_search(query, top_k=5)
            if not results:
                return {"status": "no_results", "message": f"No knowledge base articles found for: {query}"}
            return {
                "status": "success",
                "count": len(results),
                "passages": [
                    {
                        "title": r["title"],
                        "excerpt": r["text"][:400],
                        "url": r["url"],
                        "author": r["author"],
                        "date": r["date"],
                        "source": r["source"],
                    }
                    for r in results
                ],
            }

        return search_knowledge_base
    except Exception:
        return None


def _build_tools(email: str) -> list:
    tools = [
        # Memory tools come first so they're always available
        PreloadMemoryTool(),
        LoadMemoryTool(),
    ]
    # RAG knowledge base (Confluence + Drive semantic search)
    rag_tool = _make_rag_tool()
    if rag_tool:
        tools.append(rag_tool)

    tools.append(_make_gmail_tool(email))
    tools.append(_make_drive_tool(email))
    tools += _make_slack_tools(email)
    tools += _make_github_tools(email)
    tools += _make_jira_tools(email)
    tools += _make_zoho_tools(email)
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

    # VertexAiSessionService requires app_name = reasoning engine ID.
    # InMemorySessionService accepts any string.
    app_name = _AGENT_ENGINE_ID if (_USE_VERTEX and _AGENT_ENGINE_ID) else "kno"

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
