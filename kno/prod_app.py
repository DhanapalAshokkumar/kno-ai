"""
kno.ai — Production FastAPI app.
Multi-tenant: every user logs in via Cloudflare Access, connects their own apps,
and chats with an agent scoped entirely to their credentials.
"""
import os
import secrets
import urllib.parse
from typing import Optional

import requests as req
from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from kno.auth import get_current_user
from kno.user_store import (
    get_or_create_user,
    get_connected_apps,
    get_app_credentials,
    store_app_credentials,
    disconnect_app,
)
from kno.per_user_agent import run_user_query

# ── App setup ─────────────────────────────────────────────────────────────────

app = FastAPI(title="kno.ai", docs_url=None, redoc_url=None)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Auth dependency ───────────────────────────────────────────────────────────

def current_user(request: Request) -> str:
    email = get_current_user(request)
    if not email:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return email


# ── Google OAuth config ───────────────────────────────────────────────────────

GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "")
GOOGLE_REDIRECT_URI  = os.environ.get(
    "GOOGLE_REDIRECT_URI",
    "https://kno.xnukernel.com/auth/google/callback",
)
GOOGLE_SCOPES = " ".join([
    "openid",
    "email",
    "profile",
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/drive.readonly",
])

# Simple in-memory state store (survives only within one instance; good enough
# for short-lived OAuth flows; use Firestore for multi-instance if needed)
_oauth_states: dict[str, str] = {}   # state -> email


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get("/me")
def get_me(email: str = Depends(current_user)):
    """Return current user profile and connected apps."""
    user = get_or_create_user(email)
    connected = get_connected_apps(email)
    return {
        "email": email,
        "connected_apps": connected,
        "created_at": user.get("created_at"),
    }


# ── Google OAuth (Gmail + Drive) ──────────────────────────────────────────────

@app.get("/auth/google")
def auth_google(request: Request, email: str = Depends(current_user)):
    """Redirect user to Google's consent screen."""
    state = secrets.token_urlsafe(32)
    _oauth_states[state] = email

    params = {
        "client_id": GOOGLE_CLIENT_ID,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "response_type": "code",
        "scope": GOOGLE_SCOPES,
        "access_type": "offline",
        "prompt": "consent",   # force refresh_token every time
        "state": state,
    }
    url = "https://accounts.google.com/o/oauth2/v2/auth?" + urllib.parse.urlencode(params)
    return RedirectResponse(url)


@app.get("/auth/google/callback")
def auth_google_callback(code: str, state: str, request: Request):
    """Exchange auth code for refresh token and store it."""
    email = _oauth_states.pop(state, None)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    # Exchange code for tokens
    token_resp = req.post("https://oauth2.googleapis.com/token", data={
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    })
    token_data = token_resp.json()
    if "refresh_token" not in token_data:
        raise HTTPException(
            status_code=400,
            detail=f"Google did not return a refresh token: {token_data.get('error_description', token_data)}"
        )

    store_app_credentials(email, "gmail", {
        "refresh_token": token_data["refresh_token"],
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
    })

    return RedirectResponse("/?connected=gmail")


# ── GitHub PAT connection ─────────────────────────────────────────────────────

class GithubConnectRequest(BaseModel):
    token: str        # Personal Access Token (classic, repo + read:org scope)
    owner: str        # GitHub org or username, e.g. "kno-ai"


@app.post("/connect/github")
def connect_github(body: GithubConnectRequest, email: str = Depends(current_user)):
    """Store a GitHub PAT for the current user."""
    # Quick validation
    r = req.get(
        "https://api.github.com/user",
        headers={"Authorization": f"Bearer {body.token}", "Accept": "application/vnd.github+json"},
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid GitHub token — check scopes and try again")

    store_app_credentials(email, "github", {"token": body.token, "owner": body.owner})
    return {"status": "connected", "app": "github"}


# ── Slack OAuth ───────────────────────────────────────────────────────────────

SLACK_CLIENT_ID     = os.environ.get("SLACK_CLIENT_ID", "")
SLACK_CLIENT_SECRET = os.environ.get("SLACK_CLIENT_SECRET", "")
SLACK_REDIRECT_URI  = os.environ.get(
    "SLACK_REDIRECT_URI",
    "https://kno.xnukernel.com/auth/slack/callback",
)
_slack_states: dict[str, str] = {}


@app.get("/auth/slack")
def auth_slack(request: Request, email: str = Depends(current_user)):
    """Redirect user to Slack OAuth consent screen."""
    state = secrets.token_urlsafe(32)
    _slack_states[state] = email

    params = {
        "client_id": SLACK_CLIENT_ID,
        "redirect_uri": SLACK_REDIRECT_URI,
        "scope": "channels:read,channels:history,groups:read,groups:history",
        "user_scope": "search:read,channels:read,channels:history,groups:read,groups:history",
        "state": state,
    }
    url = "https://slack.com/oauth/v2/authorize?" + urllib.parse.urlencode(params)
    return RedirectResponse(url)


@app.get("/auth/slack/callback")
def auth_slack_callback(code: str, state: str):
    """Exchange Slack auth code for bot token."""
    email = _slack_states.pop(state, None)
    if not email:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    r = req.post("https://slack.com/api/oauth.v2.access", data={
        "code": code,
        "client_id": SLACK_CLIENT_ID,
        "client_secret": SLACK_CLIENT_SECRET,
        "redirect_uri": SLACK_REDIRECT_URI,
    })
    data = r.json()
    if not data.get("ok"):
        raise HTTPException(status_code=400, detail=f"Slack OAuth failed: {data.get('error')}")

    store_app_credentials(email, "slack", {
        "bot_token": data["access_token"],
        "user_token": data.get("authed_user", {}).get("access_token", ""),
        "team_id": data.get("team", {}).get("id", ""),
        "team_name": data.get("team", {}).get("name", ""),
    })

    return RedirectResponse("/?connected=slack")


# ── Jira / Confluence connection ───────────────────────────────────────────────

class JiraConnectRequest(BaseModel):
    email: str         # Atlassian account email
    api_token: str     # Atlassian API token from id.atlassian.com
    site: str          # e.g. "mycompany.atlassian.net"
    jira_project: Optional[str] = None    # default Jira project key
    confluence_space: Optional[str] = None  # default Confluence space key


@app.post("/connect/jira")
def connect_jira(body: JiraConnectRequest, caller: str = Depends(current_user)):
    """Store Jira/Confluence credentials for the current user."""
    # Quick validation — hit the Jira myself endpoint
    r = req.get(
        f"https://{body.site}/rest/api/3/myself",
        auth=(body.email, body.api_token),
    )
    if r.status_code != 200:
        raise HTTPException(status_code=400, detail="Invalid Jira credentials — check email, token, and site")

    store_app_credentials(caller, "jira", {
        "email": body.email,
        "api_token": body.api_token,
        "site": body.site,
        "jira_project": body.jira_project or "",
        "confluence_space": body.confluence_space or "",
    })
    return {"status": "connected", "app": "jira"}


# ── Zoho CRM connection ───────────────────────────────────────────────────────

class ZohoConnectRequest(BaseModel):
    client_id: str
    client_secret: str
    refresh_token: str
    api_domain: Optional[str] = "https://accounts.zoho.in"


@app.post("/connect/zoho")
def connect_zoho(body: ZohoConnectRequest, email: str = Depends(current_user)):
    """Store Zoho CRM credentials for the current user."""
    # Validate by refreshing
    r = req.post(f"{body.api_domain}/oauth/v2/token", data={
        "refresh_token": body.refresh_token,
        "client_id": body.client_id,
        "client_secret": body.client_secret,
        "grant_type": "refresh_token",
    })
    data = r.json()
    if "access_token" not in data:
        raise HTTPException(status_code=400, detail=f"Invalid Zoho credentials: {data.get('error', data)}")

    store_app_credentials(email, "zoho", {
        "client_id": body.client_id,
        "client_secret": body.client_secret,
        "refresh_token": body.refresh_token,
        "api_domain": body.api_domain,
    })
    return {"status": "connected", "app": "zoho"}


# ── Disconnect ─────────────────────────────────────────────────────────────────

@app.delete("/disconnect/{app_name}")
def disconnect(app_name: str, email: str = Depends(current_user)):
    """Remove a connected app for the current user."""
    disconnect_app(email, app_name)
    return {"status": "disconnected", "app": app_name}


# ── Knowledge base ingestion ──────────────────────────────────────────────────

@app.post("/admin/ingest/confluence")
async def ingest_confluence(email: str = Depends(current_user)):
    """Ingest all Confluence pages into the RAG knowledge base (synchronous)."""
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only")

    import re
    from kno.rag_connector import ingest_text

    creds = get_app_credentials(email, "jira")
    if not creds:
        raise HTTPException(status_code=400, detail="Jira/Confluence not connected")

    auth = (creds["email"], creds["api_token"])
    base = f"https://{creds['site']}"

    spaces_r = req.get(f"{base}/wiki/rest/api/space", auth=auth,
                       params={"limit": 50}, timeout=15)
    spaces = spaces_r.json().get("results", []) if spaces_r.ok else []

    ingested, failed, errors = 0, 0, []
    for space in spaces:
        key = space["key"]
        pages_r = req.get(
            f"{base}/wiki/rest/api/content", auth=auth, timeout=15,
            params={"spaceKey": key, "type": "page", "limit": 50,
                    "expand": "body.storage,version,history.lastUpdated"},
        )
        if not pages_r.ok:
            continue
        for page in pages_r.json().get("results", []):
            title    = page.get("title", "")
            url      = f"{base}/wiki{page.get('_links', {}).get('webui', '')}"
            updated  = page.get("version", {}).get("when", "")[:10]
            author   = page.get("history", {}).get("lastUpdated", {}).get("by", {}).get("displayName", "")
            raw_html = page.get("body", {}).get("storage", {}).get("value", "")
            text = re.sub(r"<[^>]+>", " ", raw_html)
            text = re.sub(r"\s+", " ", text).strip()
            if not text:
                continue
            try:
                ok = ingest_text(
                    text=text, display_name=title,
                    source="confluence", url=url,
                    author=author, modified_date=updated,
                )
                if ok:
                    ingested += 1
                else:
                    failed += 1
                    errors.append(f"upload failed: {title}")
            except Exception as e:
                failed += 1
                errors.append(f"{title}: {str(e)[:100]}")

    return {
        "status": "done",
        "ingested": ingested,
        "failed": failed,
        "spaces": len(spaces),
        "errors": errors[:5],   # first 5 errors for diagnosis
    }


# ── Admin ─────────────────────────────────────────────────────────────────────

ADMIN_EMAILS = {"dhana19.ece@gmail.com"}   # expand as needed

@app.get("/admin/users", response_class=HTMLResponse)
def admin_users(email: str = Depends(current_user)):
    """Admin dashboard — shows all users and their connected apps."""
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access only")

    from google.cloud import firestore
    db = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516"))
    users = list(db.collection("users").stream())

    rows = ""
    for doc in users:
        d = doc.to_dict()
        apps = list(d.get("connected_apps", {}).keys())
        joined = d.get("created_at", "")[:10]
        app_badges = "".join(
            f'<span style="background:#34d399;color:#000;padding:2px 8px;border-radius:10px;font-size:12px;margin:2px;display:inline-block">{a}</span>'
            for a in apps
        ) or '<span style="color:#888">none connected</span>'
        status_dot = "🟢" if apps else "🟡"
        rows += f"""<tr>
            <td style="padding:12px 16px">{status_dot}</td>
            <td style="padding:12px 16px;font-family:monospace">{d.get('email','')}</td>
            <td style="padding:12px 16px">{app_badges}</td>
            <td style="padding:12px 16px;color:#888">{joined}</td>
        </tr>"""

    return HTMLResponse(f"""<!DOCTYPE html>
<html><head><title>kno.ai Admin</title>
<style>body{{font-family:system-ui,sans-serif;background:#0f0f13;color:#e8e8f0;margin:0;padding:32px}}
h1{{font-size:22px;margin-bottom:4px}}p{{color:#888;margin-bottom:24px}}
table{{width:100%;border-collapse:collapse;background:#18181f;border-radius:12px;overflow:hidden}}
th{{text-align:left;padding:12px 16px;background:#22222c;color:#888;font-size:12px;text-transform:uppercase;letter-spacing:.5px}}
tr:not(:last-child){{border-bottom:1px solid #2e2e3a}}
</style></head><body>
<h1>✦ kno.ai — Users</h1>
<p>{len(users)} total · {sum(1 for d in [doc.to_dict() for doc in users] if d.get('connected_apps'))} active</p>
<table><thead><tr><th></th><th>Email</th><th>Connected Apps</th><th>Joined</th></tr></thead>
<tbody>{rows}</tbody></table>
<p style="margin-top:16px;font-size:12px;color:#444">Logged in as {email} · <a href="/" style="color:#6c63ff">← Back to kno</a></p>
</body></html>""")


# ── KB stats ─────────────────────────────────────────────────────────────────

@app.get("/admin/kb/stats")
def kb_stats(email: str = Depends(current_user)):
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin only")
    from kno.rag_connector import get_corpus_stats
    return get_corpus_stats()


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None   # resume existing session


@app.post("/chat")
async def chat(body: ChatRequest, email: str = Depends(current_user)):
    """Run a query against the current user's connected tools.

    Returns the agent response AND a session_id the client should send back
    on the next turn to maintain conversation continuity.
    """
    if not body.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    response, session_id = await run_user_query(email, body.message, body.session_id)
    return {"response": response, "session_id": session_id}


# ── Frontend ──────────────────────────────────────────────────────────────────

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(STATIC_DIR, exist_ok=True)

# Mount static files (JS, CSS, images)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
@app.get("/{full_path:path}", response_class=HTMLResponse)
def serve_frontend(full_path: str = ""):
    """Serve the SPA for all non-API routes."""
    index = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index):
        with open(index) as f:
            return HTMLResponse(f.read())
    # Fallback if static hasn't been built yet
    return HTMLResponse("<h1>kno.ai</h1><p>Frontend not found. Deploy with static/index.html</p>", status_code=200)
