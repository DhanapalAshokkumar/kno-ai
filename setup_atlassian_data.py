import os
import sys
import requests
from requests.auth import HTTPBasicAuth

EMAIL     = os.environ.get("ATLASSIAN_EMAIL", "knoaiworkspace@gmail.com")
API_TOKEN = os.environ.get("ATLASSIAN_API_TOKEN")
SITE      = os.environ.get("ATLASSIAN_SITE", "knoai-dev.atlassian.net")

if not API_TOKEN:
    sys.exit("ERROR: set ATLASSIAN_API_TOKEN environment variable before running this script.")

JIRA_BASE = f"https://{SITE}/rest/api/3"
CONFLUENCE_BASE = f"https://{SITE}/wiki/rest/api"

auth = HTTPBasicAuth(EMAIL, API_TOKEN)
headers = {"Accept": "application/json", "Content-Type": "application/json"}


def ok(label):
    print(f"  [OK] {label}")


def fail(label, resp):
    print(f"  [FAIL] {label} — HTTP {resp.status_code}: {resp.text[:300]}")


# ── 1. Test connection ────────────────────────────────────────────────────────
print("\n=== 1. Testing Jira connection ===")
r = requests.get(f"{JIRA_BASE}/myself", auth=auth, headers=headers)
if r.ok:
    me = r.json()
    ok(f"Connected as {me.get('displayName')} ({me.get('emailAddress')})")
else:
    fail("GET /myself", r)
    raise SystemExit("Cannot continue without a working connection.")


# ── 2. Find the SCRUM project key ────────────────────────────────────────────
print("\n=== 2. Finding SCRUM project ===")
r = requests.get(f"{JIRA_BASE}/project/search", auth=auth, headers=headers)
projects = r.json().get("values", []) if r.ok else []
scrum_key = None
for p in projects:
    if "SCRUM" in p.get("name", "").upper() or p.get("key", "").upper() == "SCRUM":
        scrum_key = p["key"]
        ok(f"Found project '{p['name']}' with key '{scrum_key}'")
        break

if not scrum_key:
    if projects:
        scrum_key = projects[0]["key"]
        ok(f"No SCRUM project found; using first project '{projects[0]['name']}' (key={scrum_key})")
    else:
        print("  [FAIL] No projects found — cannot create issues.")
        raise SystemExit(1)


# ── 3. Get available issue types ─────────────────────────────────────────────
print("\n=== 3. Fetching project metadata ===")
r = requests.get(
    f"{JIRA_BASE}/issue/createmeta",
    auth=auth,
    headers=headers,
    params={"projectKeys": scrum_key, "expand": "projects.issuetypes.fields"},
)
meta = r.json() if r.ok else {}
issue_types_available = {}
for proj in meta.get("projects", []):
    for it in proj.get("issuetypes", []):
        issue_types_available[it["name"].lower()] = it["name"]

if not issue_types_available:
    r2 = requests.get(f"{JIRA_BASE}/issuetype", auth=auth, headers=headers)
    if r2.ok:
        for it in r2.json():
            issue_types_available[it["name"].lower()] = it["name"]

ok(f"Issue types available: {list(issue_types_available.values())}")


def resolve_type(requested):
    if requested.lower() in issue_types_available:
        return issue_types_available[requested.lower()]
    for fallback in ["task", "story", "bug"]:
        if fallback in issue_types_available:
            return issue_types_available[fallback]
    return list(issue_types_available.values())[0]


# ── 4. Create Jira issues ────────────────────────────────────────────────────
print("\n=== 4. Creating Jira issues ===")

ISSUES = [
    {"summary": "Fix login bug on iOS",      "type": "Bug",   "status": "In Progress"},
    {"summary": "Design onboarding screens", "type": "Task",  "status": "To Do"},
    {"summary": "Write API documentation",   "type": "Task",  "status": "Done"},
    {"summary": "Performance testing",       "type": "Task",  "status": "In Progress"},
    {"summary": "App store submission",      "type": "Story", "status": "To Do"},
]

created_issues = []

for issue in ISSUES:
    type_name = resolve_type(issue["type"])
    payload = {
        "fields": {
            "project": {"key": scrum_key},
            "summary": issue["summary"],
            "issuetype": {"name": type_name},
        }
    }
    r = requests.post(f"{JIRA_BASE}/issue", auth=auth, headers=headers, json=payload)
    if r.ok:
        issue_key = r.json().get("key")
        created_issues.append((issue_key, issue["status"]))
        ok(f"Created [{issue_key}] '{issue['summary']}' ({type_name})")
    else:
        fail(f"Create '{issue['summary']}'", r)
        created_issues.append((None, issue["status"]))


# ── 5. Transition issues to the desired status ───────────────────────────────
print("\n=== 5. Transitioning issue statuses ===")

STATUS_ALIASES = {
    "in progress": ["in progress", "start progress", "start", "in development"],
    "done":        ["done", "close issue", "resolve issue", "mark as done", "complete"],
    "to do":       ["to do", "reopen issue", "reopen", "backlog"],
}

for issue_key, target_status in created_issues:
    if not issue_key or target_status.lower() == "to do":
        if issue_key:
            ok(f"{issue_key} stays at 'To Do' (default)")
        continue

    r = requests.get(f"{JIRA_BASE}/issue/{issue_key}/transitions", auth=auth, headers=headers)
    if not r.ok:
        fail(f"Get transitions for {issue_key}", r)
        continue

    transitions = {t["to"]["name"].lower(): t["id"] for t in r.json().get("transitions", [])}
    transition_id = None
    for candidate in STATUS_ALIASES.get(target_status.lower(), [target_status.lower()]):
        if candidate in transitions:
            transition_id = transitions[candidate]
            break

    if not transition_id:
        print(f"  [SKIP] No transition to '{target_status}' for {issue_key} "
              f"(available: {list(transitions.keys())})")
        continue

    r2 = requests.post(
        f"{JIRA_BASE}/issue/{issue_key}/transitions",
        auth=auth,
        headers=headers,
        json={"transition": {"id": transition_id}},
    )
    if r2.ok or r2.status_code == 204:
        ok(f"{issue_key} → '{target_status}'")
    else:
        fail(f"Transition {issue_key} to '{target_status}'", r2)


# ── 6. Create Confluence space ───────────────────────────────────────────────
print("\n=== 6. Creating Confluence space 'Engineering' (key=ENG) ===")

conf_space_key = None
r = requests.get(f"{CONFLUENCE_BASE}/space/ENG", auth=auth, headers=headers)
if r.ok:
    ok("Space 'ENG' already exists — reusing it.")
    conf_space_key = "ENG"
elif r.status_code == 401:
    print("  [FAIL] Confluence returned 401 — Confluence is not enabled as a product on this")
    print("         Atlassian site (knoai-dev.atlassian.net).")
    print("  FIX:  Go to admin.atlassian.com → Products → Add Confluence, then grant access")
    print("        to knoaiworkspace@gmail.com.  Re-run this script once Confluence is active.")
else:
    r2 = requests.post(
        f"{CONFLUENCE_BASE}/space",
        auth=auth,
        headers=headers,
        json={
            "key": "ENG",
            "name": "Engineering",
            "description": {
                "plain": {
                    "value": "Engineering team documentation, roadmaps, and guides.",
                    "representation": "plain",
                }
            },
        },
    )
    if r2.ok:
        ok("Created Confluence space 'Engineering' (key=ENG)")
        conf_space_key = "ENG"
    else:
        fail("Create Confluence space", r2)


# ── 7. Create Confluence pages ───────────────────────────────────────────────
print("\n=== 7. Creating Confluence pages ===")

ROADMAP_BODY = """<h2>Overview</h2>
<p>This document outlines the key initiatives and milestones for Q2 2026. Our focus areas are
mobile stability, onboarding experience improvements, and API readiness for third-party integrations.</p>

<h2>Themes</h2>
<ul>
  <li><strong>Mobile Quality</strong> — resolve critical iOS/Android bugs impacting retention</li>
  <li><strong>Growth</strong> — redesign onboarding to improve 7-day activation rate</li>
  <li><strong>Platform</strong> — publish public API docs and developer portal</li>
</ul>

<h2>Milestones</h2>
<table>
  <tr><th>Milestone</th><th>Owner</th><th>Target Date</th><th>Status</th></tr>
  <tr><td>iOS login bug fix shipped</td><td>Mobile Team</td><td>May 10</td><td>In Progress</td></tr>
  <tr><td>Onboarding v2 designs approved</td><td>Design</td><td>May 17</td><td>To Do</td></tr>
  <tr><td>API docs published</td><td>Backend Team</td><td>May 24</td><td>Done</td></tr>
  <tr><td>Performance baseline established</td><td>QA</td><td>Jun 7</td><td>In Progress</td></tr>
  <tr><td>App store submission</td><td>Mobile Team</td><td>Jun 21</td><td>To Do</td></tr>
</table>

<h2>Risks</h2>
<ul>
  <li>App store review times may affect the Jun 21 submission target.</li>
  <li>Performance testing scope is still being defined.</li>
  <li>OAuth sensitive-scope approval from Google required by Week 3.</li>
</ul>"""

ONBOARDING_BODY = """<h2>Purpose</h2>
<p>This guide helps new engineers get their local environment running and understand our development
workflow from day one.</p>

<h2>Prerequisites</h2>
<ul>
  <li>macOS 13+ or Ubuntu 22.04</li>
  <li>Node.js 20 LTS &amp; Python 3.11+</li>
  <li>Docker Desktop 4.x</li>
  <li>Access to the GitHub organisation (request via IT)</li>
</ul>

<h2>Setup Steps</h2>
<ol>
  <li>Clone the repo: <code>git clone git@github.com:kno-ai/kno-ai.git</code></li>
  <li>Copy env template: <code>cp .env.example kno/.env</code> and fill in secrets from 1Password vault <em>Engineering</em>.</li>
  <li>Create a virtual environment: <code>python -m venv .venv &amp;&amp; source .venv/bin/activate</code></li>
  <li>Install dependencies: <code>pip install -r requirements.txt</code></li>
  <li>Run the agent locally: <code>adk web</code></li>
</ol>

<h2>Key Contacts</h2>
<table>
  <tr><th>Topic</th><th>Contact</th></tr>
  <tr><td>Agent / ADK backend</td><td>#backend on Slack</td></tr>
  <tr><td>Mobile app (React Native)</td><td>#mobile on Slack</td></tr>
  <tr><td>Infrastructure / GCP</td><td>#infra on Slack</td></tr>
  <tr><td>Security &amp; Access</td><td>security@knoai.com</td></tr>
</table>

<h2>First-Week Checklist</h2>
<ul>
  <li>Complete local environment setup</li>
  <li>Read the Architecture Decision Records (ADRs) in this space</li>
  <li>Shadow a production deploy</li>
  <li>Pick up your first Jira ticket from the backlog</li>
  <li>Post async stand-up update in #standup by 9:30am each day</li>
</ul>"""

PAGES = [
    ("Q2 Product Roadmap", ROADMAP_BODY),
    ("Onboarding Guide",   ONBOARDING_BODY),
]

if conf_space_key:
    # Get space homepage to use as parent
    r = requests.get(
        f"{CONFLUENCE_BASE}/space/{conf_space_key}",
        auth=auth,
        headers=headers,
        params={"expand": "homepage"},
    )
    parent_id = r.json().get("homepage", {}).get("id") if r.ok else None

    for title, body in PAGES:
        # Check if already exists
        r = requests.get(
            f"{CONFLUENCE_BASE}/content",
            auth=auth,
            headers=headers,
            params={"title": title, "spaceKey": conf_space_key, "type": "page"},
        )
        if r.ok and r.json().get("results"):
            ok(f"Page '{title}' already exists — skipping.")
            continue

        payload = {
            "type": "page",
            "title": title,
            "space": {"key": conf_space_key},
            "body": {"storage": {"value": body, "representation": "storage"}},
        }
        if parent_id:
            payload["ancestors"] = [{"id": parent_id}]

        r = requests.post(f"{CONFLUENCE_BASE}/content", auth=auth, headers=headers, json=payload)
        if r.ok:
            web_url = r.json().get("_links", {}).get("webui", "")
            ok(f"Created page '{title}' — https://{SITE}/wiki{web_url}")
        else:
            fail(f"Create page '{title}'", r)
else:
    print("  [SKIP] No Confluence space available — skipping page creation.")

print("\n=== Done ===\n")
