import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv('/Users/dhanapal/kno-ai/kno/.env', override=False)  # no-op if file absent (Cloud Run)

_SITE = os.getenv("ATLASSIAN_SITE", "")
_JIRA_BASE = f"https://{_SITE}/rest/api/3"
_CONFLUENCE_BASE = f"https://{_SITE}/wiki/rest/api"


def _auth() -> HTTPBasicAuth:
    return HTTPBasicAuth(
        os.getenv("ATLASSIAN_EMAIL", ""),
        os.getenv("ATLASSIAN_API_TOKEN", ""),
    )


def _get(url: str, params: dict = None):
    return requests.get(url, auth=_auth(), params=params, headers={"Accept": "application/json"})


def search_jira_issues(query: str) -> dict:
    """Search Jira issues by text using JQL.

    Args:
        query: Text to search for in issue summaries and descriptions, e.g. 'login bug'.
               May include status keywords like 'in progress', 'done', or 'to do'.

    Returns:
        Matching issues with key, summary, status, and assignee.
    """
    # Map of recognisable status keywords → canonical Jira status names (double-quoted in JQL).
    STATUS_MAP = {
        "in progress":  "In Progress",
        "inprogress":   "In Progress",
        "done":         "Done",
        "complete":     "Done",
        "completed":    "Done",
        "closed":       "Done",
        "to do":        "To Do",
        "todo":         "To Do",
        "open":         "To Do",
        "backlog":      "To Do",
    }

    def _run_search(jql: str) -> requests.Response:
        print(f"[search_jira_issues] JQL: {jql}")
        return _get(
            f"{_JIRA_BASE}/search/jql",
            params={
                "jql": jql,
                "maxResults": 10,
                "fields": "id,key,summary,status,assignee,priority",
            },
        )

    def _build_primary_jql(q: str) -> str:
        """
        Build the primary JQL query.
        - If the query already looks like a JQL expression (contains JQL
          operators such as '=', '~', 'AND', 'OR', 'ORDER BY', 'project',
          'status', 'assignee', etc.), pass it through unchanged.
        - If the query contains a recognisable status keyword, include a
          status clause with the value in double quotes:
              status = "In Progress" AND text ~ "login" ORDER BY updated DESC
        - Otherwise wrap it in a plain text search:
              text ~ "query" ORDER BY updated DESC
        """
        import re

        # Heuristic: treat as raw JQL if it contains JQL operators/keywords
        JQL_PATTERN = re.compile(
            r'\b(project|status|assignee|reporter|issuetype|priority|'
            r'created|updated|AND|OR|NOT|ORDER\s+BY|IN\s*\()\b|[=~<>!]',
            re.IGNORECASE,
        )
        if JQL_PATTERN.search(q):
            return q  # already a JQL expression — use as-is

        q_lower = q.lower().strip()
        matched_status = None
        remaining = q

        for keyword, status_name in STATUS_MAP.items():
            if keyword in q_lower:
                matched_status = status_name
                # Remove the matched keyword from the text portion so it isn't
                # double-counted in the text ~ clause.
                remaining = re.sub(re.escape(keyword), "", q_lower, flags=re.IGNORECASE).strip()
                break

        if matched_status and remaining:
            # status = "In Progress" AND text ~ "login bug" ORDER BY updated DESC
            return f'status = "{matched_status}" AND text ~ "{remaining}" ORDER BY updated DESC'
        elif matched_status:
            # Only a status keyword, no extra text
            return f'status = "{matched_status}" ORDER BY updated DESC'
        else:
            return f'text ~ "{q}" ORDER BY updated DESC'

    def _extract_issue(raw: dict) -> dict:
        """
        Parse one issue from the search response.
        The /search/jql endpoint may return issues with fields populated or
        just bare {id, key} objects.  When fields are missing or empty, fall
        back to a GET /issue/{issue_id} call to fetch full details.
        """
        fields = raw.get("fields") or {}
        issue_id = raw.get("id", "")
        key = raw.get("key", issue_id)

        # Decide whether we need a second call
        needs_fetch = not fields.get("summary")
        if needs_fetch:
            print(f"[search_jira_issues] fields missing for {key}, fetching /issue/{issue_id}")
            detail_resp = _get(
                f"{_JIRA_BASE}/issue/{issue_id}",
                params={"fields": "summary,status,assignee,priority"},
            )
            if detail_resp.ok:
                fields = detail_resp.json().get("fields") or {}
                key = detail_resp.json().get("key", key)
            else:
                print(f"[search_jira_issues] fallback fetch failed for {key}: {detail_resp.status_code}")

        return {
            "key": key,
            "summary": fields.get("summary", ""),
            "status": (fields.get("status") or {}).get("name", ""),
            "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
            "priority": (fields.get("priority") or {}).get("name", ""),
        }

    try:
        primary_jql = _build_primary_jql(query)
        resp = _run_search(primary_jql)

        if not resp.ok:
            fallback_jql = f'text ~ "{query}"'
            print(
                f"[search_jira_issues] Primary JQL failed (HTTP {resp.status_code}), "
                f"retrying with fallback"
            )
            resp = _run_search(fallback_jql)
            if not resp.ok:
                return {"status": "error", "message": resp.text}

        raw_issues = resp.json().get("issues", [])
        if not raw_issues:
            return {"status": "no_results", "message": f"No Jira issues found for: {query}"}

        results = [_extract_issue(i) for i in raw_issues]
        return {"status": "success", "count": len(results), "issues": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_jira_issue(issue_key: str) -> dict:
    """Get full details of a single Jira issue.

    Args:
        issue_key: The Jira issue key, e.g. 'SCRUM-15'

    Returns:
        Full issue details including summary, description, status, assignee, priority, and comments.
    """
    try:
        resp = _get(
            f"{_JIRA_BASE}/issue/{issue_key}",
            params={"fields": "summary,description,status,assignee,priority,comment,created,updated"},
        )
        if resp.status_code == 404:
            return {"status": "no_results", "message": f"Issue not found: {issue_key}"}
        if not resp.ok:
            return {"status": "error", "message": resp.text}

        fields = resp.json().get("fields", {})

        # Extract plain text from Atlassian Document Format description
        description = ""
        desc_node = fields.get("description")
        if isinstance(desc_node, dict):
            description = _adf_to_text(desc_node)
        elif isinstance(desc_node, str):
            description = desc_node

        comments = [
            {
                "author": (c.get("author") or {}).get("displayName", "unknown"),
                "body": _adf_to_text(c["body"]) if isinstance(c.get("body"), dict) else c.get("body", ""),
                "created": c.get("created", ""),
            }
            for c in (fields.get("comment") or {}).get("comments", [])[-5:]
        ]

        return {
            "status": "success",
            "key": issue_key,
            "summary": fields.get("summary", ""),
            "description": description[:4000],
            "issue_status": (fields.get("status") or {}).get("name", ""),
            "assignee": (fields.get("assignee") or {}).get("displayName", "Unassigned"),
            "priority": (fields.get("priority") or {}).get("name", ""),
            "created": fields.get("created", ""),
            "updated": fields.get("updated", ""),
            "comments": comments,
            "url": f"https://{_SITE}/browse/{issue_key}",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def search_confluence_pages(query: str) -> dict:
    """Search Confluence pages by text.

    Args:
        query: Text to search for in Confluence pages, e.g. 'onboarding guide'

    Returns:
        Matching pages with title, space, URL, and excerpt.
    """
    try:
        resp = _get(
            f"{_CONFLUENCE_BASE}/content/search",
            params={"cql": f'text ~ "{query}" AND type = page', "limit": 10, "expand": "space,excerpt"},
        )
        if not resp.ok:
            return {"status": "error", "message": resp.text}

        results = resp.json().get("results", [])
        if not results:
            return {"status": "no_results", "message": f"No Confluence pages found for: {query}"}

        pages = [
            {
                "id": p["id"],
                "title": p.get("title", ""),
                "space": (p.get("space") or {}).get("name", ""),
                "url": f"https://{_SITE}/wiki{p.get('_links', {}).get('webui', '')}",
                "excerpt": p.get("excerpt", ""),
            }
            for p in results
        ]
        return {"status": "success", "count": len(pages), "pages": pages}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_confluence_page(page_id: str) -> dict:
    """Get the full content of a Confluence page.

    Args:
        page_id: The Confluence page ID (from search_confluence_pages results)

    Returns:
        Page title, space, URL, and full text content.
    """
    try:
        resp = _get(
            f"{_CONFLUENCE_BASE}/content/{page_id}",
            params={"expand": "body.storage,space,version"},
        )
        if resp.status_code == 404:
            return {"status": "no_results", "message": f"Page not found: {page_id}"}
        if not resp.ok:
            return {"status": "error", "message": resp.text}

        data = resp.json()
        html_body = (data.get("body") or {}).get("storage", {}).get("value", "")
        text = _html_to_text(html_body)

        return {
            "status": "success",
            "id": page_id,
            "title": data.get("title", ""),
            "space": (data.get("space") or {}).get("name", ""),
            "url": f"https://{_SITE}/wiki{data.get('_links', {}).get('webui', '')}",
            "version": (data.get("version") or {}).get("number"),
            "content": text[:8000],
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def _adf_to_text(node: dict) -> str:
    """Recursively extract plain text from an Atlassian Document Format node."""
    if not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = [_adf_to_text(child) for child in node.get("content", [])]
    return " ".join(p for p in parts if p)


def _html_to_text(html: str) -> str:
    """Strip HTML tags to extract plain text."""
    import re
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()
