"""
GitHub connector for kno.ai
----------------------------
Search issues, pull requests, and reviews across GitHub repos.

Config via env vars:
    GITHUB_TOKEN   — Personal Access Token (ghp_...)
    GITHUB_OWNER   — Default owner/org (e.g. DhanapalAshokkumar)
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv('/Users/dhanapal/kno-ai/kno/.env')

_TOKEN = os.environ.get("GITHUB_TOKEN", "")
_OWNER = os.environ.get("GITHUB_OWNER", "DhanapalAshokkumar")
_API   = "https://api.github.com"


def _headers() -> dict:
    h = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
    if _TOKEN:
        h["Authorization"] = f"Bearer {_TOKEN}"
    return h


def _get(path: str, params: dict = None) -> requests.Response:
    return requests.get(f"{_API}{path}", headers=_headers(), params=params or {})


def _repo(repo: str | None) -> str:
    """Normalise repo arg — accept 'owner/repo' or bare 'repo' (uses _OWNER)."""
    if repo and "/" in repo:
        return repo
    return f"{_OWNER}/{repo}" if repo else f"{_OWNER}/kno-ai"


# ── Issues ─────────────────────────────────────────────────────────────────────

def search_github_issues(query: str, repo: str | None = None, state: str = "open") -> dict:
    """Search GitHub issues by keyword.

    Args:
        query: Text to search for in issue titles and bodies.
        repo:  'owner/repo' or bare repo name (default: GITHUB_OWNER/kno-ai).
               Use 'all' to search across all repos of the owner.
        state: 'open', 'closed', or 'all' (default: 'open').

    Returns:
        Matching issues with number, title, state, labels, author, and URL.
    """
    try:
        if repo and repo.lower() == "all":
            q = f"{query} user:{_OWNER} is:issue state:{state}"
        else:
            q = f"{query} repo:{_repo(repo)} is:issue state:{state}"

        r = _get("/search/issues", {"q": q, "per_page": 10, "sort": "updated"})
        if not r.ok:
            return {"status": "error", "message": r.text[:300]}

        items = r.json().get("items", [])
        if not items:
            return {"status": "no_results", "message": f"No issues found for: {query}"}

        return {
            "status": "success",
            "count": len(items),
            "issues": [
                {
                    "number":  i["number"],
                    "title":   i["title"],
                    "state":   i["state"],
                    "author":  i["user"]["login"],
                    "labels":  [l["name"] for l in i.get("labels", [])],
                    "comments": i["comments"],
                    "url":     i["html_url"],
                    "updated": i["updated_at"][:10],
                }
                for i in items
            ],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_github_issue(issue_number: int, repo: str | None = None) -> dict:
    """Get full details of a GitHub issue including body and comments.

    Args:
        issue_number: The issue number, e.g. 42.
        repo:         'owner/repo' or bare repo name.

    Returns:
        Issue title, body, state, labels, assignees, and last 5 comments.
    """
    try:
        full_repo = _repo(repo)
        r = _get(f"/repos/{full_repo}/issues/{issue_number}")
        if r.status_code == 404:
            return {"status": "no_results", "message": f"Issue #{issue_number} not found in {full_repo}"}
        if not r.ok:
            return {"status": "error", "message": r.text[:300]}

        i = r.json()

        # Fetch last 5 comments
        rc = _get(f"/repos/{full_repo}/issues/{issue_number}/comments",
                  {"per_page": 5, "sort": "updated", "direction": "desc"})
        comments = [
            {"author": c["user"]["login"], "body": c["body"][:500], "date": c["updated_at"][:10]}
            for c in (rc.json() if rc.ok else [])
        ]

        return {
            "status":    "success",
            "number":    i["number"],
            "title":     i["title"],
            "state":     i["state"],
            "author":    i["user"]["login"],
            "assignees": [a["login"] for a in i.get("assignees", [])],
            "labels":    [l["name"] for l in i.get("labels", [])],
            "body":      (i.get("body") or "")[:2000],
            "comments":  comments,
            "url":       i["html_url"],
            "created":   i["created_at"][:10],
            "updated":   i["updated_at"][:10],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Pull Requests ──────────────────────────────────────────────────────────────

def get_github_pull_requests(repo: str | None = None, state: str = "open") -> dict:
    """List pull requests for a GitHub repo.

    Args:
        repo:  'owner/repo' or bare repo name.
        state: 'open', 'closed', or 'all' (default: 'open').

    Returns:
        PRs with number, title, author, reviewers, labels, and URL.
    """
    try:
        full_repo = _repo(repo)
        r = _get(f"/repos/{full_repo}/pulls",
                 {"state": state, "per_page": 20, "sort": "updated", "direction": "desc"})
        if not r.ok:
            return {"status": "error", "message": r.text[:300]}

        prs = r.json()
        if not prs:
            return {"status": "no_results", "message": f"No {state} PRs in {full_repo}"}

        return {
            "status": "success",
            "repo":   full_repo,
            "count":  len(prs),
            "pull_requests": [
                {
                    "number":      p["number"],
                    "title":       p["title"],
                    "state":       p["state"],
                    "author":      p["user"]["login"],
                    "base":        p["base"]["ref"],
                    "head":        p["head"]["ref"],
                    "reviewers":   [r["login"] for r in p.get("requested_reviewers", [])],
                    "labels":      [l["name"] for l in p.get("labels", [])],
                    "draft":       p.get("draft", False),
                    "url":         p["html_url"],
                    "updated":     p["updated_at"][:10],
                }
                for p in prs
            ],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_github_pr_reviews(pr_number: int, repo: str | None = None) -> dict:
    """Get reviews and comments for a specific pull request.

    Args:
        pr_number: The PR number, e.g. 7.
        repo:      'owner/repo' or bare repo name.

    Returns:
        Review decisions (approved/changes_requested/commented) and review comments.
    """
    try:
        full_repo = _repo(repo)

        # PR details
        pr_resp = _get(f"/repos/{full_repo}/pulls/{pr_number}")
        if pr_resp.status_code == 404:
            return {"status": "no_results", "message": f"PR #{pr_number} not found in {full_repo}"}
        if not pr_resp.ok:
            return {"status": "error", "message": pr_resp.text[:300]}
        pr = pr_resp.json()

        # Reviews
        rv_resp = _get(f"/repos/{full_repo}/pulls/{pr_number}/reviews")
        reviews = [
            {
                "reviewer": rv["user"]["login"],
                "state":    rv["state"],   # APPROVED / CHANGES_REQUESTED / COMMENTED
                "body":     (rv.get("body") or "")[:300],
                "date":     rv["submitted_at"][:10],
            }
            for rv in (rv_resp.json() if rv_resp.ok else [])
        ]

        # Review comments (inline)
        rc_resp = _get(f"/repos/{full_repo}/pulls/{pr_number}/comments", {"per_page": 10})
        inline = [
            {
                "reviewer": c["user"]["login"],
                "file":     c["path"],
                "comment":  c["body"][:300],
                "date":     c["updated_at"][:10],
            }
            for c in (rc_resp.json() if rc_resp.ok else [])
        ]

        return {
            "status":          "success",
            "pr_number":       pr_number,
            "title":           pr["title"],
            "state":           pr["state"],
            "author":          pr["user"]["login"],
            "url":             pr["html_url"],
            "reviews":         reviews,
            "inline_comments": inline,
            "mergeable":       pr.get("mergeable"),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ── Repo overview ──────────────────────────────────────────────────────────────

def get_github_repo_summary(repo: str | None = None) -> dict:
    """Get a summary of a GitHub repo — open issues, open PRs, recent activity.

    Args:
        repo: 'owner/repo' or bare repo name.

    Returns:
        Repo metadata plus counts of open issues and open PRs.
    """
    try:
        full_repo = _repo(repo)
        r = _get(f"/repos/{full_repo}")
        if not r.ok:
            return {"status": "error", "message": r.text[:300]}
        d = r.json()

        return {
            "status":       "success",
            "repo":         full_repo,
            "description":  d.get("description", ""),
            "default_branch": d.get("default_branch", "main"),
            "open_issues":  d.get("open_issues_count", 0),
            "stars":        d.get("stargazers_count", 0),
            "language":     d.get("language", ""),
            "url":          d.get("html_url", ""),
            "updated":      (d.get("updated_at") or "")[:10],
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
