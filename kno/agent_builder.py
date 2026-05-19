"""
Agent Generation Engine for kno.ai.

Turns a plain-English description into a fully specified AgentDefinition that
can be persisted, scheduled, and executed by agent_runner.py.

Flow:
  parse_agent_intent(text)        → Gemini extracts trigger + steps
  generate_agent_definition(...)  → builds the AgentDefinition dict
  validate_agent(definition)      → checks connected tools + schema
  deploy_agent(definition)        → saves to Firestore, schedules first run
"""
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import firestore

from kno.user_store import get_app_credentials

logger = logging.getLogger(__name__)

_PROJECT    = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_LOCATION   = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
_COLLECTION = "agents"

_db:    Optional[firestore.Client] = None
_model: Optional[GenerativeModel]  = None


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=_PROJECT)
    return _db


def _get_model() -> GenerativeModel:
    global _model
    if _model is None:
        vertexai.init(project=_PROJECT, location=_LOCATION)
        _model = GenerativeModel("gemini-2.5-flash")
    return _model


# ── Tool registry ─────────────────────────────────────────────────────────────
# Maps tool names the LLM can reference → the connected app they need.
# "write" tools are listed here now; they'll be wired in agent_runner.py.

TOOL_REGISTRY: dict[str, dict] = {
    # READ tools (already in per_user_agent)
    "search_zoho_deals":        {"app": "zoho",       "type": "read"},
    "search_zoho_contacts":     {"app": "zoho",       "type": "read"},
    "search_jira_issues":       {"app": "jira",       "type": "read"},
    "search_gmail":             {"app": "gmail",      "type": "read"},
    "search_drive":             {"app": "gmail",      "type": "read"},
    "search_slack_messages":    {"app": "slack",      "type": "read"},
    "get_slack_activity":       {"app": "slack",      "type": "read"},
    "search_github_issues":     {"app": "github",     "type": "read"},
    "get_github_pull_requests": {"app": "github",     "type": "read"},
    "search_knowledge_base":    {"app": None,         "type": "read"},
    "browse_knowledge_base":    {"app": None,         "type": "read"},
    # WRITE tools (to be built in Week 1)
    "create_jira_issue":        {"app": "jira",       "type": "write"},
    "update_jira_issue":        {"app": "jira",       "type": "write"},
    "send_gmail":               {"app": "gmail",      "type": "write"},
    "post_slack_message":       {"app": "slack",      "type": "write"},
    "update_zoho_deal":         {"app": "zoho",       "type": "write"},
    "create_zoho_activity":     {"app": "zoho",       "type": "write"},
}

# Trigger types and their config shapes
TRIGGER_TYPES = {
    "schedule": ["cron", "timezone"],           # e.g. every Monday 9am
    "event":    ["event_name", "source_app"],   # e.g. new Jira ticket
    "webhook":  ["url", "secret"],              # e.g. Zoho stage change
    "manual":   [],                             # run on demand only
}


# ── parse_agent_intent ────────────────────────────────────────────────────────

_PARSE_PROMPT = """\
You are an agent-definition parser for kno.ai, an enterprise AI assistant platform.

The user has described an automated agent in plain English. Extract the structured
intent and return ONLY a JSON object (no markdown fences, no explanation).

Available tools:
{tool_list}

Available trigger types: schedule, event, webhook, manual

JSON schema to return:
{{
  "name": "<short agent name>",
  "description": "<one-sentence description>",
  "trigger": {{
    "type": "schedule|event|webhook|manual",
    "cron": "<cron expression if schedule, else null>",
    "timezone": "Asia/Kolkata",
    "event_name": "<event name if event trigger, else null>",
    "source_app": "<app name if event trigger, else null>"
  }},
  "steps": [
    {{
      "step": 1,
      "tool": "<tool_name from registry>",
      "description": "<what this step does>",
      "parameters": {{
        "<param>": "<value or {{previous_step.field}} reference>"
      }},
      "condition": "<optional: only run if ...>",
      "for_each": "<optional: field from previous step to iterate over>"
    }}
  ],
  "notifications": {{
    "on_success": "<null or 'slack' or 'email'>",
    "on_failure": "<null or 'slack' or 'email'>"
  }}
}}

User description:
{description}
"""


def parse_agent_intent(description: str) -> dict:
    """Use Gemini to extract structured trigger + steps from plain English.

    Args:
        description: Natural language agent description, e.g.
            "Every Monday at 9am, find Zoho deals with no activity in 7 days
             and create a Jira follow-up task for each one"

    Returns:
        Dict with keys: name, description, trigger, steps, notifications
    """
    tool_list = "\n".join(
        f"  {name} ({meta['type']}, requires: {meta['app'] or 'none'})"
        for name, meta in TOOL_REGISTRY.items()
    )
    prompt = _PARSE_PROMPT.format(tool_list=tool_list, description=description)

    model    = _get_model()
    response = model.generate_content(prompt)
    raw      = response.text.strip()

    # Strip markdown fences if Gemini adds them anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        intent = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("parse_agent_intent: Gemini returned invalid JSON: %s\nRaw: %s", e, raw)
        raise ValueError(f"Intent parse failed — Gemini returned malformed JSON: {e}") from e

    logger.info("Parsed intent: %s (%d steps)", intent.get("name"), len(intent.get("steps", [])))
    return intent


# ── generate_agent_definition ─────────────────────────────────────────────────

def generate_agent_definition(intent: dict, created_by: str,
                               org_id: str = "") -> dict:
    """Build a complete AgentDefinition from a parsed intent.

    Args:
        intent:     Output of parse_agent_intent()
        created_by: Email of the user creating the agent
        org_id:     Optional organisation ID for multi-tenant isolation

    Returns:
        AgentDefinition dict ready to pass to validate_agent() + deploy_agent()
    """
    now = datetime.now(timezone.utc).isoformat()
    agent_id = str(uuid.uuid4())

    definition = {
        # Identity
        "id":          agent_id,
        "name":        intent.get("name", "Unnamed Agent"),
        "description": intent.get("description", ""),
        "created_by":  created_by,
        "org_id":      org_id,
        "created_at":  now,
        "updated_at":  now,

        # Trigger
        "trigger": {
            "type":       intent.get("trigger", {}).get("type", "manual"),
            "cron":       intent.get("trigger", {}).get("cron"),
            "timezone":   intent.get("trigger", {}).get("timezone", "UTC"),
            "event_name": intent.get("trigger", {}).get("event_name"),
            "source_app": intent.get("trigger", {}).get("source_app"),
        },

        # Execution steps
        "steps": intent.get("steps", []),

        # Notifications
        "notifications": intent.get("notifications", {
            "on_success": None,
            "on_failure": "slack",
        }),

        # Runtime state
        "status":     "draft",      # draft → active → paused | error
        "last_run":   None,
        "next_run":   None,
        "run_count":  0,
        "error_count": 0,
        "last_error": None,
    }
    return definition


# ── validate_agent ────────────────────────────────────────────────────────────

def validate_agent(definition: dict, user_email: str) -> dict:
    """Check the agent definition for completeness and connected tools.

    Args:
        definition: Output of generate_agent_definition()
        user_email: Used to check which apps are actually connected

    Returns:
        {"valid": True/False, "errors": [...], "warnings": [...]}
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # 1. Name and steps
    if not definition.get("name"):
        errors.append("Agent has no name")
    if not definition.get("steps"):
        errors.append("Agent has no steps")

    # 2. Trigger validation
    trigger = definition.get("trigger", {})
    ttype   = trigger.get("type")
    if ttype not in TRIGGER_TYPES:
        errors.append(f"Unknown trigger type '{ttype}'. Must be one of: {list(TRIGGER_TYPES)}")
    if ttype == "schedule" and not trigger.get("cron"):
        errors.append("Schedule trigger requires a cron expression")

    # 3. Check every step's tool exists and its app is connected
    for step in definition.get("steps", []):
        tool = step.get("tool")
        if not tool:
            errors.append(f"Step {step.get('step')} has no tool specified")
            continue
        if tool not in TOOL_REGISTRY:
            errors.append(f"Unknown tool '{tool}' in step {step.get('step')}. "
                          f"Available: {list(TOOL_REGISTRY)}")
            continue
        required_app = TOOL_REGISTRY[tool].get("app")
        if required_app:
            creds = get_app_credentials(user_email, required_app)
            if not creds:
                errors.append(
                    f"Step {step.get('step')} uses '{tool}' which requires "
                    f"'{required_app}' — not connected. Go to Settings to connect it."
                )
        # Note: write tools are fully implemented — no warning needed

    return {
        "valid":    len(errors) == 0,
        "errors":   errors,
        "warnings": warnings,
    }


# ── deploy_agent ──────────────────────────────────────────────────────────────

def deploy_agent(definition: dict) -> dict:
    """Persist an agent definition to Firestore and activate it.

    Args:
        definition: A validated AgentDefinition dict

    Returns:
        {"status": "deployed", "agent_id": "...", "next_run": "..."}
    """
    from kno.agent_runner import compute_next_run

    db       = _get_db()
    agent_id = definition["id"]

    # Compute first scheduled run time
    next_run = compute_next_run(definition["trigger"])
    definition["next_run"] = next_run
    definition["status"]   = "active"
    definition["updated_at"] = datetime.now(timezone.utc).isoformat()

    db.collection(_COLLECTION).document(agent_id).set(definition)
    logger.info("Deployed agent '%s' (id=%s, next_run=%s)",
                definition["name"], agent_id, next_run)

    return {
        "status":   "deployed",
        "agent_id": agent_id,
        "name":     definition["name"],
        "next_run": next_run,
        "steps":    len(definition.get("steps", [])),
        "trigger":  definition["trigger"]["type"],
    }


# ── CRUD helpers ──────────────────────────────────────────────────────────────

def list_agents(user_email: str) -> list[dict]:
    """Return all agents owned by user_email."""
    db   = _get_db()
    docs = db.collection(_COLLECTION).where("created_by", "==", user_email).stream()
    return [_strip_large_fields(d.to_dict()) for d in docs]


def get_agent(agent_id: str) -> Optional[dict]:
    db  = _get_db()
    doc = db.collection(_COLLECTION).document(agent_id).get()
    return doc.to_dict() if doc.exists else None


def _strip_large_fields(d: dict) -> dict:
    """Remove large fields not needed for list views."""
    d.pop("steps_detail", None)
    return d
