"""
Agent Runner for kno.ai.

Executes AgentDefinition steps in sequence, resolves parameter references
between steps, logs every execution, and handles retries + error alerting.

Architecture:
  execute_agent(agent_id, user_email)
      → load definition from Firestore
      → run each step in order
          → resolve {previous_step.field} references
          → call the tool function
          → apply for_each iteration if present
          → check conditions
      → log_execution(agent_id, result)
      → update definition.last_run / next_run / status

Called from:
  POST /agents/{id}/run       — manual trigger (prod_app.py)
  Cloud Scheduler (future)    — scheduled trigger via /admin/agents/tick
"""
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from croniter import croniter

from google.cloud import firestore

logger = logging.getLogger(__name__)

_PROJECT    = os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516")
_COLLECTION = "agents"
_EXEC_COLL  = "agent_executions"

_db: Optional[firestore.Client] = None


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=_PROJECT)
    return _db


# ── Scheduling helpers ────────────────────────────────────────────────────────

def compute_next_run(trigger: dict) -> Optional[str]:
    """Return ISO-8601 datetime for the next scheduled run, or None."""
    if trigger.get("type") != "schedule":
        return None
    cron = trigger.get("cron")
    if not cron:
        return None
    try:
        it = croniter(cron, datetime.now(timezone.utc))
        return it.get_next(datetime).isoformat()
    except Exception as e:
        logger.warning("compute_next_run: invalid cron '%s': %s", cron, e)
        return None


# ── Tool dispatch ─────────────────────────────────────────────────────────────

def _call_tool(tool_name: str, parameters: dict, user_email: str) -> Any:
    """Dispatch a tool call by name.

    Read tools delegate to the per_user_agent tool factories.
    Write tools are stub implementations that log what they would do
    until Week 1 write capabilities are fully wired.
    """
    # ── Read tools — delegate to live per_user_agent factories ───────────────
    from kno.per_user_agent import (
        _make_gmail_tool, _make_drive_tools, _make_slack_tools,
        _make_jira_tools, _make_github_tools,
        _make_zoho_tools, _make_rag_tools,
    )

    read_dispatch = {
        "search_gmail":             lambda: _make_gmail_tool(user_email),
        "search_drive":             lambda: _make_drive_tools(user_email)[0],
        "get_slack_activity":       lambda: _make_slack_tools(user_email)[0],
        "search_slack_messages":    lambda: _make_slack_tools(user_email)[1],
        "search_jira_issues":       lambda: _make_jira_tools(user_email)[0],
        "search_confluence_pages":  lambda: _make_jira_tools(user_email)[2],  # [0]=jira, [2]=confluence search
        "search_github_issues":     lambda: _make_github_tools(user_email)[1],  # [0]=list_repos, [1]=search
        "get_github_pull_requests": lambda: _make_github_tools(user_email)[2],
        "search_zoho_deals":        lambda: _make_zoho_tools(user_email)[1],   # [0]=contacts, [1]=deals
        "search_zoho_contacts":     lambda: _make_zoho_tools(user_email)[0],
        "search_knowledge_base":    lambda: _make_rag_tools()[0],
        "browse_knowledge_base":    lambda: _make_rag_tools()[1],
    }
    if tool_name in read_dispatch:
        tool_fn = read_dispatch[tool_name]()
        # Filter parameters to only those the tool function actually accepts,
        # preventing errors when Gemini invents parameter names that don't exist.
        import inspect
        sig        = inspect.signature(tool_fn)
        valid_keys = set(sig.parameters.keys())
        safe_params = {k: v for k, v in parameters.items() if k in valid_keys}
        if len(safe_params) < len(parameters):
            dropped = set(parameters) - valid_keys
            logger.warning("_call_tool: dropped unknown params for %s: %s", tool_name, dropped)
        return tool_fn(**safe_params)

    # ── Write tools — live implementations ───────────────────────────────────
    write_dispatch = {
        "send_gmail":           lambda: _make_gmail_tool(user_email)[1],
        "post_slack_message":   lambda: _make_slack_tools(user_email)[2],
        "create_jira_issue":    lambda: _make_jira_tools(user_email)[4],
        "update_jira_issue":    lambda: _make_jira_tools(user_email)[5],
        "update_zoho_deal":     lambda: _make_zoho_tools(user_email)[3],
        "create_zoho_activity": lambda: _make_zoho_tools(user_email)[4],
    }
    if tool_name in write_dispatch:
        tool_fn     = write_dispatch[tool_name]()
        import inspect
        sig         = inspect.signature(tool_fn)
        valid_keys  = set(sig.parameters.keys())
        safe_params = {k: v for k, v in parameters.items() if k in valid_keys}
        if len(safe_params) < len(parameters):
            logger.warning("_call_tool write: dropped unknown params for %s: %s",
                           tool_name, set(parameters) - valid_keys)
        return tool_fn(**safe_params)

    return {"status": "error", "message": f"Unknown tool: {tool_name}"}


# ── Parameter reference resolver ─────────────────────────────────────────────

def _resolve_params(params: dict, step_outputs: dict) -> dict:
    """Replace {stepN.field} references in parameter values with actual values.

    Example:
        params = {"issue_title": "{step1.deals[0].name} follow-up"}
        step_outputs = {"step1": {"deals": [{"name": "Acme Corp", ...}]}}
        → {"issue_title": "Acme Corp follow-up"}
    """
    resolved = {}
    for key, val in params.items():
        if isinstance(val, str):
            # Normalise step_N → stepN before resolving references
            val = re.sub(r"step_(\d+)", r"step\1", val)
            # Replace {stepN.field.subfield} style references
            def replacer(m):
                path = m.group(1).split(".")
                obj  = step_outputs
                for part in path:
                    if isinstance(obj, dict):
                        obj = obj.get(part, m.group(0))
                    elif isinstance(obj, list) and part.isdigit():
                        obj = obj[int(part)] if int(part) < len(obj) else m.group(0)
                    else:
                        return m.group(0)
                return str(obj)
            resolved[key] = re.sub(r"\{([^}]+)\}", replacer, val)
        else:
            resolved[key] = val
    return resolved


def _evaluate_condition(condition: str, step_outputs: dict) -> bool:
    """Simple condition evaluator for agent step gating.

    Supports:
      "step1.count > 0"
      "step1.status == 'success'"
      "step1.deals != []"
    Returns True if condition passes (or if condition is empty/None).
    """
    if not condition:
        return True
    # Replace {stepN.field} references
    resolved_cond = condition
    for step_key, output in step_outputs.items():
        if isinstance(output, dict):
            for field, value in output.items():
                resolved_cond = resolved_cond.replace(
                    f"{step_key}.{field}", repr(value)
                )
    try:
        return bool(eval(resolved_cond, {"__builtins__": {}}, {}))  # noqa: S307
    except Exception:
        logger.warning("Could not evaluate condition '%s' — defaulting to True", condition)
        return True


# ── execute_agent ─────────────────────────────────────────────────────────────

def execute_agent(agent_id: str, user_email: str,
                  triggered_by: str = "manual") -> dict:
    """Run all steps of an agent in sequence.

    Args:
        agent_id:     Firestore agent document ID
        user_email:   The owning user (for credential lookup)
        triggered_by: "manual", "schedule", or "event"

    Returns:
        Execution result dict with step_results, status, duration_ms
    """
    db  = _get_db()
    doc = db.collection(_COLLECTION).document(agent_id).get()
    if not doc.exists:
        return {"status": "error", "message": f"Agent {agent_id} not found"}

    definition   = doc.to_dict()
    agent_name   = definition.get("name", agent_id)
    steps        = definition.get("steps", [])
    exec_id      = str(uuid.uuid4())
    started_at   = datetime.now(timezone.utc)

    logger.info("▶ Executing agent '%s' (exec=%s, trigger=%s)",
                agent_name, exec_id, triggered_by)

    step_outputs: dict[str, Any] = {}
    step_results: list[dict]     = []
    overall_status                = "success"

    for step_def in sorted(steps, key=lambda s: s.get("step", 0)):
        step_num  = step_def.get("step", 0)
        step_key  = f"step{step_num}"
        tool_name = step_def.get("tool", "")
        raw_params = step_def.get("parameters", {})
        condition  = step_def.get("condition")
        for_each   = step_def.get("for_each")

        # Resolve parameter references from prior steps
        params = _resolve_params(raw_params, step_outputs)

        # Evaluate step condition
        if not _evaluate_condition(condition, step_outputs):
            logger.info("  Step %d (%s) skipped — condition not met: %s",
                        step_num, tool_name, condition)
            step_results.append({
                "step": step_num, "tool": tool_name,
                "status": "skipped", "condition": condition,
            })
            continue

        # for_each: iterate over a list from a previous step output
        if for_each:
            # Normalise: Gemini writes "step_1.deals", runner stores "step1"
            normalised = re.sub(r"step_(\d+)", r"step\1", for_each)
            path  = normalised.split(".")
            items = step_outputs
            for part in path:
                items = items.get(part, []) if isinstance(items, dict) else []
            if not isinstance(items, list):
                items = [items] if items else []

            logger.info("  Step %d for_each '%s' → %d items", step_num, for_each, len(items))

            iteration_results = []
            for item in items:
                iter_params = {k: v for k, v in params.items()}
                # Inject current item fields as {item.field} AND {step_N.current_item.field}
                if isinstance(item, dict):
                    for k, v in item.items():
                        for param_key in iter_params:
                            if isinstance(iter_params[param_key], str):
                                iter_params[param_key] = iter_params[param_key] \
                                    .replace(f"{{item.{k}}}", str(v)) \
                                    .replace(f"{{step_{step_num - 1}.current_item.{k}}}", str(v)) \
                                    .replace(f"{{step{step_num - 1}.current_item.{k}}}", str(v))
                try:
                    result = _call_tool(tool_name, iter_params, user_email)
                    iteration_results.append(result)
                except Exception as e:
                    iteration_results.append({"status": "error", "message": str(e)})

            step_output = {"items": iteration_results, "count": len(iteration_results)}
            step_results.append({
                "step": step_num, "tool": tool_name, "for_each": for_each,
                "iterations": len(items), "status": "success",
            })

        else:
            # Single call
            try:
                step_output = _call_tool(tool_name, params, user_email)
                step_results.append({
                    "step": step_num, "tool": tool_name,
                    "status": step_output.get("status", "success"),
                    "summary": _summarise_output(step_output),
                })
                if step_output.get("status") == "error":
                    overall_status = "partial"
            except Exception as e:
                logger.error("  Step %d (%s) raised: %s", step_num, tool_name, e)
                step_output    = {"status": "error", "message": str(e)}
                overall_status = "partial"
                step_results.append({
                    "step": step_num, "tool": tool_name,
                    "status": "error", "error": str(e),
                })

        step_outputs[step_key] = step_output

    finished_at  = datetime.now(timezone.utc)
    duration_ms  = int((finished_at - started_at).total_seconds() * 1000)

    execution = {
        "exec_id":      exec_id,
        "agent_id":     agent_id,
        "agent_name":   agent_name,
        "triggered_by": triggered_by,
        "status":       overall_status,
        "step_results": step_results,
        "started_at":   started_at.isoformat(),
        "finished_at":  finished_at.isoformat(),
        "duration_ms":  duration_ms,
    }

    log_execution(agent_id, execution)

    # Update agent runtime counters
    updates: dict = {
        "last_run":   started_at.isoformat(),
        "run_count":  firestore.Increment(1),
        "updated_at": finished_at.isoformat(),
    }
    if overall_status == "success":
        next_run = compute_next_run(definition.get("trigger", {}))
        if next_run:
            updates["next_run"] = next_run
    else:
        updates["error_count"] = firestore.Increment(1)
        updates["last_error"]  = step_results[-1].get("error", "unknown")

    db.collection(_COLLECTION).document(agent_id).update(updates)

    logger.info("✓ Agent '%s' finished in %dms — %s", agent_name, duration_ms, overall_status)
    return execution


# ── log_execution ─────────────────────────────────────────────────────────────

def log_execution(agent_id: str, result: dict) -> None:
    """Persist an execution record to Firestore for audit + history."""
    db = _get_db()
    exec_id = result.get("exec_id", str(uuid.uuid4()))
    db.collection(_EXEC_COLL).document(exec_id).set({
        **result,
        "logged_at": datetime.now(timezone.utc).isoformat(),
    })


def get_execution_history(agent_id: str, limit: int = 20) -> list[dict]:
    """Return the N most recent execution records for an agent."""
    db   = _get_db()
    docs = (
        db.collection(_EXEC_COLL)
          .where("agent_id", "==", agent_id)
          .order_by("started_at", direction=firestore.Query.DESCENDING)
          .limit(limit)
          .stream()
    )
    return [d.to_dict() for d in docs]


# ── handle_error ──────────────────────────────────────────────────────────────

def handle_error(agent_id: str, error: str, user_email: str) -> None:
    """Mark agent as errored and optionally send a Slack/email alert."""
    db = _get_db()
    db.collection(_COLLECTION).document(agent_id).update({
        "status":      "error",
        "last_error":  error,
        "error_count": firestore.Increment(1),
        "updated_at":  datetime.now(timezone.utc).isoformat(),
    })
    logger.error("Agent %s entered error state: %s", agent_id, error)
    # TODO Week 1: send Slack/email alert via notification tool


# ── Private helpers ───────────────────────────────────────────────────────────

def _summarise_output(output: Any) -> str:
    """Return a short human-readable summary of a tool's output."""
    if not isinstance(output, dict):
        return str(output)[:120]
    status = output.get("status", "")
    if status == "error":
        return f"error: {output.get('message', '')[:100]}"
    if "count" in output:
        return f"{output['count']} result(s)"
    if "files" in output:
        return f"{len(output['files'])} file(s)"
    if "passages" in output:
        return f"{len(output['passages'])} passage(s)"
    return status or "ok"
