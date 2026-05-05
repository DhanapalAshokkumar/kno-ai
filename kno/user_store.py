"""
Per-user profile and credential vault using Firestore.
All OAuth tokens are encrypted at rest with Fernet symmetric encryption.
"""
import os
import json
from datetime import datetime, timezone
from typing import Optional, Dict

from cryptography.fernet import Fernet
from google.cloud import firestore

SUPPORTED_APPS = ["gmail", "slack", "github", "jira", "zoho"]

_db: Optional[firestore.Client] = None


def _get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT", "kno-ai-494516"))
    return _db


def _cipher() -> Fernet:
    key = os.environ.get("ENCRYPTION_KEY", "")
    if not key:
        raise ValueError("ENCRYPTION_KEY secret not set")
    return Fernet(key.encode())


def _encrypt(data: dict) -> str:
    return _cipher().encrypt(json.dumps(data).encode()).decode()


def _decrypt(encrypted: str) -> dict:
    return json.loads(_cipher().decrypt(encrypted.encode()))


# ── User profile ──────────────────────────────────────────────────────────────

def get_or_create_user(email: str) -> dict:
    """Return existing user profile or create a new one."""
    db = _get_db()
    ref = db.collection("users").document(email)
    doc = ref.get()
    if not doc.exists:
        user = {
            "email": email,
            "connected_apps": {},
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        ref.set(user)
        return user
    return doc.to_dict()


def get_user(email: str) -> Optional[dict]:
    doc = _get_db().collection("users").document(email).get()
    return doc.to_dict() if doc.exists else None


# ── App credentials ───────────────────────────────────────────────────────────

def store_app_credentials(email: str, app: str, credentials: dict) -> None:
    """Encrypt and store credentials for a user's connected app."""
    _get_db().collection("users").document(email).set({
        f"connected_apps": {app: _encrypt(credentials)},
        f"connected_at": {app: datetime.now(timezone.utc).isoformat()},
    }, merge=True)


def get_app_credentials(email: str, app: str) -> Optional[dict]:
    """Retrieve and decrypt credentials for a user's connected app."""
    user = get_user(email)
    if not user:
        return None
    encrypted = user.get("connected_apps", {}).get(app)
    if not encrypted:
        return None
    try:
        return _decrypt(encrypted)
    except Exception:
        return None


def disconnect_app(email: str, app: str) -> None:
    """Remove a user's app connection."""
    db = _get_db()
    db.collection("users").document(email).update({
        f"connected_apps.{app}": firestore.DELETE_FIELD,
        f"connected_at.{app}": firestore.DELETE_FIELD,
    })


def get_connected_apps(email: str) -> Dict[str, bool]:
    """Return which apps are connected for this user."""
    user = get_user(email)
    if not user:
        return {app: False for app in SUPPORTED_APPS}
    connected = user.get("connected_apps", {})
    return {app: app in connected for app in SUPPORTED_APPS}
