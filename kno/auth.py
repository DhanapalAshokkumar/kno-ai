"""
User identity from Cloudflare Access JWT.
Production: reads Cf-Access-Jwt-Assertion header injected by Cloudflare.
Local dev:  reads DEV_USER_EMAIL env var.
"""
import os
import json
import base64
from typing import Optional

CF_TEAM_DOMAIN = os.environ.get("CF_TEAM_DOMAIN", "knoai.cloudflareaccess.com")
CF_AUD = os.environ.get("CF_ACCESS_AUD",
    "6fea7864fa38c1a9b3ea0ea35f835d7927c623c2e734737c336c599d8473f385")


def _b64_decode(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (4 - len(data) % 4))


def _decode_jwt_payload(token: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        return json.loads(_b64_decode(parts[1]))
    except Exception:
        return None


def get_current_user(request) -> Optional[str]:
    """Return the authenticated user's email, or None."""
    # Local dev override
    dev = os.environ.get("DEV_USER_EMAIL")
    if dev:
        return dev

    token = request.headers.get("Cf-Access-Jwt-Assertion")
    if not token:
        return None

    payload = _decode_jwt_payload(token)
    if not payload:
        return None

    # Verify audience
    aud = payload.get("aud", [])
    if isinstance(aud, str):
        aud = [aud]
    if CF_AUD not in aud:
        return None

    return payload.get("email")
