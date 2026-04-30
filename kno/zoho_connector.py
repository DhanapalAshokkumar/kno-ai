import os
import requests
from dotenv import load_dotenv

load_dotenv('/Users/dhanapal/kno-ai/kno/.env')

ZOHO_API_BASE = "https://www.zohoapis.in/crm/v2"
_token_store = {
    "access_token": os.getenv("ZOHO_ACCESS_TOKEN", ""),
}


def _client_id() -> str:
    return os.getenv("ZOHO_CLIENT_ID", "")


def _client_secret() -> str:
    return os.getenv("ZOHO_CLIENT_SECRET", "")


def _refresh_token() -> str:
    return os.getenv("ZOHO_REFRESH_TOKEN", "")


def _api_domain() -> str:
    return os.getenv("ZOHO_API_DOMAIN", "https://accounts.zoho.in")


def refresh_zoho_token() -> str:
    """Exchange the refresh token for a new Zoho access token.

    Returns:
        The new access token string, or an empty string on failure.
    """
    response = requests.post(
        f"{_api_domain()}/oauth/v2/token",
        data={
            "refresh_token": _refresh_token(),
            "client_id": _client_id(),
            "client_secret": _client_secret(),
            "grant_type": "refresh_token",
        },
    )
    data = response.json()
    token = data.get("access_token", "")
    if token:
        _token_store["access_token"] = token
    return token


def _headers() -> dict:
    return {"Authorization": f"Zoho-oauthtoken {_token_store['access_token']}"}


def _get(url: str, params: dict = None):
    """Make a GET request, retrying once after refreshing the token on 401."""
    resp = requests.get(url, headers=_headers(), params=params)
    if resp.status_code == 401:
        refresh_zoho_token()
        resp = requests.get(url, headers=_headers(), params=params)
    return resp


def search_zoho_contacts(query: str) -> dict:
    """Search Zoho CRM contacts by name or email.

    Args:
        query: Name or email to search for, e.g. 'Alice' or 'alice@acme.com'

    Returns:
        Matching contacts with First_Name, Last_Name, Email, and Phone.
    """
    try:
        resp = _get(
            f"{ZOHO_API_BASE}/Contacts/search",
            params={"word": query, "fields": "First_Name,Last_Name,Email,Phone"},
        )
        if resp.status_code == 204:
            return {"status": "no_results", "message": f"No contacts found for: {query}"}
        if not resp.ok:
            return {"status": "error", "message": resp.text}

        contacts = [
            {
                "id": c.get("id"),
                "first_name": c.get("First_Name", ""),
                "last_name": c.get("Last_Name", ""),
                "email": c.get("Email", ""),
                "phone": c.get("Phone", ""),
            }
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

    Returns:
        Deals with Deal_Name, Amount, Stage, and Closing_Date.
    """
    try:
        fields = "Deal_Name,Amount,Stage,Closing_Date"
        if stage:
            resp = _get(
                f"{ZOHO_API_BASE}/Deals/search",
                params={"criteria": f"Stage:equals:{stage}", "fields": fields},
            )
        else:
            resp = _get(f"{ZOHO_API_BASE}/Deals", params={"fields": fields})

        if resp.status_code == 204:
            msg = f"No deals found" + (f" in stage: {stage}" if stage else "")
            return {"status": "no_results", "message": msg}
        if not resp.ok:
            return {"status": "error", "message": resp.text}

        deals = [
            {
                "id": d.get("id"),
                "deal_name": d.get("Deal_Name", ""),
                "amount": d.get("Amount"),
                "stage": d.get("Stage", ""),
                "closing_date": d.get("Closing_Date", ""),
            }
            for d in resp.json().get("data", [])
        ]
        return {"status": "success", "count": len(deals), "deals": deals}

    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_zoho_contact(contact_id: str) -> dict:
    """Get full details of a single Zoho CRM contact by ID.

    Args:
        contact_id: The Zoho CRM contact ID (from search_zoho_contacts results)

    Returns:
        Full contact record with all available fields.
    """
    try:
        resp = _get(f"{ZOHO_API_BASE}/Contacts/{contact_id}")
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
