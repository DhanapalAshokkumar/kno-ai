import google.auth
from google.adk.agents.llm_agent import Agent
from googleapiclient.discovery import build

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
DRIVE_SCOPES = ['https://www.googleapis.com/auth/drive.readonly']


def _gmail_service():
    creds, _ = google.auth.default(scopes=GMAIL_SCOPES)
    return build('gmail', 'v1', credentials=creds)


def _drive_service():
    creds, _ = google.auth.default(scopes=DRIVE_SCOPES)
    return build('drive', 'v3', credentials=creds)


def _extract_body(payload: dict) -> str:
    """Recursively extract plain-text body from a Gmail message payload."""
    import base64
    mime = payload.get('mimeType', '')
    if mime == 'text/plain':
        data = payload.get('body', {}).get('data', '')
        return base64.urlsafe_b64decode(data + '==').decode('utf-8', errors='replace') if data else ''
    for part in payload.get('parts', []):
        text = _extract_body(part)
        if text:
            return text
    return ''


def search_gmail(query: str, max_results: int = 5) -> dict:
    """Search Gmail for emails and threads matching a query.

    Args:
        query: Gmail search query, e.g. 'from:ceo@company.com Q3 budget'
        max_results: Max number of email threads to return (default 5)

    Returns:
        Matching email threads with subject, sender, date, and full body text.
    """
    try:
        service = _gmail_service()
        response = service.users().threads().list(
            userId='me', q=query, maxResults=max_results
        ).execute()

        threads = response.get('threads', [])
        if not threads:
            return {'status': 'no_results', 'message': f'No emails found for: {query}'}

        results = []
        for t in threads:
            detail = service.users().threads().get(
                userId='me', id=t['id'], format='full'
            ).execute()
            msgs = detail.get('messages', [])
            headers = {}
            body_parts = []
            for msg in msgs:
                payload = msg.get('payload', {})
                if not headers:
                    headers = {h['name']: h['value'] for h in payload.get('headers', [])}
                text = _extract_body(payload)
                if text.strip():
                    body_parts.append(text.strip())
            full_body = '\n---\n'.join(body_parts)[:6000]  # cap at 6000 chars
            results.append({
                'thread_id': t['id'],
                'subject': headers.get('Subject', '(no subject)'),
                'from': headers.get('From', 'unknown'),
                'date': headers.get('Date', 'unknown'),
                'body': full_body or detail.get('snippet', ''),
            })

        return {'status': 'success', 'count': len(results), 'threads': results}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def search_drive(query: str, max_results: int = 5) -> dict:
    """Search Google Drive for files matching a query.

    Args:
        query: Natural language or keyword search, e.g. 'Q3 roadmap deck'
        max_results: Max number of files to return (default 5)

    Returns:
        Matching files with name, type, owner, modified date, and link.
    """
    try:
        service = _drive_service()
        drive_query = f"fullText contains '{query}' and trashed=false"
        response = service.files().list(
            q=drive_query,
            pageSize=max_results,
            fields='files(id,name,mimeType,modifiedTime,webViewLink,owners)',
        ).execute()

        files = response.get('files', [])
        if not files:
            return {'status': 'no_results', 'message': f'No files found for: {query}'}

        results = []
        for f in files:
            mime = f.get('mimeType', '')
            owner = (f.get('owners') or [{}])[0].get('displayName', 'unknown')
            results.append({
                'id': f['id'],
                'name': f.get('name'),
                'type': mime.split('.')[-1] if '.' in mime else mime,
                'owner': owner,
                'modified': f.get('modifiedTime'),
                'url': f.get('webViewLink'),
            })

        return {'status': 'success', 'count': len(results), 'files': results}

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


def read_drive_file(file_id: str) -> dict:
    """Read the text content of a Google Drive file.

    Args:
        file_id: The Drive file ID (from search_drive results)

    Returns:
        File name, URL, and up to 8000 characters of text content.
    """
    try:
        service = _drive_service()
        meta = service.files().get(
            fileId=file_id, fields='id,name,mimeType,webViewLink'
        ).execute()

        mime = meta.get('mimeType', '')
        export_types = {
            'application/vnd.google-apps.document': 'text/plain',
            'application/vnd.google-apps.spreadsheet': 'text/csv',
            'application/vnd.google-apps.presentation': 'text/plain',
        }

        if mime in export_types:
            raw = service.files().export(fileId=file_id, mimeType=export_types[mime]).execute()
            content = (raw.decode('utf-8') if isinstance(raw, bytes) else str(raw))[:8000]
        else:
            content = f'[Binary file — open at: {meta.get("webViewLink")}]'

        return {
            'status': 'success',
            'name': meta.get('name'),
            'url': meta.get('webViewLink'),
            'content': content,
        }

    except Exception as e:
        return {'status': 'error', 'message': str(e)}


root_agent = Agent(
    model='gemini-2.5-flash',
    name='kno_agent',
    description=(
        'kno.ai — an AI agent that answers employee questions by searching '
        'across company tools like Gmail and Google Drive.'
    ),
    instruction="""You are kno, an AI assistant for company knowledge.
Your job is to help employees find information from their company tools quickly and accurately.

You have access to:
- **Gmail** — search emails and threads (use search_gmail)
- **Google Drive** — find and read documents, spreadsheets, and slides (use search_drive, read_drive_file)

Guidelines:
- When a user asks a question, identify which tool(s) are likely to have the answer.
- Always search before saying you don't know — the answer may be in their email or Drive.
- If search returns files, use read_drive_file to get the actual content when relevant.
- Cite your sources: include file names, email subjects, senders, and links.
- Be concise. Employees are busy — get to the point.
- If results are unclear or empty, tell the user what you searched and suggest a different query.
- Do not make up information. Only answer from what you find in the tools.""",
    tools=[search_gmail, search_drive, read_drive_file],
)
