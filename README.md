# kno.ai

> Ask one question. Get answers from every tool your company uses.

kno.ai is a **production multi-tenant AI assistant** that connects your company's tools — Gmail, Google Drive, Slack, Jira, Confluence, GitHub, Zoho CRM — so employees can find information across all their systems in a single chat, with traceable citations back to the source.

Built for the **Google for Startups AI Agents Challenge**.

🔗 **Live at [kno.xnukernel.com](https://kno.xnukernel.com)**

---

## What It Does

```
You:  "What deals are we closing this month?"
kno:  Acme Corp is in Negotiation — $85,000, closing May 31 [1]
      TechCo deal moved to Proposal stage — $42,000 [2]

      Sources
      [1] Zoho CRM: Acme Corp — Stage: Negotiation | Amount: $85,000 | Closing: May 31
      [2] Zoho CRM: TechCo — Stage: Proposal | Amount: $42,000 | Closing: Jun 15
```

Every answer includes numbered citations `[1][2]` with a **Sources** block showing exactly where the information came from — email subject + sender, Jira ticket key, Confluence page title, GitHub PR number, Zoho deal name.

---

## Architecture

```
Browser → Cloudflare Zero Trust → Cloud Run (FastAPI)
                                        │
                          ┌─────────────┼──────────────┐
                          ▼             ▼               ▼
                    Firestore      Vertex AI        Per-user
                  (credentials,   Sessions +       tool calls
                  knowledge base)  Memory          (OAuth'd)
```

| Layer | Technology |
|---|---|
| Agent framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash (Vertex AI) |
| Runtime | FastAPI + uvicorn on Cloud Run |
| Auth | Cloudflare Zero Trust (JWT) |
| Credential vault | Firestore + Fernet encryption |
| Sessions | Vertex AI Session Service (persistent across instances) |
| Memory | Vertex AI Memory Bank (cross-session recall) |
| Knowledge base | Firestore full-text search (Confluence pages) |

---

## Connected Tools

| Tool | Auth method | What kno can do |
|---|---|---|
| **Gmail** | OAuth 2.0 | Search threads, read email summaries |
| **Google Drive** | OAuth 2.0 | Search files, extract text from Docs/Sheets/Slides |
| **Slack** | OAuth 2.0 (user token) | Full-text search across all channels |
| **Jira** | API token | Search issues, get ticket details and comments |
| **Confluence** | API token | Search pages, read full content |
| **GitHub** | Personal Access Token | List repos, search issues, list PRs |
| **Zoho CRM** | OAuth 2.0 | Search contacts, list and filter deals |

Each user connects their own accounts — credentials are encrypted with Fernet and stored per-user in Firestore. No user ever sees another user's data.

---

## Features

### 🔐 Multi-tenant isolation
Every user's agent is scoped to their own credentials. Tool factories are built at query time with the requesting user's tokens.

### 📌 Traceable citations
Every factual claim includes a citation marker. Responses end with a **Sources** section:
```
[1] Gmail: Budget approval — From: cfo@company.com | Date: May 8, 2026
[2] Confluence: Q2 Roadmap — Space: PROD | Updated: May 1, 2026 | [link]
[3] Jira: ENG-142 — Fix login timeout | Status: In Progress | [link]
```

### 🧠 Persistent memory
Sessions persist across Cloud Run instances via Vertex AI Session Service. Long-term memory (cross-session recall) via Vertex AI Memory Bank.

### 📚 Company knowledge base
Confluence pages are ingested into a Firestore knowledge base. The `search_knowledge_base` tool answers questions about internal docs with citations.

### 👤 Admin dashboard
`/admin/users` — live view of all signed-up users and which apps they've connected.

---

## Project Structure

```
kno-ai/
├── kno/
│   ├── prod_app.py           # FastAPI app — OAuth flows, chat endpoint, admin
│   ├── per_user_agent.py     # Per-user agent runner + tool factories
│   ├── rag_connector.py      # Firestore knowledge base (ingest + search)
│   ├── citation_formatter.py # Citation helpers for all source types
│   ├── auth.py               # Cloudflare JWT auth
│   ├── user_store.py         # Firestore credential vault (Fernet encrypted)
│   └── static/
│       └── index.html        # SPA frontend (Connect Apps + Chat)
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Deployment

### Prerequisites
- Google Cloud project with Vertex AI, Firestore, Secret Manager, Cloud Run enabled
- Cloudflare Zero Trust application protecting the Cloud Run URL
- OAuth apps configured: Google (Gmail + Drive), Slack
- Secrets in Secret Manager: `ENCRYPTION_KEY`, `GOOGLE_OAUTH_CLIENT_ID`, `GOOGLE_OAUTH_CLIENT_SECRET`, `SLACK_CLIENT_ID`, `SLACK_CLIENT_SECRET`

### Deploy to Cloud Run

```bash
# Build and push image
gcloud builds submit --tag gcr.io/YOUR_PROJECT/kno-agent

# Deploy
gcloud run deploy kno-agent \
  --image gcr.io/YOUR_PROJECT/kno-agent \
  --region us-central1 \
  --set-env-vars GOOGLE_GENAI_USE_VERTEXAI=1,GOOGLE_CLOUD_PROJECT=YOUR_PROJECT \
  --set-secrets ENCRYPTION_KEY=ENCRYPTION_KEY:latest,...
```

### Ingest Confluence into the knowledge base

Run once after connecting Jira/Confluence in Settings, or whenever pages are updated:

```bash
python /tmp/kno_ingest_firestore.py
# Or via the admin endpoint:
# POST https://kno.xnukernel.com/admin/ingest/confluence
```

---

## Local Development

```bash
git clone https://github.com/DhanapalAshokkumar/kno-ai.git
cd kno-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Set env vars
export GOOGLE_GENAI_USE_VERTEXAI=1
export GOOGLE_CLOUD_PROJECT=your-project-id
export DEV_USER_EMAIL=you@example.com   # bypasses Cloudflare auth locally
export ENCRYPTION_KEY=your-fernet-key

# Authenticate
gcloud auth application-default login

# Run
uvicorn kno.prod_app:app --reload --port 8080
```

Open [http://localhost:8080](http://localhost:8080).

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/me` | Current user info + connected apps |
| `GET` | `/auth/google` | Start Gmail/Drive OAuth flow |
| `GET` | `/auth/slack` | Start Slack OAuth flow |
| `POST` | `/connect/github` | Connect GitHub PAT |
| `POST` | `/connect/jira` | Connect Jira/Confluence |
| `POST` | `/connect/zoho` | Connect Zoho CRM |
| `DELETE` | `/disconnect/{app}` | Disconnect an app |
| `POST` | `/chat` | Send message, get cited response + session_id |
| `GET` | `/admin/users` | Admin: all users + connected apps |
| `POST` | `/admin/ingest/confluence` | Admin: ingest Confluence into knowledge base |
| `GET` | `/admin/kb/stats` | Admin: knowledge base document count |

---

## License

MIT
