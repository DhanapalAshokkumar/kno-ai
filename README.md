# kno.ai

> Ask one question. Get answers from every tool your company uses.

kno.ai is an AI agent that connects business tools — Gmail, Google Drive, Jira, Confluence, Slack — so employees can find information across all their company systems in a single chat, without switching tabs or knowing where to look.

Built for the **Google for Startups AI Agents Challenge**.

---

## The Problem

Company knowledge is scattered. An engineer needs the Q3 roadmap — it might be in a Confluence doc, a Drive slide, a Slack thread, or buried in email. They waste 20–30 minutes searching across tools. kno.ai answers in seconds.

---

## How It Works

kno.ai is a conversational agent powered by Google ADK and Gemini 2.5 Flash. You ask a natural language question; kno searches your connected tools, reads the relevant content, and gives you a cited answer with source links.

```
You: "What did the CEO say about the reorg last week?"
kno: Found 2 relevant emails from ceo@company.com (Apr 22)...
     [summary with direct Gmail links]
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Agent framework | [Google ADK](https://google.github.io/adk-docs/) |
| LLM | Gemini 2.5 Flash (via Vertex AI) |
| Email | Gmail API (OAuth 2.0) |
| Documents | Google Drive API (OAuth 2.0) |
| Runtime | Python 3.11+ |

---

## Connectors

### Available Now
- **Gmail** — search threads, read full email bodies
- **Google Drive** — search files, read Docs / Sheets / Slides content

### Roadmap
- [ ] Google Calendar — meeting context and scheduling
- [ ] Slack — search messages and channels
- [ ] Jira — query tickets, epics, and sprint status
- [ ] Confluence — search wiki pages and spaces
- [ ] GitHub — pull requests, issues, and code search
- [ ] Notion — pages and databases

---

## Local Setup

### Prerequisites
- Python 3.11+
- A Google Cloud project with Vertex AI enabled
- OAuth 2.0 credentials with Gmail and Drive scopes

### 1. Clone and install

```bash
git clone https://github.com/<your-username>/kno-ai.git
cd kno-ai
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Google Cloud project details
```

`.env` fields:
```
GOOGLE_GENAI_USE_VERTEXAI=1
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
```

### 3. Authenticate

```bash
gcloud auth application-default login \
  --scopes="https://www.googleapis.com/auth/gmail.readonly,\
https://www.googleapis.com/auth/drive.readonly,\
https://www.googleapis.com/auth/cloud-platform"
```

### 4. Run the agent

```bash
adk web
```

Open [http://localhost:8000](http://localhost:8000) and start asking questions.

---

## Project Structure

```
kno-ai/
├── kno/
│   ├── __init__.py
│   └── agent.py        # Agent definition and tool implementations
├── requirements.txt
├── .env.example
└── README.md
```

---

## License

MIT
