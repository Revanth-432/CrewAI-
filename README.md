# CREWAI — Multi-Agent AI Projects Hub

FastAPI exposes ten CrewAI workflows in a single, self-contained service (`app.py`).

## Quickstart

### 1. Install dependencies
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
# source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure Environment
Set `GEMINI_API_KEY` in your environment or in a `.env` file:
```bash
GEMINI_API_KEY=your_gemini_api_key_here
# Optional configuration:
# GEMINI_MODEL=gemini/gemini-3.1-flash-lite
# CREWAI_DATA_DIR=generated
```

### 3. Run the Server
```bash
uvicorn app:app --reload
```

- Interactive Swagger API documentation: `http://127.0.0.1:8000/docs`
- Health check: `GET http://127.0.0.1:8000/health`

---

## API Contract

Text-based routes accept:
```json
{
  "input": "text supplied to the selected project"
}
```

Successful responses return:
```json
{
  "output": "the project result as text"
}
```

Structured results are returned as JSON-formatted strings within the `output` field; prose and articles are returned as Markdown.

---

## Agent Endpoints

| Route | Agent / Flow | Input Type | Description |
|---|---|---|---|
| `/agent1` | Blog Post Crew | JSON `{"input":"..."}` | Researches and writes a ~600-word blog post from a topic. |
| `/agent2` | Competitor Analysis Crew | JSON `{"input":"..."}` | Exactly 3 public HTTP(S) competitor URLs (one per line). |
| `/agent3` | Meeting Notes Crew | JSON `{"input":"..."}` | Summarizes meeting transcript with action items and decisions. |
| `/agent4` | Data Report Crew | Multipart `file` (CSV) | Upload a `.csv` file (max 10 MB); computes statistics and insights. |
| `/agent5` | Customer Support Flow | JSON `{"input":"..."}` | Labeled fields: `Ticket ID:`, `Customer Name:`, `Issue Description:`. |
| `/agent6` | Resume Screening Flow | Multipart Form | `job_description` (text) and `resumes` (one or more PDF files). |
| `/agent7` | Bug Triage Flow | JSON `{"input":"..."}` | Labeled fields: `Issue Title:` and `Issue Body:`. |
| `/agent8` | Content Repurposing Flow | JSON `{"input":"..."}` | Repurposes article text into LinkedIn post, Twitter thread, and newsletter. |
| `/agent9` | Lead Qualification Flow | JSON `{"input":"..."}` | Labeled fields: `Lead Name:` plus `Website:` or `Business Description:`. |
| `/agent10` | Approval Report Flow | JSON `{"input":"..."}` | Multi-step research report with `START` and `REVIEW` approval states. |
| `/artifacts/{id}` | Artifact Store | GET | Download generated CSV/Markdown artifacts. |

---

### Agent 10: Approval Workflow Commands

**Start a new report draft:**
```text
START
Topic: The impact of AI on software development
Audience: Engineering Directors
Sources: Optional research material or URLs
```

*The response returns a `report_id` and initial draft.*

**Request revisions:**
```text
REVIEW
Report ID: <report_id>
Decision: revise
Feedback: Please expand on security implications.
```

**Approve and finalize report:**
```text
REVIEW
Report ID: <report_id>
Decision: approved
```

---

## Project Structure

This deployment repository consists solely of:
- `app.py` — Complete FastAPI application containing all 10 CrewAI workflows, parsers, and stores.
- `requirements.txt` — Python dependencies.
- `README.md` — API documentation and setup instructions.
