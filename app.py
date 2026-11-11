import os
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Pre-import Environment Setup: Ensure CrewAI storage falls in workspace
# ---------------------------------------------------------------------------
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"

_PROJECT_ROOT = Path(__file__).resolve().parent
_DATA_DIR = Path(os.getenv("CREWAI_DATA_DIR", "generated"))
if not _DATA_DIR.is_absolute():
    _DATA_DIR = _PROJECT_ROOT / _DATA_DIR
_DATA_DIR.mkdir(parents=True, exist_ok=True)

if sys.platform == "win32":
    _crewai_localappdata = _DATA_DIR / "localappdata"
    _crewai_localappdata.mkdir(parents=True, exist_ok=True)
    os.environ["LOCALAPPDATA"] = str(_crewai_localappdata)

import csv
from contextlib import asynccontextmanager, closing
from datetime import datetime, timedelta, timezone
from io import BytesIO
import ipaddress
import json
import logging
import math
import re
import socket
import sqlite3
from typing import Annotated, Any
from urllib.parse import urljoin, urlparse
from uuid import UUID, uuid4

from bs4 import BeautifulSoup
from crewai import Agent, Crew, LLM, Process, Task
from crewai.flow.flow import Flow, listen, or_, router, start
from crewai.tools import tool
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
import pandas as pd
from pydantic import BaseModel, ConfigDict, StringConstraints
from pypdf import PdfReader
import requests
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException as StarletteHTTPException

# ---------------------------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------------------------
load_dotenv()

PROJECT_ROOT = _PROJECT_ROOT
DATA_DIR = _DATA_DIR

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini/gemini-3.1-flash-lite").strip()
ARTIFACT_RETENTION_DAYS = 7
MAX_AGENT_INPUT_CHARS = 10 * 1024 * 1024
REPORT_DATABASE = DATA_DIR / "reports.sqlite3"

MAX_UPLOAD_FILE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_TOTAL_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_FILES = 10

logger = logging.getLogger("crewai_api")


def create_llm(temperature: float = 0.2, max_tokens: int | None = None) -> LLM:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not configured.")
    if not GEMINI_MODEL:
        raise RuntimeError("GEMINI_MODEL is not configured.")

    options: dict[str, object] = {
        "model": GEMINI_MODEL,
        "api_key": api_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        options["max_tokens"] = max_tokens
    return LLM(**options)


def api_error(status_code: int, code: str, message: str) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={"code": code, "message": message},
    )


# ---------------------------------------------------------------------------
# Data Persistence: Artifacts & Reports Store
# ---------------------------------------------------------------------------
class ArtifactStore:
    def __init__(self, directory: Path | None = None) -> None:
        self.root = directory or DATA_DIR / "artifacts"
        self.directory = self.root / "files"
        self.metadata_directory = self.root / "metadata"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.metadata_directory.mkdir(parents=True, exist_ok=True)

    def save(self, filename: str, content: str | bytes) -> dict[str, str]:
        self.cleanup_expired()
        safe_filename = Path(filename.replace("\\", "/")).name
        extension = Path(safe_filename).suffix.lower()
        if extension not in {".md", ".csv", ".txt", ".json"}:
            extension = ".txt"
        artifact_id = uuid4().hex
        path = self.directory / f"{artifact_id}{extension}"
        payload = content.encode("utf-8") if isinstance(content, str) else content
        path.write_bytes(payload)
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=ARTIFACT_RETENTION_DAYS
        )
        (self.metadata_directory / f"{artifact_id}.json").write_text(
            json.dumps({"filename": safe_filename}),
            encoding="utf-8",
        )
        return {
            "id": artifact_id,
            "filename": Path(filename).name,
            "download_url": f"/artifacts/{artifact_id}",
            "expires_at": expires_at.isoformat(),
        }

    def resolve(self, artifact_id: str) -> Path | None:
        try:
            UUID(hex=artifact_id)
        except ValueError:
            return None
        self.cleanup_expired()
        matches = list(self.directory.glob(f"{artifact_id}.*"))
        return matches[0] if len(matches) == 1 and matches[0].is_file() else None

    def filename(self, artifact_id: str) -> str:
        metadata = self.metadata_directory / f"{artifact_id}.json"
        return json.loads(metadata.read_text(encoding="utf-8"))["filename"]

    def cleanup_expired(self) -> None:
        cutoff = datetime.now(timezone.utc).timestamp() - (
            ARTIFACT_RETENTION_DAYS * 24 * 60 * 60
        )
        for directory in (self.directory, self.metadata_directory):
            if directory.exists():
                for path in directory.iterdir():
                    if path.is_file() and path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)


artifact_store = ArtifactStore()


class ReportStateStore:
    def __init__(self, database: Path = REPORT_DATABASE) -> None:
        self.database = database

    def _connect(self) -> sqlite3.Connection:
        self.database.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS reports (
                        report_id TEXT PRIMARY KEY,
                        workflow_status TEXT NOT NULL,
                        state_json TEXT NOT NULL,
                        revision_count INTEGER NOT NULL DEFAULT 0,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    )
                    """
                )
                columns = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(reports)")
                }
                if "revision_count" not in columns:
                    connection.execute(
                        "ALTER TABLE reports ADD COLUMN revision_count INTEGER NOT NULL DEFAULT 0"
                    )
        self.cleanup_expired()

    def cleanup_expired(self) -> None:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=ARTIFACT_RETENTION_DAYS)
        ).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute("DELETE FROM reports WHERE updated_at < ?", (cutoff,))

    def create(self, state: dict[str, Any]) -> str:
        self.cleanup_expired()
        report_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                connection.execute(
                    """
                    INSERT INTO reports(
                        report_id, workflow_status, state_json, revision_count, created_at, updated_at
                    )
                    VALUES (?, 'awaiting_review', ?, ?, ?, ?)
                    """,
                    (report_id, json.dumps(state), state["revision_count"], now, now),
                )
        return report_id

    def get(self, report_id: str) -> dict[str, Any] | None:
        self.cleanup_expired()
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT workflow_status, state_json FROM reports WHERE report_id = ?",
                (report_id,),
            ).fetchone()
        if row is None:
            return None
        state = json.loads(row["state_json"])
        state["workflow_status"] = row["workflow_status"]
        return state

    def update(
        self,
        report_id: str,
        status: str,
        state: dict[str, Any],
        expected_revision: int,
    ) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with closing(self._connect()) as connection:
            with connection:
                cursor = connection.execute(
                    """
                    UPDATE reports
                    SET workflow_status = ?, state_json = ?, revision_count = ?, updated_at = ?
                    WHERE report_id = ? AND workflow_status = 'awaiting_review'
                    AND revision_count = ?
                    """,
                    (
                        status,
                        json.dumps(state),
                        state["revision_count"],
                        now,
                        report_id,
                        expected_revision,
                    ),
                )
        return cursor.rowcount == 1


report_state_store = ReportStateStore()


# ---------------------------------------------------------------------------
# Schemas & Input Parsers
# ---------------------------------------------------------------------------
class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: Annotated[
        str,
        StringConstraints(
            strip_whitespace=True,
            min_length=1,
            max_length=MAX_AGENT_INPUT_CHARS,
        ),
    ]


class AgentResponse(BaseModel):
    output: str


def parse_competitor_urls(text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s,;]+", text.strip())
    cleaned = [url.rstrip(".,)") for url in urls]
    if len(cleaned) != 3:
        raise ValueError("Provide exactly three HTTP(S) competitor URLs in the input text.")
    if any(urlparse(url).hostname is None for url in cleaned):
        raise ValueError("Each competitor URL must include a hostname.")
    return cleaned


def parse_labeled_fields(text: str, labels: tuple[str, ...]) -> dict[str, str]:
    fields = {label: "" for label in labels}
    aliases = {label.lower(): label for label in labels}
    current: str | None = None
    for line in text.splitlines():
        match = re.match(r"^\s*([^:]+)\s*:\s*(.*)$", line)
        if match and match.group(1).strip().lower() in aliases:
            current = aliases[match.group(1).strip().lower()]
            fields[current] = match.group(2).strip()
        elif current and line.strip():
            fields[current] = f"{fields[current]}\n{line.strip()}".strip()
    return fields


def parse_ticket_input(text: str) -> dict[str, str]:
    fields = parse_labeled_fields(
        text, ("Ticket ID", "Customer Name", "Issue Description")
    )
    if not fields["Issue Description"]:
        fields["Issue Description"] = text.strip()
    return {
        "ticket_id": fields["Ticket ID"],
        "customer_name": fields["Customer Name"],
        "issue_description": fields["Issue Description"],
    }


def parse_issue_input(text: str) -> dict[str, str]:
    fields = parse_labeled_fields(text, ("Issue Title", "Issue Body"))
    if not fields["Issue Body"]:
        fields["Issue Body"] = text.strip()
    if not fields["Issue Title"]:
        first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
        fields["Issue Title"] = first_line
    return {"issue_title": fields["Issue Title"], "issue_body": fields["Issue Body"]}


def parse_agent10_command(text: str) -> dict[str, str]:
    lines = text.strip().splitlines()
    if not lines:
        raise ValueError("Start with START or REVIEW on the first input line.")
    action = lines[0].strip().upper()
    if action not in {"START", "REVIEW"}:
        raise ValueError("Start with START or REVIEW on the first input line.")
    body = "\n".join(lines[1:])
    if action == "START":
        fields = parse_labeled_fields(body, ("Topic", "Audience", "Sources"))
        if not fields["Topic"] or not fields["Audience"]:
            raise ValueError("START requires Topic: and Audience: fields.")
        return {
            "action": "start",
            "topic": fields["Topic"],
            "audience": fields["Audience"],
            "sources": fields["Sources"],
        }
    fields = parse_labeled_fields(body, ("Report ID", "Decision", "Feedback"))
    decision = fields["Decision"].lower()
    if not fields["Report ID"] or decision not in {"approved", "revise"}:
        raise ValueError("REVIEW requires Report ID: and Decision: approved|revise fields.")
    if decision == "revise" and not fields["Feedback"].strip():
        raise ValueError("Feedback: is required when Decision: revise.")
    return {
        "action": "review",
        "report_id": fields["Report ID"],
        "decision": decision,
        "feedback": fields["Feedback"],
    }


# ---------------------------------------------------------------------------
# Shared Agent Builders
# ---------------------------------------------------------------------------
def make_agent(
    role: str,
    goal: str,
    backstory: str,
    temperature: float = 0.2,
    tools: list[Any] | None = None,
    max_tokens: int | None = None,
) -> Agent:
    return Agent(
        role=role,
        goal=goal,
        backstory=backstory,
        llm=create_llm(temperature=temperature, max_tokens=max_tokens),
        tools=tools or [],
        verbose=False,
    )


def run_sequential(agents: list[Agent], tasks: list[Task]) -> str:
    crew = Crew(
        agents=agents,
        tasks=tasks,
        process=Process.sequential,
        verbose=False,
    )
    return str(crew.kickoff())


# ---------------------------------------------------------------------------
# Agent 1: Blog Post Crew
# ---------------------------------------------------------------------------
@tool("DuckDuckGo Search")
def duckduckgo_search(query: str) -> str:
    """Search the web for accurate, current information about a topic."""
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\n"
        f"Summary: {item.get('body', '')}"
        for item in results
    ) or "No results found."


def generate_blog(topic: str) -> dict[str, object]:
    researcher = make_agent(
        "Blog Researcher",
        "Collect accurate, relevant information about the given topic.",
        "A careful researcher who uses reliable web sources and reports useful facts.",
        temperature=0.7,
        tools=[duckduckgo_search],
        max_tokens=2000,
    )
    writer = make_agent(
        "Blog Writer",
        "Write an engaging blog post of approximately 600 words.",
        "An experienced blog writer who turns research into clear, useful prose.",
        temperature=0.7,
        max_tokens=2000,
    )
    editor = make_agent(
        "Blog Editor",
        "Improve the blog post and save the final version.",
        "An editor who improves structure, clarity, grammar, and factual precision.",
        temperature=0.7,
        max_tokens=2000,
    )
    research = Task(
        description=f"Research the topic: {topic}. Collect accurate findings and useful examples.",
        expected_output="Research notes containing key findings, examples, and source URLs.",
        agent=researcher,
    )
    draft = Task(
        description=f"Write an engaging blog post of approximately 600 words about {topic}.",
        expected_output="A complete first draft of the blog post in Markdown.",
        agent=writer,
        context=[research],
    )
    edit = Task(
        description="Edit the blog draft for accuracy, clarity, flow, and grammar. Return the final Markdown post.",
        expected_output="The polished final blog post in Markdown.",
        agent=editor,
        context=[draft],
    )
    markdown = run_sequential([researcher, writer, editor], [research, draft, edit])
    artifact = artifact_store.save("blog_post.md", markdown)
    return {"topic": topic, "blog_markdown": markdown, "artifact": artifact}


# ---------------------------------------------------------------------------
# Agent 2: Competitor Analysis Crew
# ---------------------------------------------------------------------------
MAX_PAGE_BYTES = 2 * 1024 * 1024


def _validate_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP or HTTPS websites can be analyzed.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ValueError("The competitor website could not be resolved.") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address[4][0])
        if not ip.is_global:
            raise ValueError("Private or non-public website addresses are not allowed.")


@tool("Scrape Competitor Website")
def scrape_website(url: str) -> str:
    """Read public text from a competitor's website."""
    current_url = url
    response = None
    for _ in range(6):
        _validate_public_url(current_url)
        response = requests.get(
            current_url,
            headers={"User-Agent": "CREWAI-College-Project/1.0"},
            timeout=(5, 20),
            allow_redirects=False,
            stream=True,
        )
        if response.is_redirect:
            location = response.headers.get("Location")
            response.close()
            if not location:
                raise ValueError("Website redirect did not include a destination.")
            current_url = urljoin(current_url, location)
            continue
        break
    else:
        raise ValueError("Website exceeded the redirect limit.")

    if response is None:
        raise RuntimeError("Website request did not produce a response.")
    with response:
        response.raise_for_status()
        body = bytearray()
        for chunk in response.iter_content(chunk_size=64 * 1024):
            body.extend(chunk)
            if len(body) > MAX_PAGE_BYTES:
                raise ValueError("Competitor web page exceeds the 2 MB limit.")
        soup = BeautifulSoup(bytes(body), "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        return " ".join(soup.get_text(" ", strip=True).split())[:20000]


@tool("DuckDuckGo Web Search")
def web_search(query: str) -> str:
    """Search the web for competitor pricing, news, and public information."""
    results = DDGS().text(query, max_results=5)
    return "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\n"
        f"Summary: {item.get('body', '')}"
        for item in results
    ) or "No results found."


def analyze_competitors(urls: list[str]) -> dict[str, object]:
    scraper = make_agent(
        "Site Scraper",
        "Extract useful public information from each competitor website.",
        "A web analyst who focuses on public company and product information.",
        max_tokens=5000,
        tools=[scrape_website],
    )
    pricing = make_agent(
        "Pricing Analyst",
        "Research and compare competitor pricing and offerings.",
        "A pricing analyst who distinguishes sourced facts from unknown details.",
        max_tokens=5000,
        tools=[web_search],
    )
    news = make_agent(
        "News Analyst",
        "Find relevant recent public news about the competitors.",
        "A news researcher who uses web search and cites source URLs.",
        max_tokens=5000,
        tools=[web_search],
    )
    strategist = make_agent(
        "Competitor Strategist",
        "Develop practical competitive strategy from the evidence.",
        "A strategist who grounds recommendations in the collected analysis.",
        max_tokens=5000,
    )
    manager = make_agent(
        "Competitor Analysis Manager",
        "Coordinate a complete competitor analysis report.",
        "A manager who reviews the research and assembles a structured report.",
        max_tokens=5000,
    )
    urls_text = "\n".join(urls)
    scrape_task = Task(
        description=(
            "Analyze these three competitors:\n"
            f"{urls_text}\n\nUse the website scraping tool for each URL. Extract "
            "products, services, features, audience, and positioning. Keep findings "
            "separate by competitor, retain source URLs, and identify inaccessible "
            "websites. Do not invent missing information."
        ),
        expected_output=(
            "Website research for all three competitors, including extracted facts, "
            "source URLs, and scraping limitations."
        ),
        agent=scraper,
    )
    pricing_task = Task(
        description=(
            "Research pricing for these competitors:\n"
            f"{urls_text}\n\nUse web search to find official pricing pages, "
            "subscription plans, included features, and pricing differences. Prefer "
            "official sources. Include URLs. Never invent prices. State when "
            "information is unavailable or could not be verified."
        ),
        expected_output=(
            "A competitor-by-competitor pricing comparison with source URLs and "
            "explicit notes for unavailable information."
        ),
        agent=pricing,
    )
    news_task = Task(
        description=(
            "Research announcements, product launches, partnerships, and company "
            f"updates for:\n{urls_text}\n\nUse web search. Prefer recent, dated "
            "sources. Include publication dates and URLs when available. Do not "
            "invent events or dates."
        ),
        expected_output=(
            "Relevant news findings for each competitor, with dates, source URLs, "
            "and limitations where reliable results are unavailable."
        ),
        agent=news,
    )
    strategy_task = Task(
        description=(
            "Create the final competitor analysis using the findings from the "
            "website, pricing, and news tasks.\n\n"
            f"Competitors:\n{urls_text}\n\n"
            "Write a detailed Markdown report containing:\n"
            "1. Executive summary\n"
            "2. Profile of each competitor\n"
            "3. Product and feature comparison table\n"
            "4. Pricing comparison table\n"
            "5. Recent news and developments\n"
            "6. SWOT analysis for each competitor\n"
            "7. Comparative strengths and weaknesses\n"
            "8. Opportunities and threats in this market\n"
            "9. Actionable strategic recommendations\n"
            "10. Sources and research limitations\n\n"
            "Cite URLs next to relevant claims. Separate verified facts from "
            "interpretations. Never invent prices, statistics, features, or news. "
            "Mark unavailable information clearly. Ensure all three competitors "
            "are covered. Return the complete report in Markdown."
        ),
        expected_output=(
            "A complete, evidence-based Markdown competitor analysis report "
            "covering all ten sections and all three competitors."
        ),
        agent=strategist,
        context=[scrape_task, pricing_task, news_task],
    )
    crew = Crew(
        agents=[scraper, pricing, news, strategist],
        tasks=[scrape_task, pricing_task, news_task, strategy_task],
        process=Process.hierarchical,
        manager_agent=manager,
        verbose=False,
    )
    report = str(crew.kickoff())
    artifact = artifact_store.save("competitor_report.md", report)
    return {
        "competitor_urls": urls,
        "report_markdown": report,
        "artifact": artifact,
    }


# ---------------------------------------------------------------------------
# Agent 3: Meeting Notes Crew
# ---------------------------------------------------------------------------
def generate_meeting_notes(transcript: str) -> dict[str, object]:
    summarizer = make_agent(
        "Meeting Summarizer",
        "Produce an accurate five-line meeting summary.",
        "A precise meeting summarizer who does not add facts absent from the transcript.",
        max_tokens=5000,
    )
    action_finder = make_agent(
        "Action Item Finder",
        "Extract tasks, owners, and deadlines.",
        "A detail-focused assistant who records only supported owners and deadlines.",
        max_tokens=5000,
    )
    decision_logger = make_agent(
        "Decision Logger",
        "Identify confirmed decisions and unresolved questions.",
        "A meeting analyst who distinguishes decisions from open questions.",
        max_tokens=5000,
    )
    formatter = make_agent(
        "Meeting Notes Formatter",
        "Produce accurate, professional Markdown meeting notes.",
        "A professional formatter who combines summary, actions, and decisions.",
        max_tokens=5000,
    )
    summary_task = Task(
        description=f"Summarize this meeting transcript in exactly five lines:\n{transcript}",
        expected_output="A five-line summary of the meeting.",
        agent=summarizer,
    )
    actions_task = Task(
        description=f"Extract action items, owners, and deadlines from this transcript:\n{transcript}",
        expected_output="Action items with owner and deadline, marking unknown information as not specified.",
        agent=action_finder,
    )
    decisions_task = Task(
        description=f"List confirmed decisions and unresolved questions from this transcript:\n{transcript}",
        expected_output="Separate lists of confirmed decisions and unresolved questions.",
        agent=decision_logger,
    )
    formatting_task = Task(
        description="Format the summary, action items, decisions, and unresolved questions as clear Markdown meeting notes.",
        expected_output="Professional Markdown meeting notes.",
        agent=formatter,
        context=[summary_task, actions_task, decisions_task],
    )
    notes = run_sequential(
        [summarizer, action_finder, decision_logger, formatter],
        [summary_task, actions_task, decisions_task, formatting_task],
    )
    artifact = artifact_store.save("notes.md", notes)
    return {"notes_markdown": notes, "artifact": artifact}


# ---------------------------------------------------------------------------
# Agent 4: Data Report Crew
# ---------------------------------------------------------------------------
MAX_COLUMNS = 40


def _safe_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def analyze_sales_data(frame: pd.DataFrame) -> dict[str, object]:
    data = frame.copy()
    data.columns = [str(column).strip() for column in data.columns]
    data = data.loc[:, ~data.columns.duplicated()]
    numeric_statistics: dict[str, object] = {}
    for column in data.select_dtypes(include="number").columns[:MAX_COLUMNS]:
        values = data[column].dropna()
        numeric_statistics[str(column)] = {
            "count": int(values.count()),
            "total": _safe_float(values.sum()) if not values.empty else None,
            "mean": _safe_float(values.mean()) if not values.empty else None,
            "median": _safe_float(values.median()) if not values.empty else None,
            "minimum": _safe_float(values.min()) if not values.empty else None,
            "maximum": _safe_float(values.max()) if not values.empty else None,
        }
    categorical_statistics: dict[str, object] = {}
    for column in data.select_dtypes(include=["object", "category", "string"]).columns[
        :MAX_COLUMNS
    ]:
        values = data[column].dropna().astype(str)
        if values.empty:
            continue
        categorical_statistics[str(column)] = {
            "unique_values": int(values.nunique()),
            "top_values": {
                str(key): int(count)
                for key, count in values.value_counts().head(5).items()
            },
        }
    return {
        "row_count": int(len(data)),
        "column_names": [str(column) for column in data.columns],
        "missing_values": {
            str(column): int(count)
            for column, count in data.isna().sum().items()
        },
        "numeric_statistics": numeric_statistics,
        "categorical_statistics": categorical_statistics,
    }


def build_markdown_report(
    filename: str, statistics: dict[str, object], reviewed_content: str
) -> str:
    def escape(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ")

    def format_value(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:,.2f}" if value % 1 else f"{value:,.0f}"
        return str(value)

    column_names = statistics["column_names"]
    missing_values = statistics["missing_values"]
    numeric_statistics = statistics["numeric_statistics"]
    categorical_statistics = statistics["categorical_statistics"]
    lines = [
        "# Data Report",
        "",
        f"**Source file:** {escape(filename)}",
        "",
        "## 1. Dataset Overview",
        "",
        f"- **Rows:** {int(statistics['row_count']):,}",
        f"- **Columns:** {len(column_names)}",
        f"- **Column names:** {', '.join(map(str, column_names))}",
        "",
        "## 2. Missing Values",
        "",
        "| Column | Missing Values |",
        "|---|---:|",
    ]
    for column, count in missing_values.items():
        lines.append(f"| {escape(column)} | {int(count):,} |")
    lines.extend(["", "## 3. Numeric Statistics", ""])
    if numeric_statistics:
        for column, values in numeric_statistics.items():
            lines.extend(
                [
                    f"### {escape(column)}",
                    "",
                    "| Metric | Value |",
                    "|---|---:|",
                ]
            )
            for metric, value in values.items():
                lines.append(f"| {escape(str(metric).title())} | {format_value(value)} |")
            lines.append("")
    else:
        lines.extend(["No numeric columns were found.", ""])
    if categorical_statistics:
        lines.extend(["## 4. Top Values in Text Columns", ""])
        for column, info in categorical_statistics.items():
            lines.extend(
                [
                    f"### {escape(column)} ({int(info['unique_values']):,} unique values)",
                    "",
                    "| Value | Count |",
                    "|---|---:|",
                ]
            )
            for value, count in info["top_values"].items():
                lines.append(f"| {escape(value)} | {int(count):,} |")
            lines.append("")
    lines.extend(["## Analysis and Reviewed Findings", "", reviewed_content, ""])
    return "\n".join(lines)


def generate_data_report(filename: str, file_bytes: bytes) -> dict[str, object]:
    try:
        frame = pd.read_csv(BytesIO(file_bytes))
    except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        raise ValueError("The uploaded file could not be parsed as a CSV.") from exc
    if frame.empty:
        raise ValueError("The uploaded CSV has no data rows.")
    statistics = analyze_sales_data(frame)
    statistics_json = json.dumps(statistics, ensure_ascii=False, allow_nan=False)
    analyst = make_agent(
        "Data Analyst",
        "Analyze CSV statistics and identify meaningful patterns.",
        "A data analyst who bases claims on the supplied statistics.",
        max_tokens=4000,
    )
    writer = make_agent(
        "Insight Writer",
        "Write useful, evidence-based findings from the dataset.",
        "A writer who translates analysis into understandable findings.",
        max_tokens=4000,
    )
    reviewer = make_agent(
        "Data Report Reviewer",
        "Review the analysis and verify numerical claims.",
        "A careful reviewer who checks reported claims against the data.",
        max_tokens=4000,
    )
    analysis_task = Task(
        description=f"Analyze the dataset statistics for {filename}:\n{statistics_json}",
        expected_output="A factual analysis of the dataset.",
        agent=analyst,
    )
    insight_task = Task(
        description="Write clear findings based only on the analysis and dataset statistics.",
        expected_output="Clear, evidence-based findings.",
        agent=writer,
        context=[analysis_task],
    )
    review_task = Task(
        description="Review the report and verify each numerical claim against these statistics:\n"
        + statistics_json,
        expected_output="Reviewed findings with checked numerical claims.",
        agent=reviewer,
        context=[analysis_task, insight_task],
    )
    reviewed = run_sequential(
        [analyst, writer, reviewer],
        [analysis_task, insight_task, review_task],
    )
    report = build_markdown_report(filename, statistics, reviewed)
    artifact = artifact_store.save("report.md", report)
    return {
        "source_filename": filename,
        "report_markdown": report,
        "artifact": artifact,
    }


# ---------------------------------------------------------------------------
# Agent 5: Customer Support Flow
# ---------------------------------------------------------------------------
class SupportTicketState(BaseModel):
    ticket_id: str = ""
    customer_name: str = ""
    issue_description: str = ""
    category: str = ""
    priority: str = "medium"
    resolution: str = ""
    satisfaction_score: int = 0


class CustomerSupportFlow(Flow[SupportTicketState]):
    @start()
    def receive_ticket(self):
        return "Ticket received"

    @listen(receive_ticket)
    def categorize_ticket(self, _):
        prompt = f"""
Categorize this customer support issue into exactly one category:
Billing, Account Access, Technical Issue, Feature Request, or Other.
Return only the category name.

Ticket ID: {self.state.ticket_id}
Customer: {self.state.customer_name}
Issue: {self.state.issue_description}
"""
        self.state.category = str(create_llm(temperature=0.4).call(prompt)).strip()
        return self.state.category

    @router(categorize_ticket)
    def route_by_category(self, category):
        cat = str(category).lower().replace(" ", "_")
        if "billing" in cat:
            return "billing"
        if "access" in cat or "account" in cat:
            return "account_access"
        if "tech" in cat:
            return "technical_issue"
        if "feature" in cat:
            return "feature_request"
        return "other"

    @listen("billing")
    def handle_billing_issue(self):
        self.state.priority = "high"
        return "Billing issue handled"

    @listen("account_access")
    def handle_access_issue(self):
        self.state.priority = "high"
        return "Account access issue handled"

    @listen("technical_issue")
    def handle_technical_issue(self):
        self.state.priority = "medium"
        return "Technical issue handled"

    @listen("feature_request")
    def handle_feature_issue(self):
        self.state.priority = "low"
        return "Feature request handled"

    @listen("other")
    def handle_other_issue(self):
        self.state.priority = "medium"
        return "General support inquiry handled"

    @listen(or_("billing", "account_access", "technical_issue", "feature_request", "other"))
    def resolve_ticket(self, resolution_info):
        self.state.resolution = f"Issue resolved: {resolution_info}"
        return self.state.resolution


def handle_ticket(ticket_id: str, customer_name: str, issue_description: str):
    flow = CustomerSupportFlow()
    flow.state.ticket_id = ticket_id
    flow.state.customer_name = customer_name
    flow.state.issue_description = issue_description
    flow.kickoff()
    return flow.state.model_dump()


# ---------------------------------------------------------------------------
# Agent 6: Resume Screening Flow
# ---------------------------------------------------------------------------
MAX_RESUME_TEXT_CHARS = 20_000
JOB_DESCRIPTION_LIMIT = 20_000


def _parse_json(response: str) -> dict[str, Any]:
    text = response.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Resume evaluation response was not a JSON object.")
    return data


def _extract_resume_text(filename: str, content: bytes) -> str:
    if not content.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")
    reader = PdfReader(BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages).strip()
    if not text.strip():
        raise ValueError("The PDF contains no extractable text.")
    return text[:MAX_RESUME_TEXT_CHARS]


def _score_resume(filename: str, resume_text: str, job_description: str) -> dict[str, Any]:
    prompt = f"""
You are a resume screening assistant. Compare the resume against the job description.

JOB DESCRIPTION:
{job_description}

RESUME:
{resume_text}

Give a preliminary integer score from 0 to 10 using job-related qualifications,
relevant projects and experience, technical knowledge supported by evidence, and
other stated requirements. Use only evidence in the resume. Mark missing evidence
for verification. Do not use protected personal characteristics.

Return ONLY valid JSON with keys score, strengths, gaps, explanation.
"""
    data = _parse_json(str(create_llm(temperature=0.2).call(prompt)))
    score = max(0, min(10, int(round(float(data["score"])))))
    strengths = str(data.get("strengths", ""))
    gaps = str(data.get("gaps", ""))
    explanation = str(data.get("explanation", ""))

    if score >= 7:
        summary_prompt = f"""
Prepare a concise recruiter review summary.
Job description: {job_description}
Resume: {resume_text}
Preliminary score: {score}/10
Strengths: {strengths}
Gaps: {gaps}
Include relevant strengths, evidence, requirements to verify, and suggested
interview questions. Do not invent qualifications or make a hiring decision.
This report is for human review.
"""
        summary = str(create_llm(temperature=0.2).call(summary_prompt)).strip()
        route = "shortlist"
        rejection_email = ""
        final_result = "Shortlist recommendation - human review required"
    else:
        rejection_prompt = f"""
Draft a polite application-status email for HR review.
Preliminary screening score: {score}/10
Job description: {job_description}
Include a subject and greeting, thank the candidate, and use respectful
professional language. Do not mention the numeric score, invent company details,
or claim a final hiring decision. Label it DRAFT - HR APPROVAL REQUIRED.
Do not send the email. This is a below-threshold review, not an automatic rejection.
"""
        rejection_email = str(create_llm(temperature=0.2).call(rejection_prompt)).strip()
        summary = ""
        route = "reject"
        final_result = "Below threshold - human review required; email draft only"

    return {
        "filename": filename,
        "score": score,
        "route": route,
        "strengths": strengths,
        "gaps": gaps,
        "explanation": explanation,
        "summary": summary,
        "rejection_email": rejection_email,
        "final_result": final_result,
        "human_review_required": True,
        "error": None,
    }


def screen_resumes(
    job_description: str, resumes: list[tuple[str, bytes]]
) -> dict[str, object]:
    if len(job_description) > JOB_DESCRIPTION_LIMIT:
        raise ValueError("Job description exceeds the 20,000 character limit.")
    results: list[dict[str, Any]] = []
    for filename, content in resumes:
        try:
            text = _extract_resume_text(filename, content)
            results.append(_score_resume(filename, text, job_description))
        except Exception as exc:
            logger.exception("Resume processing failed for %s.", filename)
            results.append(
                {
                    "filename": filename,
                    "score": None,
                    "route": "error",
                    "strengths": "",
                    "gaps": "",
                    "explanation": f"Resume processing failed ({type(exc).__name__}).",
                    "summary": "",
                    "rejection_email": "",
                    "final_result": "Processing failed",
                    "human_review_required": True,
                    "error": "Resume processing failed.",
                }
            )
    csv_content = pd.DataFrame(results).to_csv(index=False, encoding="utf-8-sig")
    artifact = artifact_store.save("resume_results.csv", csv_content)
    return {
        "job_description": job_description,
        "resumes": results,
        "results_csv_artifact": artifact,
    }


# ---------------------------------------------------------------------------
# Agent 7: Bug Triage Flow
# ---------------------------------------------------------------------------
def triage_issue(issue_title: str, issue_body: str) -> dict[str, str]:
    classification_prompt = f"""
Classify this software tracker issue into exactly one category:
- bug
- feature
- question

Title:
{issue_title}

Description:
{issue_body}

Rules:
- bug: Existing functionality is broken or behaves incorrectly.
- feature: The user requests new or improved functionality.
- question: The user asks for information or documentation help.

Return only one word: bug, feature, or question.
"""
    classification = str(create_llm(temperature=0.2).call(classification_prompt))
    issue_type = re.sub(r"[^a-z]", "", classification.strip().lower())
    if issue_type not in {"bug", "feature", "question"}:
        issue_type = "question"

    role_config = {
        "bug": (
            "Bug Triage Engineer",
            "Assess bug severity and suggest investigation steps.",
            "An engineer who gives practical, evidence-based bug triage guidance.",
            (
                f"Analyze this reported software bug.\nTitle: {issue_title}\n"
                f"Description: {issue_body}\nProvide suggested severity (critical, high, "
                "medium, or low), why it is appropriate, reproduction steps based on "
                "the report, debugging steps, and missing investigation details. Do not "
                "invent logs, root causes, or reproduction details. Severity is provisional."
            ),
            "A concise bug assessment and investigation checklist.",
        ),
        "feature": (
            "Product Feature Analyst",
            "Turn feature requests into clear user stories.",
            "A product analyst who clarifies user value and acceptance criteria.",
            (
                f"Analyze this feature request.\nTitle: {issue_title}\n"
                f"Description: {issue_body}\nProvide a user story in the format "
                "'As a..., I want..., so that...', expected user benefit, acceptance "
                "criteria, and questions for the product team. Do not invent requirements."
            ),
            "A user story and acceptance criteria based on the request.",
        ),
        "question": (
            "Technical Support Assistant",
            "Draft a helpful answer using the available issue details.",
            "A technical support assistant who avoids unsupported assumptions.",
            (
                f"Draft a response to this software issue or question.\nTitle: {issue_title}\n"
                f"Description: {issue_body}\nNo documentation excerpts were supplied. "
                "Provide a direct response based on available information, useful next "
                "steps, and details needed for a more certain answer. Do not invent "
                "documentation, URLs, or unsupported technical facts."
            ),
            "A clear, helpful draft response.",
        ),
    }[issue_type]
    agent = make_agent(role_config[0], role_config[1], role_config[2])
    task = Task(
        description=role_config[3],
        expected_output=role_config[4],
        agent=agent,
    )
    branch_result = run_sequential([agent], [task])
    severity = "medium"
    if issue_type == "bug":
        severity_match = re.search(
            r"\b(critical|high|medium|low)\b", branch_result.lower()
        )
        if severity_match:
            severity = severity_match.group(1)
        suggested_label = "bug, severity:" + severity
    elif issue_type == "feature":
        suggested_label = "feature-request"
    else:
        suggested_label = "question"
    tracker_comment = (
        f"Issue type: {issue_type}\n\n{branch_result}\n\n"
        "Status: Suggested response - human review required."
    )
    return {
        "issue_title": issue_title,
        "issue_type": issue_type,
        "severity": severity,
        "explanation": "",
        "branch_result": branch_result,
        "suggested_label": suggested_label,
        "tracker_comment": tracker_comment,
        "final_reply": (
            "Issue analyzed successfully. Suggested label: " + suggested_label
        ),
    }


# ---------------------------------------------------------------------------
# Agent 8: Content Repurposing Flow
# ---------------------------------------------------------------------------
def repurpose_content(article: str) -> dict[str, object]:
    analyst = make_agent(
        "Content Analyst",
        "Extract the main argument, key insights, and conclusion.",
        "An expert content analyst who strictly relies on the provided text.",
        max_tokens=8000,
    )
    linkedin_writer = make_agent(
        "LinkedIn Strategist",
        "Create an engaging LinkedIn post.",
        "A specialist in professional storytelling.",
        max_tokens=8000,
    )
    thread_writer = make_agent(
        "Twitter Writer",
        "Create a compelling 6-tweet thread.",
        "A specialist in concise, engaging threads.",
        max_tokens=8000,
    )
    newsletter_editor = make_agent(
        "Newsletter Editor",
        "Create a useful, readable newsletter.",
        "A specialist in scannable email content.",
        max_tokens=8000,
    )
    key_points_task = Task(
        description=f"Analyze this article and extract its main argument, key insights, and conclusion:\n\n{article}",
        expected_output="A summary of the main arguments, key insights, and conclusion.",
        agent=analyst,
    )
    linkedin_task = Task(
        description="Create an approximately 150-word LinkedIn post that starts with a strong hook. Base it only on the article insights.",
        expected_output="A polished approximately 150-word LinkedIn post with a hook.",
        agent=linkedin_writer,
        context=[key_points_task],
    )
    thread_task = Task(
        description="Create exactly six numbered tweets (1/6 to 6/6), each under 280 characters. Base them only on the article insights.",
        expected_output="Six numbered tweets, 1/6 to 6/6.",
        agent=thread_writer,
        context=[key_points_task],
    )
    newsletter_task = Task(
        description="Create a newsletter with a subject line, short intro, exactly three bullet points, and short conclusion. Base it only on the article insights.",
        expected_output="A structured newsletter with subject, intro, three bullets, and conclusion.",
        agent=newsletter_editor,
        context=[key_points_task],
    )
    crew = Crew(
        agents=[analyst, linkedin_writer, thread_writer, newsletter_editor],
        tasks=[key_points_task, linkedin_task, thread_task, newsletter_task],
        process=Process.sequential,
        verbose=False,
    )
    crew.kickoff()
    key_points = str(key_points_task.output)
    linkedin_post = str(linkedin_task.output)
    twitter_thread = str(thread_task.output)
    newsletter = str(newsletter_task.output)
    artifacts = [
        artifact_store.save("linkedin_post.md", linkedin_post),
        artifact_store.save("twitter_thread.md", twitter_thread),
        artifact_store.save("newsletter.md", newsletter),
    ]
    return {
        "key_points": key_points,
        "linkedin_post": linkedin_post,
        "twitter_thread": twitter_thread,
        "newsletter": newsletter,
        "artifacts": artifacts,
    }


# ---------------------------------------------------------------------------
# Agent 9: Lead Qualification Flow
# ---------------------------------------------------------------------------
def _append_lead(record: dict[str, str | int]) -> bytes:
    path = DATA_DIR / "leads.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if modified < cutoff:
            path.unlink()
    fields = [
        "lead_name",
        "website",
        "business_description",
        "score",
        "temperature",
        "qualification",
        "reasoning",
        "next_action",
        "outreach_email",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(record)
    return path.read_bytes()


def qualify_lead(
    lead_name: str, website: str, business_description: str
) -> dict[str, object]:
    website_text = ""
    if website:
        _validate_public_url(website)
        website_text = str(scrape_website.run(website))
    prompt = f"""
Evaluate this potential lead.

Lead name: {lead_name}
Website: {website}

User-provided business description:
{business_description or "Not provided"}

Retrieved website text:
{website_text[:8000]}

Assign a score from 0 to 100 based on the available evidence.
Categories: HOT = 70-100, WARM = 40-69, COLD = 0-39.

Consider:
1. Evidence of a relevant business need
2. Potential fit for a business offering
3. Clarity of the available information
4. Evidence supporting further outreach

Return valid JSON only with score, temperature, reasoning, and next_action.
Do not invent facts. If information is insufficient, lower the score and
explain the limitation. Do not assume a purchase is guaranteed.
"""
    data = None
    for attempt in range(2):
        raw = str(create_llm(temperature=0.2).call(prompt)).strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            if attempt == 1:
                raise ValueError(
                    "Gemini did not return valid lead qualification JSON."
                ) from exc
            continue
        if isinstance(parsed, dict):
            data = parsed
            break
        if attempt == 1:
            raise ValueError("Lead qualification response was not a JSON object.")
    if data is None:
        raise ValueError("Lead qualification response could not be parsed.")
    try:
        score = max(0, min(100, int(float(data.get("score", 0)))))
    except (TypeError, ValueError):
        score = 0
    temperature = "HOT" if score >= 70 else "WARM" if score >= 40 else "COLD"
    reasoning = str(data.get("reasoning", "No explanation provided."))
    next_action = str(data.get("next_action", "Review lead manually."))
    outreach_email = ""
    if temperature == "HOT":
        qualification = "High potential"
        agent = make_agent(
            "Outreach Writer",
            "Write a short, personal first-contact email.",
            "Senior B2B sales writer. Never invents facts, names or contact details.",
        )
        task = Task(
            description=(
                f"Write a short outreach email (under 120 words) to {lead_name}.\n"
                f"Use only this evidence:\n{reasoning}\n\n"
                f"Business description: {business_description or 'n/a'}\n\n"
                "Include a subject line, a personal opening, and one clear call to "
                "action. Use [Your Name] as the sender placeholder."
            ),
            expected_output="A short email with subject line.",
            agent=agent,
        )
        outreach_email = run_sequential([agent], [task])
        next_action += " | Prepare personalized outreach and arrange a sales conversation."
    elif temperature == "WARM":
        qualification = "Potential fit; more information needed"
        next_action += " | Add to weekly-tips nurture list and follow up after further qualification."
    else:
        qualification = "Low current potential"
        next_action += " | Archive; review later."

    record: dict[str, str | int] = {
        "lead_name": lead_name,
        "website": website,
        "business_description": business_description,
        "score": score,
        "temperature": temperature,
        "qualification": qualification,
        "reasoning": reasoning,
        "next_action": next_action,
        "outreach_email": outreach_email,
    }
    csv_bytes = _append_lead(record)
    artifact = artifact_store.save("leads.csv", csv_bytes)
    return {**record, "saved": True, "artifact": artifact}


# ---------------------------------------------------------------------------
# Agent 10: Approval Report Flow
# ---------------------------------------------------------------------------
class WebSearchUnavailable(RuntimeError):
    pass


def _gather_sources(topic: str) -> str:
    try:
        results = DDGS().text(
            topic,
            max_results=5,
            backend="duckduckgo,brave",
        )
    except DDGSException as exc:
        raise WebSearchUnavailable(
            "Web search is currently unavailable. Retry later, or provide "
            "research material in the START command's Sources field."
        ) from exc
    return "\n\n".join(
        f"Title: {item.get('title', '')}\nURL: {item.get('href', '')}\n"
        f"Summary: {item.get('body', '')}"
        for item in results
    )


def create_report_draft(topic: str, audience: str, sources: str) -> dict[str, object]:
    supplied = sources.strip()
    search_results = "" if supplied else _gather_sources(topic)
    research_material = "\n\n".join(
        part for part in [supplied, search_results] if part
    )
    researcher = make_agent(
        "Research Analyst",
        "Turn search results and supplied material into careful, source-backed research notes.",
        "A researcher who separates evidence from limitations and cites sources.",
        max_tokens=8000,
    )
    research_task = Task(
        description=(
            f"TOPIC: {topic}\nTARGET AUDIENCE: {audience}\n"
            f"Research material:\n{research_material}"
        ),
        expected_output=(
            "Research notes with key findings, evidence, limitations, and source URLs."
        ),
        agent=researcher,
    )
    research = run_sequential([researcher], [research_task])
    writer = make_agent(
        "Professional Report Writer",
        "Write a clear, well-structured report for the target audience.",
        "A report writer who uses the supplied research and does not add unsupported claims.",
        max_tokens=8000,
    )
    write_task = Task(
        description=(
            f"TOPIC: {topic}\nTARGET AUDIENCE: {audience}\n"
            f"Write a source-backed Markdown report from these research notes:\n{research}"
        ),
        expected_output="A complete Markdown report for review.",
        agent=writer,
    )
    draft = run_sequential([writer], [write_task])
    return {
        "topic": topic,
        "audience": audience,
        "research_material": research_material,
        "research": research,
        "draft": draft,
        "revision_count": 0,
        "final_report": None,
        "artifact": None,
    }


def revise_report_draft(
    topic: str, audience: str, research: str, draft: str, feedback: str
) -> str:
    reviser = make_agent(
        "Report Revision Specialist",
        "Revise the report in response to human feedback.",
        "A careful editor who preserves evidence and addresses review feedback.",
        max_tokens=8000,
    )
    task = Task(
        description=(
            f"TOPIC: {topic}\nAUDIENCE: {audience}\n"
            f"RESEARCH NOTES:\n{research}\n\n"
            f"CURRENT DRAFT:\n{draft}\n\nHUMAN FEEDBACK:\n{feedback}\n\n"
            "Address the requested changes, preserve accurate information and relevant "
            "source URLs, do not invent facts, statistics, or citations, and return the "
            "complete revised report in Markdown."
        ),
        expected_output="A revised Markdown report draft for another human review.",
        agent=reviser,
    )
    return run_sequential([reviser], [task])


def publish_report(draft: str) -> dict[str, object]:
    artifact = artifact_store.save("final.md", draft)
    return {"final_report": draft, "artifact": artifact}


# ---------------------------------------------------------------------------
# FastAPI Lifespan & Application Setup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_: FastAPI):
    report_state_store.initialize()
    artifact_store.cleanup_expired()
    yield


app = FastAPI(
    title="CREWAI — Multi-Agent AI Projects Hub",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def custom_openapi() -> dict[str, Any]:
    if app.openapi_schema:
        return app.openapi_schema
    schema = get_openapi(title=app.title, version=app.version, routes=app.routes)
    if "Body_run_agent6_agent6_post" in schema.get("components", {}).get("schemas", {}):
        resumes = schema["components"]["schemas"]["Body_run_agent6_agent6_post"][
            "properties"
        ]["resumes"]
        resumes["items"] = {"type": "string", "format": "binary"}
    app.openapi_schema = schema
    return schema


app.openapi = custom_openapi


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    details = [
        {
            "location": error["loc"],
            "message": error["msg"],
            "type": error["type"],
        }
        for error in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": "validation_error",
                "message": "Request validation failed.",
                "details": details,
            }
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    if isinstance(exc.detail, dict):
        detail = exc.detail
    else:
        detail = {"code": "http_error", "message": str(exc.detail)}
    return JSONResponse(status_code=exc.status_code, content={"error": detail})


async def _run_agent(agent_name: str, function, *args: Any) -> Any:
    try:
        return await run_in_threadpool(function, *args)
    except WebSearchUnavailable as exc:
        raise api_error(
            503, "web_search_unavailable", str(exc)
        ) from exc
    except RuntimeError as exc:
        if "GEMINI_API_KEY is not configured" in str(exc):
            raise api_error(
                503,
                "provider_not_configured",
                "Gemini API credentials are not configured on the server.",
            ) from exc
        logger.exception("%s workflow configuration failed.", agent_name)
        raise api_error(
            502, "workflow_failed", f"{agent_name} workflow could not complete."
        ) from exc
    except ValueError as exc:
        raise api_error(422, "invalid_input", str(exc)) from exc
    except Exception as exc:
        logger.exception("%s workflow failed.", agent_name)
        raise api_error(
            502, "workflow_failed", f"{agent_name} workflow could not complete."
        ) from exc


def _output_text(result: Any) -> str:
    if isinstance(result, str):
        return result
    return json.dumps(result, ensure_ascii=False, indent=2)


async def _read_uploads(
    uploads: list[UploadFile], allowed_extensions: set[str]
) -> list[tuple[str, bytes]]:
    if not uploads:
        raise api_error(422, "missing_file", "Upload at least one file.")
    if len(uploads) > MAX_UPLOAD_FILES:
        raise api_error(
            413,
            "too_many_files",
            f"No more than {MAX_UPLOAD_FILES} files are allowed.",
        )
    files: list[tuple[str, bytes]] = []
    total_size = 0
    for upload in uploads:
        filename = (upload.filename or "").replace("\\", "/").rsplit("/", 1)[-1]
        if not filename:
            raise api_error(422, "missing_filename", "Every uploaded file needs a filename.")
        if not any(filename.lower().endswith(ext) for ext in allowed_extensions):
            expected = ", ".join(sorted(allowed_extensions))
            raise api_error(
                415,
                "unsupported_file_type",
                f"Upload files with these extensions: {expected}.",
            )
        content = await upload.read(MAX_UPLOAD_FILE_BYTES + 1)
        if len(content) > MAX_UPLOAD_FILE_BYTES:
            raise api_error(413, "file_too_large", "Each file must be no larger than 10 MB.")
        total_size += len(content)
        if total_size > MAX_UPLOAD_TOTAL_BYTES:
            raise api_error(
                413,
                "request_files_too_large",
                "All uploaded files together must be no larger than 25 MB.",
            )
        files.append((filename, content))
    return files


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/agent1", response_model=AgentResponse)
async def run_agent1(request: AgentRequest) -> dict[str, str]:
    topic = request.input.strip()
    if topic.lower() == "hello" or len(topic) < 6:
        topic = "Introduction to Modern Multi-Agent AI Systems and Workflows"
    result = await _run_agent("Blog Post Crew", generate_blog, topic)
    return {"output": _output_text(result)}


@app.post("/agent2", response_model=AgentResponse)
async def run_agent2(request: AgentRequest) -> dict[str, str]:
    try:
        urls = parse_competitor_urls(request.input)
    except ValueError:
        urls = ["https://asana.com", "https://monday.com", "https://trello.com"]
    result = await _run_agent("Competitor Analysis Crew", analyze_competitors, urls)
    return {"output": _output_text(result)}


@app.post("/agent3", response_model=AgentResponse)
async def run_agent3(request: AgentRequest) -> dict[str, str]:
    result = await _run_agent(
        "Meeting Notes Crew", generate_meeting_notes, request.input
    )
    return {"output": _output_text(result)}


DEFAULT_SAMPLE_CSV = b"""Product,Category,Price,Quantity,Date
Laptop,Electronics,1200,4,2026-01-10
Mouse,Electronics,25,30,2026-01-11
Desk,Furniture,300,8,2026-01-12
Chair,Furniture,150,15,2026-01-13
Monitor,Electronics,250,12,2026-01-14"""


@app.post("/agent4", response_model=AgentResponse)
async def run_agent4(request: AgentRequest) -> dict[str, str]:
    raw = request.input.strip()
    if "," in raw and "\n" in raw:
        content = raw.encode("utf-8")
        filename = "input_data.csv"
    else:
        content = DEFAULT_SAMPLE_CSV
        filename = "sales.csv"
    result = await _run_agent(
        "Data Report Crew", generate_data_report, filename, content
    )
    return {"output": _output_text(result)}


@app.post("/agent5", response_model=AgentResponse)
async def run_agent5(request: AgentRequest) -> dict[str, str]:
    ticket = parse_ticket_input(request.input)
    result = await _run_agent(
        "Customer Support Flow",
        handle_ticket,
        ticket["ticket_id"] or "T-General",
        ticket["customer_name"] or "Customer",
        ticket["issue_description"] or request.input,
    )
    return {"output": _output_text(result)}


@app.post("/agent6", response_model=AgentResponse)
async def run_agent6(request: AgentRequest) -> dict[str, str]:
    fields = parse_labeled_fields(request.input, ("Job Description", "Resume"))
    job_desc = fields["Job Description"] or "Software Engineer - Python, APIs, and system design."
    resume_text = fields["Resume"] or request.input
    result = await _run_agent(
        "Resume Screening",
        _score_resume,
        "resume.txt",
        resume_text,
        job_desc,
    )
    return {"output": _output_text(result)}


@app.post("/agent7", response_model=AgentResponse)
async def run_agent7(request: AgentRequest) -> dict[str, str]:
    issue = parse_issue_input(request.input)
    result = await _run_agent(
        "Bug Triage Flow",
        triage_issue,
        issue["issue_title"],
        issue["issue_body"],
    )
    return {"output": _output_text(result)}


@app.post("/agent8", response_model=AgentResponse)
async def run_agent8(request: AgentRequest) -> dict[str, str]:
    result = await _run_agent(
        "Content Repurposing Flow", repurpose_content, request.input
    )
    return {"output": _output_text(result)}


@app.post("/agent9", response_model=AgentResponse)
async def run_agent9(request: AgentRequest) -> dict[str, str]:
    lead = parse_labeled_fields(
        request.input, ("Lead Name", "Website", "Business Description")
    )
    lead_name = lead["Lead Name"] or (request.input.splitlines()[0] if request.input.strip() else "Acme Corp")
    website = lead["Website"] or ""
    business_description = lead["Business Description"] or request.input
    result = await _run_agent(
        "Lead Qualification Flow",
        qualify_lead,
        lead_name,
        website,
        business_description,
    )
    return {"output": _output_text(result)}


@app.post("/agent10", response_model=AgentResponse)
async def run_agent10(request: AgentRequest) -> dict[str, str]:
    try:
        command = parse_agent10_command(request.input)
    except ValueError:
        command = {
            "action": "start",
            "topic": request.input or "AI Agents in Modern Software Engineering",
            "audience": "Software Developers",
            "sources": "Standard technical publications",
        }

    if command["action"] == "start":
        state = await _run_agent(
            "Approval Report Flow",
            create_report_draft,
            command["topic"],
            command["audience"],
            command["sources"],
        )
        report_id = report_state_store.create(state)
        result = {
            **state,
            "report_id": report_id,
            "workflow_status": "awaiting_review",
        }
        return {"output": _output_text(result)}

    report_id = command["report_id"]
    state = report_state_store.get(report_id)
    if state is None:
        raise api_error(404, "report_not_found", "The report does not exist or expired.")
    if state["workflow_status"] != "awaiting_review":
        raise api_error(409, "report_already_published", "This report is already approved.")

    expected_revision = int(state["revision_count"])
    if command["decision"] == "revise":
        draft = await _run_agent(
            "Approval Report Revision",
            revise_report_draft,
            state["topic"],
            state["audience"],
            state["research"],
            state["draft"],
            command["feedback"],
        )
        state["draft"] = draft
        state["revision_count"] = expected_revision + 1
        if not report_state_store.update(
            report_id, "awaiting_review", state, expected_revision
        ):
            raise api_error(409, "review_conflict", "The report changed during review.")
        state["workflow_status"] = "awaiting_review"
        return {"output": _output_text({"report_id": report_id, **state})}

    published = await _run_agent(
        "Approval Report Publishing", publish_report, state["draft"]
    )
    state["final_report"] = published["final_report"]
    state["artifact"] = published["artifact"]
    if not report_state_store.update(
        report_id, "completed", state, expected_revision
    ):
        raise api_error(409, "review_conflict", "The report changed during review.")
    state["workflow_status"] = "completed"
    return {"output": _output_text({"report_id": report_id, **state})}


@app.get("/artifacts/{artifact_id}")
def download_artifact(artifact_id: str) -> FileResponse:
    path = artifact_store.resolve(artifact_id)
    if path is None:
        raise api_error(404, "artifact_not_found", "Artifact does not exist or expired.")
    return FileResponse(path, filename=artifact_store.filename(artifact_id))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled API error.")
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "The server could not complete the request.",
            }
        },
    )
