import json
import os
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)
DB_PATH = BASE_DIR / "recallops.db"
BANK_ID = os.getenv("HINDSIGHT_BANK_ID", "recallops-incidents")

app = FastAPI(title="RecallOps AI", description="Incident intelligence powered by persistent Hindsight memory")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# Hindsight is the persistent memory layer. The app remains runnable in local-demo mode
# if credentials are not configured, but demo mode is not the final hackathon integration.
hindsight = None
hindsight_error = None
hindsight_ready = False
if os.getenv("HINDSIGHT_API_KEY"):
    try:
        from hindsight_client import Hindsight
        hindsight = Hindsight(
            base_url=os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"),
            api_key=os.getenv("HINDSIGHT_API_KEY"),
        )
    except Exception as exc:
        hindsight_error = str(exc)
        hindsight = None


@app.on_event("startup")
async def initialize_hindsight_bank():
    """Create/update the Hindsight bank using the SDK's native async API.

    Calling the synchronous create_bank() during module import conflicts with
    Uvicorn's already-running event loop. acreate_bank() avoids nested loops.
    """
    global hindsight_error, hindsight_ready
    if hindsight is None:
        hindsight_error = hindsight_error or "HINDSIGHT_API_KEY is not configured."
        return
    try:
        await hindsight.acreate_bank(
            bank_id=BANK_ID,
            name="RecallOps Production Incidents",
            background=(
                "Memory bank for software production incidents, root causes, mitigations, "
                "and verified resolutions. Prefer evidence-backed fixes and distinguish "
                "confirmed causes from hypotheses."
            ),
        )
        hindsight_ready = True
        hindsight_error = None
    except Exception as exc:
        hindsight_ready = False
        hindsight_error = str(exc)
        print("Hindsight bank initialization failed:", hindsight_error)

llm = OpenAI(
    api_key=os.getenv("GROQ_API_KEY", "missing-key"),
    base_url="https://api.groq.com/openai/v1",
) if os.getenv("GROQ_API_KEY") else None


def init_db():
    with sqlite3.connect(DB_PATH) as con:
        con.execute("""CREATE TABLE IF NOT EXISTS incidents (
            id TEXT PRIMARY KEY, created_at TEXT NOT NULL, service TEXT NOT NULL,
            severity TEXT NOT NULL, title TEXT NOT NULL, logs TEXT NOT NULL,
            root_cause TEXT NOT NULL, resolution TEXT NOT NULL, confidence TEXT NOT NULL,
            status TEXT NOT NULL, memory_used INTEGER NOT NULL DEFAULT 0
        )""")
        con.commit()

init_db()

class IncidentIn(BaseModel):
    service: str = Field(min_length=2, max_length=100)
    severity: str = Field(default="High")
    title: str = Field(min_length=4, max_length=180)
    logs: str = Field(min_length=10, max_length=12000)

class ResolveIn(BaseModel):
    resolution: str = Field(min_length=5, max_length=4000)
    outcome: str = Field(default="resolved")


def local_recall(query: str, limit: int = 5) -> list[dict[str, Any]]:
    # Fallback only for local UI testing. Hindsight is used whenever configured.
    terms = [t.lower() for t in re.findall(r"[a-zA-Z0-9_]+", query) if len(t) > 3]
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT 100").fetchall()
    ranked = []
    for row in rows:
        hay = (row["service"] + " " + row["title"] + " " + row["logs"] + " " + row["root_cause"] + " " + row["resolution"]).lower()
        score = sum(1 for t in terms if t in hay)
        if score:
            ranked.append((score, dict(row)))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in ranked[:limit]]


def recall_memories(query: str) -> tuple[list[str], bool]:
    if hindsight_ready and hindsight:
        try:
            result = hindsight.recall(bank_id=BANK_ID, query=query, budget="mid", max_tokens=3000)
            return [m.text for m in result.results], True
        except Exception as exc:
            print("Hindsight recall failed:", exc)
    return [
        f"Past incident: {x['title']} | service={x['service']} | root cause={x['root_cause']} | resolution={x['resolution']} | status={x['status']}"
        for x in local_recall(query)
    ], False


def analyse_with_llm(incident: IncidentIn, memories: list[str]) -> dict[str, str]:
    if llm:
        prompt = f"""You are RecallOps, a cautious site reliability engineering incident assistant.
Analyze the incident using the supplied past incident memories. Do not claim certainty without evidence.
Return ONLY valid JSON with keys: summary, root_cause, resolution_steps, confidence, memory_insight, safety_note.
resolution_steps should be concise numbered steps in plain text. If evidence is insufficient, say what to inspect next.

INCIDENT
Service: {incident.service}
Severity: {incident.severity}
Title: {incident.title}
Logs:
{incident.logs}

PAST HINDSIGHT MEMORIES:
{json.dumps(memories, ensure_ascii=False)}
"""
        try:
            response = llm.chat.completions.create(
                model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content or "{}"
            data = json.loads(raw)
            return {k: str(data.get(k, "Not enough evidence; inspect metrics and recent changes.")) for k in ["summary", "root_cause", "resolution_steps", "confidence", "memory_insight", "safety_note"]}
        except Exception as exc:
            print("Groq analysis failed:", exc)

    # Deterministic local fallback for setup/testing; not represented as LLM output.
    text = (incident.title + " " + incident.logs).lower()
    if "too many connections" in text or "connection pool" in text:
        root = "Likely database connection pool exhaustion. Confirm active connections, pool saturation, and connection leaks before changing limits."
        steps = "1. Check active and idle DB connections.\n2. Inspect application pool metrics and connection acquisition timeouts.\n3. Check for connections not being closed.\n4. Apply a temporary safe pool adjustment only after checking database capacity.\n5. Monitor saturation and error rate."
    elif "timeout" in text or "timed out" in text:
        root = "Likely request or dependency timeout. Correlate latency, dependency health, and recent deployments before assigning a definitive cause."
        steps = "1. Identify the slow dependency from traces.\n2. Compare latency and error rate with the last healthy period.\n3. Check recent deployments and configuration changes.\n4. Roll back only if evidence links the change to the incident.\n5. Verify recovery with metrics."
    elif "memory" in text or "oomkilled" in text:
        root = "Possible memory pressure or memory leak. Validate container memory usage, limits, and restart history."
        steps = "1. Inspect memory usage over time and container restart events.\n2. Check recent traffic and deployment changes.\n3. Capture heap/profile data if safe.\n4. Scale or roll back according to the incident runbook.\n5. Verify memory stabilizes."
    else:
        root = "Insufficient evidence for a reliable root-cause conclusion. The logs need correlation with metrics, traces, and recent changes."
        steps = "1. Identify the first failing timestamp.\n2. Check service health, dependency status, and error-rate metrics.\n3. Inspect distributed traces for the failing request.\n4. Compare deployments and configuration changes.\n5. Record the verified cause and resolution after mitigation."
    insight = "No matching prior incident was available; this is a first-pass analysis." if not memories else "Related past incident context was found and should be checked against current telemetry."
    return {"summary": "Incident triage completed using local fallback rules.", "root_cause": root,
            "resolution_steps": steps, "confidence": "Low–medium (heuristic fallback)",
            "memory_insight": insight, "safety_note": "Do not run destructive commands or change production settings without an authorised engineer and evidence."}


@app.get("/", response_class=HTMLResponse)
def home():
    return (BASE_DIR / "templates" / "index.html").read_text()

@app.get("/api/status")
def status():
    return {"hindsight_connected": hindsight_ready, "groq_connected": llm is not None,
            "mode": "live" if hindsight_ready and llm else "local-demo", "bank_id": BANK_ID,
            "hindsight_error": hindsight_error}

@app.get("/api/incidents")
def incidents():
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        rows = con.execute("SELECT * FROM incidents ORDER BY created_at DESC LIMIT 50").fetchall()
    return [dict(r) for r in rows]

@app.post("/api/analyse")
def analyse(incident: IncidentIn):
    query = f"Similar production incidents, root causes and verified fixes for service {incident.service}. Symptoms: {incident.title}. Logs: {incident.logs[:2500]}"
    memories, used_hindsight = recall_memories(query)
    result = analyse_with_llm(incident, memories)
    incident_id = str(uuid.uuid4())[:8].upper()
    now = datetime.now(timezone.utc).isoformat()
    with sqlite3.connect(DB_PATH) as con:
        con.execute("INSERT INTO incidents (id,created_at,service,severity,title,logs,root_cause,resolution,confidence,status,memory_used) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (incident_id, now, incident.service, incident.severity, incident.title, incident.logs,
                     result["root_cause"], result["resolution_steps"], result["confidence"], "Investigating", int(bool(memories))))
        con.commit()
    return {"id": incident_id, "created_at": now, **result, "memories": memories,
            "memory_count": len(memories), "hindsight_used": used_hindsight,
            "llm_used": llm is not None}

@app.post("/api/incidents/{incident_id}/resolve")
def resolve(incident_id: str, payload: ResolveIn):
    with sqlite3.connect(DB_PATH) as con:
        con.row_factory = sqlite3.Row
        row = con.execute("SELECT * FROM incidents WHERE id=?", (incident_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Incident not found")
        con.execute("UPDATE incidents SET resolution=?, status=? WHERE id=?",
                    (payload.resolution, "Resolved" if payload.outcome == "resolved" else "Unresolved", incident_id))
        con.commit()
    memory_text = (f"Production incident {row['id']} for service {row['service']}. Severity: {row['severity']}. "
                   f"Title: {row['title']}. Logs/symptoms: {row['logs']}. "
                   f"Root cause identified: {row['root_cause']}. Resolution applied: {payload.resolution}. "
                   f"Outcome: {payload.outcome}. Recorded at: {datetime.now(timezone.utc).isoformat()}.")
    stored = False
    if hindsight_ready and hindsight:
        try:
            hindsight.retain(bank_id=BANK_ID, content=memory_text,
                             context="Verified production incident postmortem and resolution",
                             metadata={"incident_id": incident_id, "service": row["service"], "outcome": payload.outcome})
            stored = True
        except Exception as exc:
            print("Hindsight retain failed:", exc)
    return {"ok": True, "stored_in_hindsight": stored, "message": "Resolution saved. " + ("Hindsight memory updated." if stored else "Local record saved; configure Hindsight to persist agent memory.")}

@app.get("/api/metrics")
def metrics():
    with sqlite3.connect(DB_PATH) as con:
        total = con.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        resolved = con.execute("SELECT COUNT(*) FROM incidents WHERE status='Resolved'").fetchone()[0]
        remembered = con.execute("SELECT COUNT(*) FROM incidents WHERE memory_used=1").fetchone()[0]
    return {"total": total, "resolved": resolved, "remembered": remembered}
