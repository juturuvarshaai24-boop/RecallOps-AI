# RecallOps AI

Memory-powered incident response assistant for Hack With Bangalore 4.0.

## Features
- Incident intake and triage
- Groq-powered incident analysis
- Hindsight persistent memory (`recall` before analysis, `retain` after a verified resolution)
- Incident history and memory-aware recommendations
- Local demo fallback for UI development when API keys are not configured

## Setup (macOS)

```bash
cd recallops-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Add your Hindsight Cloud API key and Groq API key to `.env`.

- Hindsight Cloud: https://ui.hindsight.vectorize.io/
- Hindsight docs: https://hindsight.vectorize.io/sdks/python
- Groq: https://console.groq.com/keys

Then run:

```bash
uvicorn app:app --reload
```

Open http://127.0.0.1:8000

## Hindsight memory design
- One memory bank: `recallops-incidents`
- Before analysing an incident, RecallOps calls `recall()` with the service, title and log symptoms.
- After a human confirms the actual resolution, RecallOps calls `retain()` with symptoms, root cause, fix and outcome.
- This creates an explicit learning loop: recall previous evidence → investigate → human verifies fix → retain postmortem.

## Important
The local-demo fallback is for development only. For the hackathon demonstration, configure both Hindsight and Groq and verify `/api/status` reports `live` before recording the demo. Never execute production-changing commands automatically; recommendations require an authorised engineer to verify them.
