from __future__ import annotations

import io
from datetime import datetime

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy.orm import Session
from fastapi.staticfiles import StaticFiles
from pathlib import Path

import sys
import os

if __package__ is None or __name__ == "__main__":
    # Add current directory to sys.path so we can do absolute imports of local modules
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.append(current_dir)
    from db import SessionLocal, engine
    from config import get_settings
    from models import (
        AiEvaluation, Base, Candidate, FinalRanking, GithubAnalysis,
        InterviewEvent, PipelineRun, Ranking, ResumeExtraction, TestLink, TestResult
    )
    from pipeline_service import (
        create_and_init_run, recompute_final_scores_for_candidate, run_phase3_process_resumes,
        run_phase4_evaluate, run_phase5_github, run_phase6_rank, run_phase7_generate_test_links,
        run_phase7_send_test_emails, run_phase8_upload_tests_and_rank_final,
        run_phase9_schedule_interviews_dry_run, run_phase9_schedule_interviews_google_calendar,
        update_run_status, upsert_candidates_from_df
    )
    from schemas import (
        CreatePipelineRunResponse, JobDescriptionIn, LeaderboardResponse,
        LeaderboardRow, ScheduleIn, StartPhase7In, UploadTestResultsIn,
        UploadTestResultsResponse, WeightsPhase6In
    )
else:
    from .db import SessionLocal, engine
    from .config import get_settings
    from .models import (
        AiEvaluation, Base, Candidate, FinalRanking, GithubAnalysis,
        InterviewEvent, PipelineRun, Ranking, ResumeExtraction, TestLink, TestResult
    )
    from .pipeline_service import (
        create_and_init_run, recompute_final_scores_for_candidate, run_phase3_process_resumes,
        run_phase4_evaluate, run_phase5_github, run_phase6_rank, run_phase7_generate_test_links,
        run_phase7_send_test_emails, run_phase8_upload_tests_and_rank_final,
        run_phase9_schedule_interviews_dry_run, run_phase9_schedule_interviews_google_calendar,
        update_run_status, upsert_candidates_from_df
    )
    from .schemas import (
        CreatePipelineRunResponse, JobDescriptionIn, LeaderboardResponse,
        LeaderboardRow, ScheduleIn, StartPhase7In, UploadTestResultsIn,
        UploadTestResultsResponse, WeightsPhase6In
    )

from pydantic import BaseModel

settings = get_settings()


app = FastAPI(title="AI Candidate Screening Platform - Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def _startup() -> None:
    Base.metadata.create_all(bind=engine)


# Serve static demo UI (plain HTML/JS).
_STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


def _ensure_run_exists(db: Session, run_id: str) -> PipelineRun:
    run = db.get(PipelineRun, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found.")
    return run


def _cleanup_run_data(db: Session, run_id: str) -> None:
    # Remove run-scoped rows in a safe order.
    candidate_ids = [c.id for c in db.query(Candidate.id).filter(Candidate.run_id == run_id).all()]
    if not candidate_ids:
        return

    db.query(ResumeExtraction).filter(ResumeExtraction.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(AiEvaluation).filter(AiEvaluation.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(GithubAnalysis).filter(GithubAnalysis.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(Ranking).filter(Ranking.candidate_id.in_(candidate_ids)).delete(synchronize_session=False)
    db.query(TestResult).filter(TestResult.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(FinalRanking).filter(FinalRanking.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(TestLink).filter(TestLink.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(InterviewEvent).filter(InterviewEvent.candidate_id.in_(candidate_ids)).delete(
        synchronize_session=False
    )
    db.query(Candidate).filter(Candidate.id.in_(candidate_ids)).delete(synchronize_session=False)

    db.commit()


@app.post("/api/pipeline", response_model=CreatePipelineRunResponse)
def create_pipeline_run(db: Session = Depends(get_db)) -> CreatePipelineRunResponse:
    run = create_and_init_run(db)
    return CreatePipelineRunResponse(run_id=run.id)


@app.post("/api/pipeline/{run_id}/job-description")
def set_job_description(run_id: str, body: JobDescriptionIn, db: Session = Depends(get_db)) -> dict:
    run = _ensure_run_exists(db, run_id)
    run.job_description = body.job_description
    run.updated_at = datetime.utcnow()
    db.commit()
    return {"ok": True}


@app.post("/api/pipeline/{run_id}/candidates/csv")
async def upload_candidates_csv(
    run_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> dict:
    _ensure_run_exists(db, run_id)
    raw = await file.read()
    filename = file.filename or ""
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            sheet_dict = pd.read_excel(io.BytesIO(raw), sheet_name=None)
            df = pd.concat(sheet_dict.values(), ignore_index=True)
        else:
            # Fallback to CSV parsing.
            csv_text = raw.decode("utf-8-sig", errors="ignore")
            df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")

    try:
        # Cleanup only candidates + derived data for this run.
        _cleanup_run_data(db, run_id)
        upsert_candidates_from_df(db, run_id, df)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "rows": int(len(df))}


@app.post("/api/pipeline/{run_id}/start")
def start_pipeline_phase7(
    run_id: str,
    body: StartPhase7In,
    background: BackgroundTasks,
    db: Session = Depends(get_db),
) -> dict:
    run = _ensure_run_exists(db, run_id)
    if not run.job_description:
        raise HTTPException(status_code=400, detail="Set job description first.")

    run.w_resume = float(body.weights.w_resume)
    run.w_github = float(body.weights.w_github)
    run.w_cgpa = float(body.weights.w_cgpa)
    run.shortlist_threshold = float(body.weights.threshold)
    run.updated_at = datetime.utcnow()
    run.status = "running_phase7"
    db.commit()

    def task():
        local_db = SessionLocal()
        try:
            update_run_status(local_db, run_id, "running_phase3")
            _ = run_phase3_process_resumes(local_db, run_id)
            update_run_status(local_db, run_id, "running_phase4")
            _ = run_phase4_evaluate(local_db, run_id)
            update_run_status(local_db, run_id, "running_phase5")
            _ = run_phase5_github(local_db, run_id)
            update_run_status(local_db, run_id, "running_phase6")
            _, qualified = run_phase6_rank(local_db, run_id)
            update_run_status(local_db, run_id, f"running_phase7 (qualified={qualified})")
            _ = run_phase7_generate_test_links(local_db, run_id, body.test_link_base)
            update_run_status(local_db, run_id, "completed_phase7")
        except Exception as e:
            update_run_status(local_db, run_id, "failed", error=str(e)[:2000])
        finally:
            local_db.close()

    background.add_task(task)
    return {"ok": True, "run_id": run_id}


@app.post("/api/pipeline/{run_id}/test-results/csv", response_model=UploadTestResultsResponse)
async def upload_test_results_csv(
    run_id: str,
    test_threshold: float = 60.0,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> UploadTestResultsResponse:
    _ensure_run_exists(db, run_id)
    raw = await file.read()
    filename = file.filename or ""
    try:
        if filename.lower().endswith((".xlsx", ".xls")):
            sheet_dict = pd.read_excel(io.BytesIO(raw), sheet_name=None)
            df = pd.concat(sheet_dict.values(), ignore_index=True)
        else:
            csv_text = raw.decode("utf-8-sig", errors="ignore")
            df = pd.read_csv(io.StringIO(csv_text))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")
        
    try:
        qualified = run_phase8_upload_tests_and_rank_final(db, run_id, df, test_threshold=test_threshold)
        return UploadTestResultsResponse(qualified_count=int(qualified))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/api/pipeline/{run_id}/schedule-interviews")
def schedule_interviews(run_id: str, body: ScheduleIn, db: Session = Depends(get_db)) -> dict:
    _ensure_run_exists(db, run_id)
    # Real scheduling only if GOOGLE_CALENDAR_ENABLED + OAuth credentials are present.
    start_dt = datetime.fromisoformat(body.start_datetime.replace("Z", "+00:00"))
    if settings.GOOGLE_CALENDAR_ENABLED and Path(settings.GOOGLE_OAUTH_CREDENTIALS_JSON).exists():
        created = run_phase9_schedule_interviews_google_calendar(
            db=db,
            run_id=run_id,
            start_datetime=start_dt,
            slot_minutes=body.slot_minutes,
            timezone_str=body.timezone,
            calendar_id=body.calendar_id,
        )
    else:
        created = run_phase9_schedule_interviews_dry_run(
            db=db,
            run_id=run_id,
            start_datetime=start_dt,
            slot_minutes=body.slot_minutes,
            timezone_str=body.timezone,
        )
    return {"ok": True, "events_created": int(created)}


@app.get("/api/pipeline/{run_id}/leaderboard", response_model=LeaderboardResponse)
def leaderboard(run_id: str, db: Session = Depends(get_db)) -> LeaderboardResponse:
    run = _ensure_run_exists(db, run_id)

    # Determine if test results were uploaded (FinalRanking table rows exist).
    has_final = (
        db.query(FinalRanking).join(Candidate, Candidate.id == FinalRanking.candidate_id).filter(Candidate.run_id == run_id).first()
        is not None
    )

    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()
    rows: list[LeaderboardRow] = []

    for cand in candidates:
        ai = db.get(AiEvaluation, cand.id)
        gh = db.get(GithubAnalysis, cand.id)
        ranking = db.get(Ranking, cand.id)
        row = LeaderboardRow(
            external_candidate_id=cand.external_candidate_id,
            name=cand.name,
            email=cand.email,
            cgpa=cand.cgpa,
            resume_ai_score=ai.resume_ai_score if ai else None,
            github_technical_score=gh.github_technical_score if gh else None,
            overall_score=ranking.overall_score if ranking else None,
        )

        if has_final:
            tr = db.get(TestResult, cand.id)
            fr = db.get(FinalRanking, cand.id)
            row.test_la = tr.test_la if tr else None
            row.test_code = tr.test_code if tr else None
            row.test_performance_score = fr.test_performance_score if fr else None
            row.final_score = fr.final_score if fr else None
            row.is_qualified = fr is not None and fr.test_performance_score is not None
            # Qualified definition is test performance threshold used at upload; for demo we treat existence as qualified.

        rows.append(row)

    return LeaderboardResponse(run_id=run.id, rows=rows)


@app.get("/api/pipeline/{run_id}")
def get_pipeline_status(run_id: str, db: Session = Depends(get_db)) -> dict:
    run = _ensure_run_exists(db, run_id)

    candidates_count = db.query(Candidate).filter(Candidate.run_id == run_id).count()
    shortlisted_count = (
        db.query(Ranking)
        .join(Candidate, Candidate.id == Ranking.candidate_id)
        .filter(Candidate.run_id == run_id)
        .filter(Ranking.overall_score >= (run.shortlist_threshold or 70.0))
        .count()
    )
    test_links_count = db.query(TestLink).join(Candidate, Candidate.id == TestLink.candidate_id).filter(Candidate.run_id == run_id).count()

    qualified_count = (
        db.query(FinalRanking)
        .join(Candidate, Candidate.id == FinalRanking.candidate_id)
        .filter(Candidate.run_id == run_id)
        .count()
    )

    return {
        "run_id": run.id,
        "status": run.status,
        "error": run.error,
        "job_description_set": bool(run.job_description),
        "candidates_count": candidates_count,
        "shortlisted_count": shortlisted_count,
        "qualified_count": qualified_count,
        "test_links_count": test_links_count,
        "shortlist_threshold": run.shortlist_threshold,
    }


@app.get("/api/pipeline/{run_id}/test-links")
def get_test_links(run_id: str, db: Session = Depends(get_db)) -> list[dict]:
    # Expose tokens for demo UI. In production, you'd typically not expose raw tokens.
    links = (
        db.query(TestLink, Candidate.email, Candidate.name)
        .join(Candidate, Candidate.id == TestLink.candidate_id)
        .filter(Candidate.run_id == run_id)
        .all()
    )
    out = []
    for tl, email, name in links:
        out.append(
            {
                "candidate_id": tl.candidate_id,
                "email": email,
                "name": name,
                "token": tl.token,
                "test_link_url": tl.test_link_url,
                "created_at": str(tl.created_at) if tl.created_at else None,
            }
        )
    return out


@app.get("/tests/take")
def take_test_page(token: str, db: Session = Depends(get_db)) -> HTMLResponse:
    tl = db.query(TestLink).filter(TestLink.token == token).first()
    if not tl:
        return HTMLResponse(
            "<h3>Invalid or expired test link token.</h3>", status_code=400
        )

    # Build canonical URL shown inside the page.
    test_link_url = tl.test_link_url or (settings.TEST_LINK_BASE_URL + tl.token)

    html = (
        # Inline minimal HTML; keep it in sync with pipeline_service fallback.
        f"""<!doctype html>
<html lang="en">
  <head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Technical Test</title>
    <style>body{{font-family:Arial,sans-serif;margin:24px;max-width:720px}}
      label{{display:block;margin-top:12px;font-weight:600}}
      input{{width:100%;padding:10px}} button{{margin-top:14px;padding:10px 14px;cursor:pointer}}
      .muted{{color:#666;font-size:13px}} .box{{border:1px solid #ddd;padding:16px;border-radius:8px}}
    </style></head>
  <body>
    <h2>Technical Test</h2>
    <div class="muted">Demo page. Submit your scores below to simulate completion.</div>
    <div class="box">
      <div class="muted">Token: <code>{token}</code></div>
      <form id="f">
        <label>Logical Aptitude Score (0-100)</label>
        <input name="test_la" type="number" min="0" max="100" step="1" value="50"/>
        <label>Coding Test Score (0-100)</label>
        <input name="test_code" type="number" min="0" max="100" step="1" value="50"/>
        <button type="submit">Submit Results</button>
      </form>
    </div>
    <pre id="out"></pre>
    <script>
      const f=document.getElementById('f'); const out=document.getElementById('out');
      f.addEventListener('submit', async (e)=>{{
        e.preventDefault();
        const body={{ test_la:Number(f.test_la.value), test_code:Number(f.test_code.value) }};
        out.textContent='Submitting...';
        const res=await fetch('/api/tests/submit?token=' + encodeURIComponent('{token}'), {{
          method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify(body)
        }});
        const data=await res.json().catch(() => ({{}}));
        if(!res.ok) out.textContent='Error: ' + JSON.stringify(data);
        else out.textContent=JSON.stringify(data,null,2);
      }});
    </script>
  </body>
</html>"""
    )
    return HTMLResponse(html)


class SubmitTestIn(BaseModel):
    test_la: int
    test_code: int


@app.post("/api/tests/submit")
def submit_test_results(token: str, body: SubmitTestIn, db: Session = Depends(get_db)) -> dict:
    tl = db.query(TestLink).filter(TestLink.token == token).first()
    if not tl:
        raise HTTPException(status_code=400, detail="Invalid test token.")

    cand_id = tl.candidate_id
    # Candidate submission updates TestResult; recruiter CSV upload will also work.
    tr = db.get(TestResult, cand_id)
    if not tr:
        tr = TestResult(candidate_id=cand_id)
        db.add(tr)
    tr.test_la = int(body.test_la)
    tr.test_code = int(body.test_code)
    tr.uploaded_at = datetime.utcnow()
    db.commit()

    # Recompute final ranking if pipeline exists.
    # Find run_id via candidate relation.
    cand = db.get(Candidate, cand_id)
    if cand and cand.run_id:
        recompute_final_scores_for_candidate(db, cand.run_id, cand_id)

    return {"ok": True}


@app.post("/api/pipeline/{run_id}/send-test-emails")
def send_test_emails(
    run_id: str,
    email_subject: str = "Your Technical Test Link",
    prefer_channel: str = "auto",
    db: Session = Depends(get_db),
) -> dict:
    # Decide shortlist threshold from PipelineRun unless UI passes it explicitly (demo keeps it simple).
    run = _ensure_run_exists(db, run_id)
    shortlist_threshold = float(run.shortlist_threshold) if run.shortlist_threshold is not None else None
    template = (
        "Hi {name},\n\n"
        "Thanks for your interest. Here is your link to complete the technical test.\n\n"
        "Test link: {test_link}\n\n"
        "Best regards,\nRecruitment Team"
    )

    channel = None
    if prefer_channel != "auto":
        channel = prefer_channel

    try:
        sent = run_phase7_send_test_emails(
            db,
            run_id,
            shortlist_threshold=shortlist_threshold,
            email_subject=email_subject,
            email_body_template=template,
            prefer_channel=channel,
        )
        return {"ok": True, "sent": int(sent)}
    except Exception as e:
        # Convert unexpected SMTP/SendGrid misconfiguration into a clean JSON error.
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/pipeline/{run_id}/interview-events")
def get_interview_events(run_id: str, db: Session = Depends(get_db)) -> list[dict]:
    _ensure_run_exists(db, run_id)
    events = (
        db.query(InterviewEvent, Candidate.name, Candidate.email)
        .join(Candidate, Candidate.id == InterviewEvent.candidate_id)
        .filter(Candidate.run_id == run_id)
        .all()
    )
    return [
        {
            "candidate_id": ev.candidate_id,
            "name": name,
            "email": email,
            "scheduled_start": str(ev.scheduled_start) if ev.scheduled_start else None,
            "scheduled_end": str(ev.scheduled_end) if ev.scheduled_end else None,
            "meet_link": ev.meet_link,
            "calendar_event_id": ev.calendar_event_id,
        }
        for ev, name, email in events
    ]


if __name__ == "__main__":
    import uvicorn
    import sys
    import os
    # Ensure current directory is in sys.path for local imports
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
