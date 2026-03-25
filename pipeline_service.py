import base64
import math
import os
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
from sqlalchemy.orm import Session

from .config import get_settings
from .models import (
    AiEvaluation,
    Candidate,
    EmailSendLog,
    FinalRanking,
    GithubAnalysis,
    InterviewEvent,
    PipelineRun,
    Ranking,
    ResumeExtraction,
    TestLink,
    TestResult,
)


settings = get_settings()


STOP_WORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "to",
    "of",
    "in",
    "for",
    "with",
    "on",
    "at",
    "by",
    "is",
    "are",
    "be",
    "as",
    "from",
    "that",
    "this",
    "it",
    "will",
    "you",
    "your",
    "we",
    "they",
    "their",
    "our",
    "can",
    "may",
    "include",
    "includes",
    "included",
    "using",
    "use",
    "used",
    "develop",
    "development",
    "experience",
    "required",
    "responsibilities",
}


def google_drive_download_url(url: str) -> str:
    """
    Convert common Google Drive "file/d/<id>/view" links into direct download links.
    """
    if not isinstance(url, str):
        return ""
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        return url
    file_id = m.group(1)
    return f"https://drive.google.com/uc?export=download&id={file_id}"


def download_file(url: str, dest: Path, timeout_s: int = 120) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout_s) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 128):
                if chunk:
                    f.write(chunk)


def extract_text_from_pdf(pdf_path: Path) -> str:
    import fitz  # PyMuPDF

    doc = fitz.open(str(pdf_path))
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    return text[:200_000]  # bound prompt size later


def extract_keywords(jd: str, max_keywords: int = 25) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\\-]{2,}", jd.lower())
    freq: dict[str, int] = {}
    for t in tokens:
        if t in STOP_WORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    keywords = sorted(freq.keys(), key=lambda k: (-freq[k], k))[:max_keywords]
    return keywords


def keyword_fallback_score(resume_text: str, jd: str) -> tuple[int, str]:
    resume_lower = resume_text.lower()
    keywords = extract_keywords(jd)
    if not keywords:
        return 50, "Could not extract job keywords; returning neutral score."

    matched = [kw for kw in keywords if kw in resume_lower]
    coverage = len(matched) / len(keywords)
    score = int(round(100 * coverage))
    score = max(0, min(100, score))

    matched_preview = ", ".join(matched[:10]) if matched else "None"
    explanation = (
        "Keyword overlap approach (LLM key not configured). "
        f"JD keywords: {len(keywords)}, matched in resume: {len(matched)}. "
        f"Matched keywords: {matched_preview}."
    )
    return score, explanation


def llm_score_openai(resume_text: str, jd: str) -> tuple[int, str]:
    """
    Optional real LLM scoring. Requires OPENAI_API_KEY.
    """
    from openai import OpenAI

    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    system = (
        "You are an expert resume evaluator for AI engineering roles. "
        "Return ONLY valid JSON with keys: score (0-100 integer) and explanation (string)."
    )
    user = (
        f"Job Description:\n{jd}\n\n"
        f"Candidate Resume Text:\n{resume_text}\n\n"
        "Score how relevant the candidate resume is to the job description from 0 to 100. "
        "Base your reasoning on skills, projects, research, and any matching technical terms. "
        "Return strict JSON only."
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON. Content was: {content[:200]}")
    import json

    data = json.loads(m.group(0))
    return int(data["score"]), str(data["explanation"])


def llm_score_openrouter(resume_text: str, jd: str) -> tuple[int, str]:
    """
    OpenRouter OpenAI-compatible scoring.
    Requires OPENROUTER_API_KEY.
    """
    from openai import OpenAI
    import json

    if not settings.OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    client = OpenAI(api_key=settings.OPENROUTER_API_KEY, base_url=settings.OPENROUTER_BASE_URL)
    model = settings.OPENROUTER_MODEL

    system = (
        "You are an expert resume evaluator for AI engineering roles. "
        "Return ONLY valid JSON with keys: score (0-100 integer) and explanation (string)."
    )
    user = (
        f"Job Description:\n{jd}\n\n"
        f"Candidate Resume Text:\n{resume_text}\n\n"
        "Score how relevant the candidate resume is to the job description from 0 to 100. "
        "Base your reasoning on skills, projects, research, and any matching technical terms. "
        "Return strict JSON only."
    )

    resp = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    content = resp.choices[0].message.content or ""
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON. Content was: {content[:200]}")
    data = json.loads(m.group(0))
    return int(data["score"]), str(data["explanation"])


def ai_score(resume_text: str, jd: str) -> tuple[int, str]:
    if settings.OPENROUTER_API_KEY:
        return llm_score_openrouter(resume_text=resume_text, jd=jd)
    if settings.OPENAI_API_KEY:
        return llm_score_openai(resume_text=resume_text, jd=jd)
    return keyword_fallback_score(resume_text=resume_text, jd=jd)


@dataclass
class RepoMetrics:
    full_name: str
    stars: int
    pushed_at: datetime | None
    languages: dict | None
    readme_present: bool
    readme_length_chars: int
    readme_quality_hits: int
    commit_count_sample: int


def github_username_from_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        return ""
    m = re.search(r"github\.com/([^/]+)", url.strip())
    if not m:
        return ""
    return m.group(1)


def github_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def github_score_repo(repo: RepoMetrics) -> int:
    # Stars: log-scaled.
    stars_score = min(50, int(round(math.log10(1 + repo.stars) * 25)))

    # README: presence + size + a few quality keywords.
    readme_score = 0
    if repo.readme_present:
        if repo.readme_length_chars >= 1200:
            readme_score = 40
        elif repo.readme_length_chars >= 600:
            readme_score = 30
        elif repo.readme_length_chars >= 200:
            readme_score = 20
        else:
            readme_score = 10

        readme_score += min(10, repo.readme_quality_hits * 2)

    commit_score = min(30, repo.commit_count_sample)

    recency_bonus = 0
    if repo.pushed_at:
        age_days = (datetime.now(timezone.utc) - repo.pushed_at).days
        if age_days <= 30:
            recency_bonus = 10
        elif age_days <= 90:
            recency_bonus = 6
        elif age_days <= 180:
            recency_bonus = 3

    commit_score = min(40, commit_score + recency_bonus)

    language_score = 0
    if repo.languages:
        n_langs = len([k for k, v in repo.languages.items() if v and v > 50])
        language_score = min(20, n_langs * 5)

    total = stars_score * 0.3 + readme_score * 0.3 + commit_score * 0.3 + language_score * 0.1
    return int(round(total))


def github_analyze_user(session: requests.Session, username: str) -> tuple[int, str]:
    api = "https://api.github.com"
    repos = session.get(f"{api}/users/{username}/repos?per_page=100&sort=updated", timeout=60)
    if repos.status_code == 403 and "rate limit" in repos.text.lower():
        raise RuntimeError("GitHub API rate limited.")
    repos.raise_for_status()
    repos_json = repos.json()
    if not isinstance(repos_json, list) or len(repos_json) == 0:
        return 0, "No public repositories found."

    repos_sorted = sorted(
        repos_json, key=lambda r: int(r.get("stargazers_count", 0) or 0), reverse=True
    )[:5]

    repo_summaries: list[str] = []
    scores: list[int] = []
    for repo in repos_sorted:
        repo_name = repo.get("name")
        if not repo_name:
            continue

        rj = session.get(f"{api}/repos/{username}/{repo_name}", timeout=60).json()
        pushed_at = github_datetime(rj.get("pushed_at"))
        languages = None
        try:
            langs_resp = session.get(f"{api}/repos/{username}/{repo_name}/languages", timeout=60)
            if langs_resp.ok:
                languages = langs_resp.json()
        except Exception:
            languages = None

        readme_present = False
        readme_length_chars = 0
        readme_quality_hits = 0
        try:
            readme_resp = session.get(f"{api}/repos/{username}/{repo_name}/readme", timeout=60)
            if readme_resp.ok:
                content_b64 = readme_resp.json().get("content", "")
                decoded = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
                readme_present = True
                readme_length_chars = len(decoded)
                q_keywords = [
                    "installation",
                    "usage",
                    "quickstart",
                    "example",
                    "license",
                    "contributing",
                    "setup",
                ]
                low = decoded.lower()
                readme_quality_hits = sum(1 for kw in q_keywords if kw in low)
        except Exception:
            pass

        commit_count_sample = 0
        try:
            commits = session.get(
                f"{api}/repos/{username}/{repo_name}/commits?per_page=30", timeout=60
            )
            if commits.ok:
                commits_json = commits.json()
                if isinstance(commits_json, list):
                    commit_count_sample = len(commits_json)
        except Exception:
            commit_count_sample = 0

        metrics = RepoMetrics(
            full_name=rj.get("full_name", f"{username}/{repo_name}"),
            stars=int(rj.get("stargazers_count", 0) or 0),
            pushed_at=pushed_at,
            languages=languages,
            readme_present=readme_present,
            readme_length_chars=readme_length_chars,
            readme_quality_hits=readme_quality_hits,
            commit_count_sample=commit_count_sample,
        )

        s = github_score_repo(metrics)
        scores.append(s)
        top_langs = ""
        if metrics.languages:
            top_langs = ", ".join(list(metrics.languages.keys())[:3])
        repo_summaries.append(
            f"- {metrics.full_name}: score={s}, stars={metrics.stars}, commits_sample={metrics.commit_count_sample}, "
            f"readme={'yes' if metrics.readme_present else 'no'}, langs={top_langs or 'n/a'}"
        )

    final_score = int(round(sum(scores) / max(1, len(scores))))
    summary = "Heuristic repo-level analysis (no LLM key):\n" + "\n".join(repo_summaries[:3])
    return max(0, min(100, final_score)), summary


def normalize_cgpa(cgpa: float | None, max_cgpa: float = 10.0) -> float:
    if cgpa is None:
        return 0.0
    try:
        v = float(cgpa)
    except Exception:
        v = 0.0
    return float(max(0.0, min(1.0, v / max_cgpa)))


def upsert_candidates_from_df(db: Session, run_id: str, df: pd.DataFrame) -> None:
    # Normalize columns to make headers case-insensitive and handle aliases
    col_map = {str(c).strip().lower(): c for c in df.columns}
    rename_mapping = {}
    
    aliases = {
        "Name": ["name", "candidate name", "first name", "full name"],
        "College": ["college", "university", "institute", "institution"],
        "Branch": ["branch", "department", "degree", "course"],
        "CGPA": ["cgpa", "gpa", "score"],
        "Best AI Project": ["best ai project", "best_ai_pr", "best_ai_project", "project"],
        "Research Work": ["research work", "research_w", "research_work", "research"],
        "GitHub Profile": ["github profile", "github", "github_link", "github link"],
        "Resume Link": ["resume link", "resume", "resume url", "resume_url"],
        "Candidate ID": ["candidate id", "candidate_id", "id"],
        "s_no": ["s_no", "sno", "s.no", "serial no", "serial number"],
        "Email": ["email", "email id", "email address"],
    }
    
    for final_col, possible_names in aliases.items():
        for pname in possible_names:
            if pname in col_map:
                rename_mapping[col_map[pname]] = final_col
                break

    df = df.rename(columns=rename_mapping)

    # Accept either "Candidate ID" or "s_no" as external ID.
    external_id_col = "Candidate ID" if "Candidate ID" in df.columns else None
    if external_id_col is None and "s_no" in df.columns:
        external_id_col = "s_no"

    required_cols = [
        "Name", "College", "Branch", "CGPA", "Best AI Project", 
        "Research Work", "GitHub Profile", "Resume Link"
    ]
    for c in required_cols:
        if c not in df.columns:
            df[c] = None
            
    if "Email" not in df.columns:
        df["Email"] = None

    now = datetime.now(timezone.utc)

    # Upsert by (run_id, external_candidate_id if provided else row index).
    candidates_existing = (
        db.query(Candidate)
        .filter(Candidate.run_id == run_id)
        .all()
    )
    existing_by_key = {}
    for c in candidates_existing:
        existing_by_key[c.external_candidate_id or c.id] = c

    # If we have no external id column, generate a deterministic one from row index.
    if external_id_col is None:
        df = df.copy()
        df["Candidate ID"] = [str(i + 1) for i in range(len(df))]
        external_id_col = "Candidate ID"

    for idx, row in df.iterrows():
        ext_id = str(row.get(external_id_col, "")).strip()
        if not ext_id:
            ext_id = str(idx + 1)
        key = ext_id
        existing = existing_by_key.get(key)
        if existing:
            cand = existing
        else:
            cand = Candidate(run_id=run_id, external_candidate_id=ext_id)
            db.add(cand)

        cand.name = str(row.get("Name") or "").strip() or None
        cand.email = str(row.get("Email") or "").strip() or None
        cand.college = str(row.get("College") or "").strip() or None
        cand.branch = str(row.get("Branch") or "").strip() or None
        cand.cgpa = float(row.get("CGPA")) if pd.notna(row.get("CGPA")) else None
        cand.best_ai_project = str(row.get("Best AI Project") or "") or None
        cand.research_work = str(row.get("Research Work") or "") or None
        cand.github_profile_url = str(row.get("GitHub Profile") or "") or None
        cand.resume_link_url = str(row.get("Resume Link") or "") or None

    db.commit()


def run_phase3_process_resumes(db: Session, run_id: str) -> int:
    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()
    if not candidates:
        return 0

    run_dir = Path(f"./data/{run_id}")
    resumes_dir = run_dir / "resumes"
    resumes_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    for cand in candidates:
        extraction = db.get(ResumeExtraction, cand.id)
        if extraction and extraction.resume_text:
            continue
        if not cand.resume_link_url:
            continue
        pdf_path = resumes_dir / f"{cand.id}.pdf"
        download_url = google_drive_download_url(cand.resume_link_url)
        download_file(download_url, pdf_path)
        text = extract_text_from_pdf(pdf_path)

        if not extraction:
            extraction = ResumeExtraction(candidate_id=cand.id)
            db.add(extraction)
        extraction.resume_text = text
        extraction.extracted_at = datetime.now(timezone.utc)
        extracted += 1
        db.commit()

    return extracted


def run_phase4_evaluate(db: Session, run_id: str) -> int:
    run = db.get(PipelineRun, run_id)
    if not run or not run.job_description:
        raise ValueError("Job description missing for run.")

    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()
    job_desc = run.job_description

    done = 0
    for cand in candidates:
        extraction = db.get(ResumeExtraction, cand.id)
        if not extraction or not extraction.resume_text:
            continue
        existing = db.get(AiEvaluation, cand.id)
        if existing and existing.resume_ai_score is not None:
            continue

        score, explanation = ai_score(resume_text=extraction.resume_text, jd=job_desc)

        if not existing:
            existing = AiEvaluation(candidate_id=cand.id)
            db.add(existing)
        existing.resume_ai_score = int(score)
        existing.resume_ai_explanation = explanation
        existing.evaluated_at = datetime.now(timezone.utc)
        done += 1
        db.commit()

    return done


def run_phase5_github(db: Session, run_id: str) -> int:
    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()

    token = settings.GITHUB_TOKEN.strip()
    session = requests.Session()
    session.headers.update({"Accept": "application/vnd.github+json"})
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    done = 0
    for cand in candidates:
        existing = db.get(GithubAnalysis, cand.id)
        if existing and existing.github_technical_score is not None:
            continue
        if not cand.github_profile_url:
            continue
        username = github_username_from_url(cand.github_profile_url)
        if not username:
            continue
        try:
            score, summary = github_analyze_user(session=session, username=username)
        except Exception as e:
            # If GitHub is rate-limited or a candidate URL is invalid, continue the pipeline.
            score, summary = 0, f"GitHub analysis skipped/failed: {str(e)[:300]}"

        if not existing:
            existing = GithubAnalysis(candidate_id=cand.id)
            db.add(existing)
        existing.github_technical_score = int(score)
        existing.github_summary = summary
        existing.analyzed_at = datetime.now(timezone.utc)
        done += 1
        db.commit()

    return done


def run_phase6_rank(db: Session, run_id: str) -> tuple[int, int]:
    run = db.get(PipelineRun, run_id)
    if not run:
        raise ValueError("Run not found.")
    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()

    now = datetime.now(timezone.utc)
    for cand in candidates:
        ai = db.get(AiEvaluation, cand.id)
        gh = db.get(GithubAnalysis, cand.id)
        if not ai or ai.resume_ai_score is None:
            continue
        resume_ai_score = float(ai.resume_ai_score or 0)
        github_score = float(gh.github_technical_score) if gh and gh.github_technical_score is not None else 0.0
        cgpa_score = normalize_cgpa(cand.cgpa) * 100.0

        overall = (
            run.w_resume * resume_ai_score + run.w_github * github_score + run.w_cgpa * cgpa_score
        )
        ranking = db.get(Ranking, cand.id)
        if not ranking:
            ranking = Ranking(candidate_id=cand.id)
            db.add(ranking)
        ranking.overall_score = float(overall)
        ranking.created_at = now

    db.commit()

    # Shortlist + create test links
    threshold = run.shortlist_threshold if run.shortlist_threshold is not None else 70.0
    qualified = 0
    total = 0
    for cand in candidates:
        ranking = db.get(Ranking, cand.id)
        if not ranking or ranking.overall_score is None:
            continue
        total += 1
        if ranking.overall_score >= threshold:
            qualified += 1
            tl = db.get(TestLink, cand.id)
            if not tl:
                tl = TestLink(
                    candidate_id=cand.id,
                    token=secrets.token_urlsafe(24),
                    test_link_url=None,
                    created_at=now,
                )
                db.add(tl)
                db.commit()

    return total, qualified


def run_phase7_generate_test_links(db: Session, run_id: str, test_link_base: str) -> int:
    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()
    updated = 0
    now = datetime.now(timezone.utc)
    for cand in candidates:
        ranking = db.get(Ranking, cand.id)
        if not ranking or ranking.overall_score is None:
            continue
        threshold = db.get(PipelineRun, run_id).shortlist_threshold or 70.0
        if ranking.overall_score < threshold:
            continue
        tl = db.get(TestLink, cand.id)
        if not tl:
            tl = TestLink(
                candidate_id=cand.id,
                token=secrets.token_urlsafe(24),
                test_link_url=None,
                created_at=now,
            )
            db.add(tl)
            db.commit()
        # Set URL if not set.
        if not tl.test_link_url:
            tl.test_link_url = test_link_base + tl.token
            tl.created_at = now
            updated += 1
            db.commit()
    return updated


def run_phase8_upload_tests_and_rank_final(
    db: Session, run_id: str, test_df: pd.DataFrame, test_threshold: float
) -> int:
    # Normalize columns to make headers case-insensitive and handle aliases
    col_map = {str(c).strip().lower(): c for c in test_df.columns}
    rename_mapping = {}
    aliases_test = {
        "Candidate ID": ["candidate id", "candidate_id", "id", "s_no", "sno", "s.no"],
        "Email": ["email", "email id", "email address"],
        "test_la": ["test_la", "logical aptitude", "la score"],
        "test_code": ["test_code", "coding test", "coding score"],
    }
    for final_col, possible_names in aliases_test.items():
        for pname in possible_names:
            if pname in col_map:
                rename_mapping[col_map[pname]] = final_col
                break
    test_df = test_df.rename(columns=rename_mapping)

    # test_df should contain either:
    # - Candidate ID, test_la, test_code
    # - or Email, test_la, test_code
    candidates = db.query(Candidate).filter(Candidate.run_id == run_id).all()
    candidates_by_external = {}
    candidates_by_email = {}
    for c in candidates:
        if c.external_candidate_id:
            candidates_by_external[c.external_candidate_id] = c
        if c.email:
            candidates_by_email[c.email] = c

    if "Candidate ID" in test_df.columns:
        join_col = "Candidate ID"
        use_external = True
    else:
        join_col = "Email"
        use_external = False

    if "test_la" not in test_df.columns or "test_code" not in test_df.columns:
        raise ValueError("Test CSV must contain 'test_la' and 'test_code' columns.")
    test_df = test_df.copy()
    test_df["test_la"] = pd.to_numeric(test_df["test_la"], errors="coerce").fillna(0).astype(int)
    test_df["test_code"] = pd.to_numeric(test_df["test_code"], errors="coerce").fillna(0).astype(int)

    qualified = 0
    for _, row in test_df.iterrows():
        key = str(row.get(join_col, "")).strip()
        if not key:
            continue
        cand = candidates_by_external.get(key) if use_external else candidates_by_email.get(key)
        if not cand:
            continue

        test = db.get(TestResult, cand.id)
        if not test:
            test = TestResult(candidate_id=cand.id)
            db.add(test)
        test.test_la = int(row["test_la"])
        test.test_code = int(row["test_code"])
        test.uploaded_at = datetime.now(timezone.utc)

        # Final score
        ranking = db.get(Ranking, cand.id)
        if ranking is None or ranking.overall_score is None:
            continue
        test_performance_score = (
            db.get(PipelineRun, run_id).w_test_la * float(test.test_la)
            + db.get(PipelineRun, run_id).w_test_code * float(test.test_code)
        )
        final_score = (
            db.get(PipelineRun, run_id).w_pipeline * float(ranking.overall_score)
            + db.get(PipelineRun, run_id).w_test * float(test_performance_score)
        )

        fr = db.get(FinalRanking, cand.id)
        if not fr:
            fr = FinalRanking(candidate_id=cand.id)
            db.add(fr)
        fr.test_performance_score = float(test_performance_score)
        fr.final_score = float(final_score)
        fr.computed_at = datetime.now(timezone.utc)

        if test_performance_score >= test_threshold:
            qualified += 1

    db.commit()
    return qualified


def run_phase9_schedule_interviews_dry_run(
    db: Session, run_id: str, start_datetime: datetime, slot_minutes: int, timezone_str: str
) -> int:
    # Dry-run: creates placeholder InterviewEvent records.
    qualified_candidates = (
        db.query(FinalRanking, Candidate)
        .join(Candidate, Candidate.id == FinalRanking.candidate_id)
        .filter(Candidate.run_id == run_id)
        .all()
    )
    if not qualified_candidates:
        return 0

    # Sort by final score descending.
    qualified_candidates_sorted = sorted(
        qualified_candidates, key=lambda t: (t[0].final_score if t[0].final_score is not None else -1), reverse=True
    )

    now = datetime.now(timezone.utc)
    created = 0
    for i, (fr, cand) in enumerate(qualified_candidates_sorted):
        existing = db.get(InterviewEvent, cand.id)
        if existing:
            continue
        event_start = start_datetime + timedelta(minutes=i * slot_minutes)
        event_end = event_start + timedelta(minutes=slot_minutes)

        ev = InterviewEvent(
            candidate_id=cand.id,
            calendar_event_id=None,
            meet_link=None,
            scheduled_start=event_start,
            scheduled_end=event_end,
            created_at=now,
        )
        db.add(ev)
        created += 1
        db.commit()
    return created


def _render_test_submission_html(test_link_url: str) -> str:
    # Minimal HTML to prove the link works. In a real app you would embed your real test page.
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8"/>
    <meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Technical Test</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 24px; max-width: 720px; }}
      label {{ display:block; margin-top: 12px; font-weight: 600; }}
      input {{ width: 100%; padding: 10px; }}
      button {{ margin-top: 14px; padding: 10px 14px; cursor: pointer; }}
      .muted {{ color: #666; font-size: 13px; }}
      .box {{ border: 1px solid #ddd; padding: 16px; border-radius: 8px; }}
    </style>
  </head>
  <body>
    <h2>Technical Test</h2>
    <div class="muted">This is a demo test page. Use the submit form below to simulate test completion.</div>
    <div class="box">
      <div class="muted">If you already have an external test, replace this page with your real test URL.</div>
      <div class="muted">Token-validated test link: <code>{test_link_url}</code></div>
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
      const params = new URLSearchParams(window.location.search);
      const token = params.get('token') || '';
      const f = document.getElementById('f');
      const out = document.getElementById('out');
      f.addEventListener('submit', async (e) => {{
        e.preventDefault();
        const body = {{
          test_la: Number(f.test_la.value),
          test_code: Number(f.test_code.value)
        }};
        out.textContent = 'Submitting...';
        const res = await fetch('/api/tests/submit?token=' + encodeURIComponent(token), {{
          method: 'POST',
          headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify(body)
        }});
        const data = await res.json().catch(() => ({{}}));
        if (!res.ok) out.textContent = 'Error: ' + JSON.stringify(data);
        else out.textContent = JSON.stringify(data, null, 2);
      }});
    </script>
  </body>
</html>
"""


def send_email_smtp(*, to_email: str, subject: str, body: str) -> str:
    import smtplib
    from email.message import EmailMessage

    if not settings.SMTP_HOST or not settings.SMTP_USER or not settings.SMTP_PASS:
        raise RuntimeError("SMTP is not configured (set SMTP_HOST/SMTP_USER/SMTP_PASS).")
    if not settings.SMTP_FROM_EMAIL:
        raise RuntimeError("SMTP_FROM_EMAIL not configured.")

    msg = EmailMessage()
    msg["From"] = settings.SMTP_FROM_EMAIL
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT, timeout=60) as server:
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASS)
        server.send_message(msg)

    # SMTP doesn't always provide message-id; return a best-effort marker.
    return f"smtp-{datetime.now(timezone.utc).isoformat()}"


def send_email_sendgrid(*, to_email: str, subject: str, body: str) -> str:
    if not settings.SENDGRID_API_KEY:
        raise RuntimeError("SENDGRID_API_KEY is not configured.")
    if not settings.SENDGRID_FROM_EMAIL:
        raise RuntimeError("SENDGRID_FROM_EMAIL not configured.")

    url = "https://api.sendgrid.com/v3/mail/send"
    headers = {"Authorization": f"Bearer {settings.SENDGRID_API_KEY}"}
    payload = {
        "personalizations": [{"to": [{"email": to_email}]}],
        "from": {"email": settings.SENDGRID_FROM_EMAIL},
        "subject": subject,
        "content": [{"type": "text/plain", "value": body}],
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    # SendGrid can respond with headers; keep best-effort.
    return resp.headers.get("x-message-id") or f"sendgrid-{uuid.uuid4()}"


def send_test_email_for_candidate(
    *, channel: str, to_email: str, subject: str, body: str
) -> str:
    if channel == "sendgrid":
        return send_email_sendgrid(to_email=to_email, subject=subject, body=body)
    # Default SMTP
    return send_email_smtp(to_email=to_email, subject=subject, body=body)


def run_phase7_send_test_emails(
    db: Session,
    run_id: str,
    *,
    shortlist_threshold: float | None,
    email_subject: str,
    email_body_template: str,
    prefer_channel: str | None = None,
) -> int:
    """
    Sends test emails to candidates who passed Phase 6 overall threshold.
    Uses TestLink token mapping stored in DB.
    """
    run = db.get(PipelineRun, run_id)
    if not run:
        raise ValueError("Run not found.")
    threshold = shortlist_threshold if shortlist_threshold is not None else (run.shortlist_threshold or 70.0)

    # Find candidates with enough overall score + have email.
    qualified = (
        db.query(Candidate.id)
        .join(Ranking, Ranking.candidate_id == Candidate.id)
        .filter(Candidate.run_id == run_id)
        .filter(Ranking.overall_score >= threshold)
        .all()
    )
    ids = [r[0] for r in qualified]
    if not ids:
        return 0

    # Ensure we have TestLink tokens.
    candidates = db.query(Candidate).filter(Candidate.id.in_(ids)).all()
    base_url = settings.TEST_LINK_BASE_URL

    sent = 0
    now = datetime.now(timezone.utc)
    for cand in candidates:
        if not cand.email:
            continue
        tl = db.get(TestLink, cand.id)
        if not tl:
            tl = TestLink(
                candidate_id=cand.id,
                token=secrets.token_urlsafe(24),
                test_link_url=None,
                created_at=now,
            )
            db.add(tl)
            db.commit()
        if not tl.token:
            continue
        test_link = base_url + tl.token

        # Idempotency: only send once per candidate per run.
        already = (
            db.query(EmailSendLog)
            .filter(EmailSendLog.run_id == run_id, EmailSendLog.candidate_id == cand.id)
            .first()
        )
        if already and already.sent_at:
            continue

        body = email_body_template.format(name=cand.name or "", test_link=test_link)
        channel = prefer_channel or ("sendgrid" if settings.SENDGRID_API_KEY else "smtp")
        msg_id = send_test_email_for_candidate(channel=channel, to_email=cand.email, subject=email_subject, body=body)

        log = EmailSendLog(
            candidate_id=cand.id,
            run_id=run_id,
            channel=channel,
            message_id=msg_id,
            sent_at=now,
        )
        db.add(log)
        sent += 1
        db.commit()

    return sent


def run_phase9_schedule_interviews_google_calendar(
    db: Session,
    run_id: str,
    *,
    start_datetime: datetime,
    slot_minutes: int,
    timezone_str: str,
    calendar_id: str,
) -> int:
    if not settings.GOOGLE_CALENDAR_ENABLED:
        # Keep it safe: if enabled flag isn't set, don't try real scheduling.
        return run_phase9_schedule_interviews_dry_run(
            db=db,
            run_id=run_id,
            start_datetime=start_datetime,
            slot_minutes=slot_minutes,
            timezone_str=timezone_str,
        )

    if not settings.GOOGLE_OAUTH_CREDENTIALS_JSON or not Path(settings.GOOGLE_OAUTH_CREDENTIALS_JSON).exists():
        raise RuntimeError("Google OAuth credentials JSON not found. Set GOOGLE_OAUTH_CREDENTIALS_JSON.")

    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

    creds = None
    token_json = Path(settings.GOOGLE_OAUTH_TOKEN_JSON)
    if token_json.exists():
        creds = Credentials.from_authorized_user_file(str(token_json), SCOPES)

    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(str(settings.GOOGLE_OAUTH_CREDENTIALS_JSON), SCOPES)
        creds = flow.run_local_server(port=0)
        token_json.parent.mkdir(parents=True, exist_ok=True)
        token_json.write_text(creds.to_json(), encoding="utf-8")

    service = build("calendar", "v3", credentials=creds)

    qualified_candidates = (
        db.query(FinalRanking, Candidate)
        .join(Candidate, Candidate.id == FinalRanking.candidate_id)
        .filter(Candidate.run_id == run_id)
        .all()
    )
    if not qualified_candidates:
        return 0

    qualified_candidates_sorted = sorted(
        qualified_candidates,
        key=lambda t: (t[0].final_score if t[0].final_score is not None else -1),
        reverse=True,
    )

    created = 0
    now = datetime.now(timezone.utc)
    tz = timezone.utc if not timezone_str else timezone.utc  # keep simple; Google will interpret dateTime with timezone.
    for i, (fr, cand) in enumerate(qualified_candidates_sorted):
        existing = db.get(InterviewEvent, cand.id)
        if existing:
            continue
        event_start = start_datetime + timedelta(minutes=i * slot_minutes)
        event_end = event_start + timedelta(minutes=slot_minutes)

        body = "Interview invitation generated by AI Candidate Screening Platform."
        if not cand.email:
            # Calendar invite requires attendee; still create event without attendees if missing.
            attendees = []
        else:
            attendees = [{"email": cand.email}]

        event = {
            "summary": f"AI Engineering Interview - {cand.name or ''}".strip(),
            "description": body,
            "start": {"dateTime": event_start.isoformat(), "timeZone": timezone_str},
            "end": {"dateTime": event_end.isoformat(), "timeZone": timezone_str},
            "attendees": attendees,
            "conferenceData": {
                "createRequest": {
                    "requestId": str(uuid.uuid4()),
                    "conferenceSolutionKey": {"type": "hangoutsMeet"},
                }
            },
        }

        created_event = (
            service.events()
            .insert(
                calendarId=calendar_id,
                body=event,
                conferenceDataVersion=1,
                sendUpdates="all" if attendees else "none",
            )
            .execute()
        )

        conf = created_event.get("conferenceData") or {}
        meet_link = (conf.get("entryPoints") or [{}])[0].get("uri", "") if conf else ""

        ev = InterviewEvent(
            candidate_id=cand.id,
            calendar_event_id=created_event.get("id"),
            meet_link=meet_link,
            scheduled_start=event_start,
            scheduled_end=event_end,
            created_at=now,
        )
        db.add(ev)
        db.commit()
        created += 1

    return created


def recompute_final_scores_for_candidate(db: Session, run_id: str, candidate_id: str) -> None:
    run = db.get(PipelineRun, run_id)
    if not run:
        return
    ranking = db.get(Ranking, candidate_id)
    test = db.get(TestResult, candidate_id)
    if not ranking or ranking.overall_score is None or not test or test.test_la is None or test.test_code is None:
        return

    test_performance_score = (run.w_test_la * float(test.test_la) + run.w_test_code * float(test.test_code))
    final_score = run.w_pipeline * float(ranking.overall_score) + run.w_test * float(test_performance_score)

    fr = db.get(FinalRanking, candidate_id)
    now = datetime.now(timezone.utc)
    if not fr:
        fr = FinalRanking(candidate_id=candidate_id)
        db.add(fr)
    fr.test_performance_score = float(test_performance_score)
    fr.final_score = float(final_score)
    fr.computed_at = now
    db.commit()


def create_and_init_run(db: Session) -> PipelineRun:
    now = datetime.now(timezone.utc)
    run = PipelineRun(created_at=now, updated_at=now, status="created")
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def update_run_status(db: Session, run_id: str, status: str, error: str | None = None) -> None:
    run = db.get(PipelineRun, run_id)
    if not run:
        return
    run.status = status
    run.updated_at = datetime.now(timezone.utc)
    run.error = error
    db.commit()

