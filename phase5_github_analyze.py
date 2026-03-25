import argparse
import base64
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


GITHUB_API = "https://api.github.com"


def github_username_from_url(url: str) -> str:
    """
    Accepts urls like:
      - https://github.com/<username>
      - https://github.com/<username>/
    """
    if not isinstance(url, str) or not url.strip():
        return ""
    m = re.search(r"github\.com/([^/]+)", url.strip())
    if not m:
        return ""
    return m.group(1)


def github_get(session: requests.Session, url: str) -> dict | list:
    resp = session.get(url, timeout=60)
    # If rate-limited, surface a clear error.
    if resp.status_code == 403 and "rate limit" in resp.text.lower():
        raise RuntimeError(f"GitHub API rate limited: {resp.text[:200]}")
    resp.raise_for_status()
    return resp.json()


@dataclass
class RepoMetrics:
    name: str
    full_name: str
    stars: int
    forks: int
    pushed_at: datetime | None
    languages: dict | None
    readme_present: bool
    readme_length_chars: int
    readme_quality_hits: int
    commit_count_sample: int


def parse_github_datetime(s: str | None) -> datetime | None:
    if not s:
        return None
    # GitHub returns ISO 8601 with 'Z'
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def score_repo(repo: RepoMetrics) -> dict:
    # Stars: log-scaled to avoid domination by huge-star repos.
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
    else:
        readme_score = 0

    # Commit frequency/activity: use sampled commit count (limited by API call).
    commit_score = min(30, repo.commit_count_sample)  # 0..30

    # Recency bonus (last push within ~90 days).
    recency_bonus = 0
    if repo.pushed_at:
        age_days = (datetime.now(timezone.utc) - repo.pushed_at).days
        if age_days <= 30:
            recency_bonus = 10
        elif age_days <= 90:
            recency_bonus = 6
        elif age_days <= 180:
            recency_bonus = 3
    commit_score = min(40, commit_score + recency_bonus)  # keep bounded

    # Languages: reward projects with multiple meaningful languages; keep small impact.
    language_score = 0
    if repo.languages:
        n_langs = len([k for k, v in repo.languages.items() if v and v > 50])
        language_score = min(20, n_langs * 5)

    # Total per-repo score (0..100-ish, but will average across repos).
    total = stars_score * 0.3 + readme_score * 0.3 + commit_score * 0.3 + language_score * 0.1
    return {
        "stars_score": stars_score,
        "readme_score": readme_score,
        "commit_score": commit_score,
        "language_score": language_score,
        "total": int(round(total)),
    }


def analyze_repo(session: requests.Session, owner: str, repo_name: str) -> RepoMetrics:
    repo_url = f"{GITHUB_API}/repos/{owner}/{repo_name}"
    rj = github_get(session, repo_url)

    pushed_at = parse_github_datetime(rj.get("pushed_at"))

    # Languages
    languages = None
    try:
        langs = github_get(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/languages")
        languages = langs if isinstance(langs, dict) else None
    except Exception:
        languages = None

    # README
    readme_present = False
    readme_length_chars = 0
    readme_quality_hits = 0
    try:
        readme = github_get(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/readme")
        content_b64 = readme.get("content", "")
        if content_b64:
            decoded = base64.b64decode(content_b64).decode("utf-8", errors="ignore")
            readme_present = True
            readme_length_chars = len(decoded)
            q_keywords = ["installation", "usage", "quickstart", "example", "license", "contributing", "setup"]
            low = decoded.lower()
            readme_quality_hits = sum(1 for kw in q_keywords if kw in low)
    except Exception:
        pass

    # Commit frequency: sample commits (API returns most recent commits).
    commit_count_sample = 0
    try:
        commits = github_get(session, f"{GITHUB_API}/repos/{owner}/{repo_name}/commits?per_page=30")
        if isinstance(commits, list):
            commit_count_sample = len(commits)
    except Exception:
        commit_count_sample = 0

    return RepoMetrics(
        name=repo_name,
        full_name=rj.get("full_name", f"{owner}/{repo_name}"),
        stars=int(rj.get("stargazers_count", 0) or 0),
        forks=int(rj.get("forks_count", 0) or 0),
        pushed_at=pushed_at,
        languages=languages,
        readme_present=readme_present,
        readme_length_chars=readme_length_chars,
        readme_quality_hits=readme_quality_hits,
        commit_count_sample=commit_count_sample,
    )


def analyze_user(session: requests.Session, username: str) -> tuple[int, str]:
    # Fetch repos
    repos = github_get(session, f"{GITHUB_API}/users/{username}/repos?per_page=100&sort=updated")
    if not isinstance(repos, list) or len(repos) == 0:
        return 0, "No public repositories found."

    # Keep top repos by stars to bound API calls.
    repos_sorted = sorted(repos, key=lambda r: int(r.get("stargazers_count", 0) or 0), reverse=True)
    repos_sorted = repos_sorted[:5]  # limit for performance

    repo_scores = []
    for repo in repos_sorted:
        repo_name = repo.get("name")
        if not repo_name:
            continue
        metrics = analyze_repo(session, username, repo_name)
        s = score_repo(metrics)
        repo_scores.append((metrics, s))

    if not repo_scores:
        return 0, "Could not analyze repos."

    total = int(round(sum(s["total"] for _, s in repo_scores) / len(repo_scores)))

    # Build explainable summary
    top = []
    for m, s in sorted(repo_scores, key=lambda t: t[1]["total"], reverse=True)[:3]:
        top_langs = ""
        if m.languages:
            top_langs = ", ".join(list(m.languages.keys())[:3])
        top.append(
            f"- {m.full_name}: score={s['total']}, stars={m.stars}, commits_sample={m.commit_count_sample}, "
            f"readme={'yes' if m.readme_present else 'no'}, langs={top_langs or 'n/a'}"
        )
    summary = "Heuristic repo-level analysis (no LLM key):\n" + "\n".join(top)
    return max(0, min(100, total)), summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5: analyze GitHub profiles (repo-level).")
    parser.add_argument(
        "--candidates_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\candidates.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\github_analysis.csv",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.candidates_csv)
    if "Candidate ID" not in df.columns or "GitHub Profile" not in df.columns:
        raise ValueError("candidates.csv must contain 'Candidate ID' and 'GitHub Profile'.")

    token = os.getenv("GITHUB_TOKEN", "").strip()
    session = requests.Session()
    session.headers.update(
        {
            "Accept": "application/vnd.github+json",
        }
    )
    if token:
        session.headers.update({"Authorization": f"Bearer {token}"})

    out_rows = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Analyzing GitHub profiles"):
        candidate_id = row["Candidate ID"]
        email = row["Email"] if "Email" in df.columns else ""
        gh_url = row["GitHub Profile"]
        username = github_username_from_url(gh_url)

        if not username:
            out_rows.append(
                {
                    "Candidate ID": candidate_id,
                    "Email": email,
                    "github_technical_score": 0,
                    "github_summary": "No valid GitHub profile URL provided.",
                }
            )
            continue

        try:
            score, summary = analyze_user(session, username=username)
            out_rows.append(
                {
                    "Candidate ID": candidate_id,
                    "Email": email,
                    "github_technical_score": int(score),
                    "github_summary": summary,
                }
            )
        except Exception as e:
            out_rows.append(
                {
                    "Candidate ID": candidate_id,
                    "Email": email,
                    "github_technical_score": 0,
                    "github_summary": f"GitHub analysis failed: {str(e)[:500]}",
                }
            )

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"Wrote GitHub analysis for {len(out_rows)} candidates to {args.out_csv}")


if __name__ == "__main__":
    main()

