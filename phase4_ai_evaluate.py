import argparse
import json
import os
import re
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


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


def extract_keywords(jd: str, max_keywords: int = 25) -> list[str]:
    tokens = re.findall(r"[a-zA-Z][a-zA-Z0-9_\\-]{2,}", jd.lower())
    freq = {}
    for t in tokens:
        if t in STOP_WORDS:
            continue
        freq[t] = freq.get(t, 0) + 1
    # Sort by frequency then keep top N.
    keywords = sorted(freq.keys(), key=lambda k: (-freq[k], k))[:max_keywords]
    # Prefer more distinctive words early.
    return keywords


def keyword_fallback_score(resume_text: str, jd: str) -> dict:
    resume_lower = resume_text.lower()
    keywords = extract_keywords(jd)
    if not keywords:
        return {"score": 50, "explanation": "Could not extract job keywords; returning neutral score."}

    matched = [kw for kw in keywords if kw in resume_lower]
    coverage = len(matched) / len(keywords)
    score = int(round(100 * coverage))
    score = max(0, min(100, score))

    matched_preview = ", ".join(matched[:10]) if matched else "None"
    return {
        "score": score,
        "explanation": (
            f"Keyword overlap approach (LLM key not configured). "
            f"JD keywords: {len(keywords)}, matched in resume: {len(matched)}. "
            f"Matched keywords: {matched_preview}."
        ),
    }


def openai_score(resume_text: str, jd: str) -> dict:
    """
    Optional real LLM scoring.
    Requires OPENAI_API_KEY and (optionally) OPENAI_MODEL.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set.")

    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    url = "https://api.openai.com/v1/chat/completions"

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

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    # Best-effort JSON extraction.
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON. Content was: {content[:200]}")
    return json.loads(m.group(0))


def openrouter_score(resume_text: str, jd: str) -> dict:
    """
    OpenRouter OpenAI-compatible scoring.
    Requires OPENROUTER_API_KEY.
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set.")

    model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
    url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1/chat/completions")

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

    resp = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    content = data["choices"][0]["message"]["content"]
    m = re.search(r"\{.*\}", content, flags=re.S)
    if not m:
        raise ValueError(f"LLM did not return JSON. Content was: {content[:200]}")
    return json.loads(m.group(0))


def evaluate(resume_text: str, jd: str) -> dict:
    # If keys are present, prefer real LLM; otherwise fallback.
    if os.getenv("OPENROUTER_API_KEY"):
        return openrouter_score(resume_text, jd)
    if os.getenv("OPENAI_API_KEY"):
        return openai_score(resume_text, jd)
    return keyword_fallback_score(resume_text, jd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4: AI relevance scoring against a Job Description.")
    parser.add_argument(
        "--candidates_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\candidates.csv",
    )
    parser.add_argument(
        "--resume_text_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\resume_text.csv",
    )
    parser.add_argument(
        "--job_description",
        type=str,
        default="",
        help="Job description text. If empty, use --job_description_file.",
    )
    parser.add_argument(
        "--job_description_file",
        type=str,
        default="",
        help="Path to a text file containing the job description.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\ai_scores.csv",
    )
    args = parser.parse_args()

    jd = args.job_description.strip()
    if not jd:
        if not args.job_description_file:
            # Minimal default so the pipeline remains runnable.
            jd = (
                "AI Engineer role: develop machine learning models, experience with deep learning, "
                "NLP/Computer Vision, PyTorch/TensorFlow, building end-to-end ML pipelines, "
                "and publishing AI projects."
            )
        else:
            jd = Path(args.job_description_file).read_text(encoding="utf-8").strip()

    candidates = pd.read_csv(args.candidates_csv)
    resume_text = pd.read_csv(args.resume_text_csv)
    join_col = "Candidate ID" if "Candidate ID" in candidates.columns and "Candidate ID" in resume_text.columns else "Email"
    merged = candidates.merge(resume_text, on=join_col, how="inner")

    out_rows = []
    for _, row in tqdm(merged.iterrows(), total=len(merged), desc="AI scoring"):
        if "Email" in row:
            email = row["Email"]
        elif "Email_x" in row:
            email = row["Email_x"]
        elif "Email_y" in row:
            email = row["Email_y"]
        else:
            email = ""
        candidate_id = row["Candidate ID"] if "Candidate ID" in row else ""
        resume = str(row["resume_text"])
        result = evaluate(resume_text=resume, jd=jd)
        out_rows.append(
            {
                "Candidate ID": candidate_id,
                "Email": email,
                "resume_ai_score": int(result["score"]),
                "resume_ai_explanation": str(result["explanation"]),
            }
        )

    pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(out_rows)} AI scores to {args.out_csv}")


if __name__ == "__main__":
    main()

