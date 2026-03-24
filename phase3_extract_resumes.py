import argparse
import re
from pathlib import Path

import pandas as pd
import requests
import fitz  # PyMuPDF
from tqdm import tqdm


def google_drive_download_url(url: str) -> str:
    """
    Convert common Google Drive "file/d/<id>/view" links into a direct download URL.
    """
    m = re.search(r"/file/d/([^/]+)/", url)
    if not m:
        # Fallback: some links may already be a direct download URL.
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
    doc = fitz.open(pdf_path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text"))
    text = "\n".join(parts).strip()
    # Keep size bounded for LLM prompt construction later.
    return text[:200_000]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download resumes and extract text (Phase 3).")
    parser.add_argument(
        "--candidates_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\candidates.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    resumes_dir = out_dir / "resumes"
    resume_text_dir = out_dir / "resume_text"
    resume_text_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.candidates_csv)
    if "Resume Link" not in df.columns:
        raise ValueError("candidates.csv must contain 'Resume Link' column.")
    key_col = "Candidate ID" if "Candidate ID" in df.columns else "Email"
    if key_col not in df.columns or "Resume Link" not in df.columns:
        raise ValueError("candidates.csv must contain a unique key ('Candidate ID' or 'Email') and 'Resume Link'.")

    rows = []
    errors = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting resumes"):
        key = str(row[key_col]).strip()
        email = str(row["Email"]).strip() if "Email" in df.columns else ""
        resume_url = str(row["Resume Link"]).strip()

        pdf_path = resumes_dir / f"{key}.pdf"
        text_path = resume_text_dir / f"{key}.txt"

        try:
            download_url = google_drive_download_url(resume_url)
            download_file(download_url, pdf_path)
            text = extract_text_from_pdf(pdf_path)
            text_path.write_text(text, encoding="utf-8")
            rows.append({key_col: key, "Email": email, "resume_text": text})
        except Exception as e:
            errors.append({key_col: key, "Email": email, "error": str(e)})

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_dir / "resume_text.csv", index=False)
    pd.DataFrame(errors).to_csv(out_dir / "resume_errors.csv", index=False)

    print(f"Extracted: {len(rows)} resumes; Failed: {len(errors)}")


if __name__ == "__main__":
    main()

