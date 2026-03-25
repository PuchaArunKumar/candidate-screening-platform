import argparse
from pathlib import Path

import pandas as pd


def normalize_cgpa(cgpa_series: pd.Series, max_cgpa: float = 10.0) -> pd.Series:
    cgpa_num = pd.to_numeric(cgpa_series, errors="coerce").fillna(0.0)
    return (cgpa_num / max_cgpa).clip(0.0, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 6: rank candidates and shortlist.")
    parser.add_argument(
        "--candidates_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\candidates.csv",
    )
    parser.add_argument(
        "--ai_scores_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\ai_scores.csv",
    )
    parser.add_argument(
        "--github_analysis_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\github_analysis.csv",
    )
    parser.add_argument(
        "--out_rankings_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\rankings.csv",
    )
    parser.add_argument(
        "--out_shortlist_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\shortlisted_candidates.csv",
    )
    parser.add_argument("--w_resume", type=float, default=0.5)
    parser.add_argument("--w_github", type=float, default=0.3)
    parser.add_argument("--w_cgpa", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=70.0)
    args = parser.parse_args()

    if abs((args.w_resume + args.w_github + args.w_cgpa) - 1.0) > 1e-6:
        raise ValueError("Weights must sum to 1.0")

    candidates = pd.read_csv(args.candidates_csv)
    ai = pd.read_csv(args.ai_scores_csv)
    gh = pd.read_csv(args.github_analysis_csv)

    merged = candidates.merge(ai, on="Candidate ID", how="inner").merge(gh, on="Candidate ID", how="left")

    # Handle missing github analysis if any.
    merged["github_technical_score"] = pd.to_numeric(merged["github_technical_score"], errors="coerce").fillna(0).astype(int)
    merged["resume_ai_score"] = pd.to_numeric(merged["resume_ai_score"], errors="coerce").fillna(0).astype(int)

    cgpa_norm = normalize_cgpa(merged["CGPA"])
    merged["cgpa_score"] = (cgpa_norm * 100).round(1)

    merged["overall_score"] = (
        args.w_resume * merged["resume_ai_score"]
        + args.w_github * merged["github_technical_score"]
        + args.w_cgpa * merged["cgpa_score"]
    ).round(1)

    merged = merged.sort_values("overall_score", ascending=False)

    merged[[
        "Candidate ID",
        "Name",
        "Email",
        "CGPA",
        "resume_ai_score",
        "github_technical_score",
        "cgpa_score",
        "overall_score",
    ]].to_csv(args.out_rankings_csv, index=False)

    shortlist = merged[merged["overall_score"] >= args.threshold]
    shortlist[["Candidate ID", "Name", "Email", "overall_score"]].to_csv(args.out_shortlist_csv, index=False)

    print(f"Wrote rankings: {args.out_rankings_csv} ({len(merged)} candidates)")
    print(f"Shortlisted (>= {args.threshold}): {args.out_shortlist_csv} ({len(shortlist)} candidates)")


if __name__ == "__main__":
    main()

