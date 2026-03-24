import argparse
from pathlib import Path

import pandas as pd


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 8: merge test results and compute final scores.")
    parser.add_argument(
        "--shortlist_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\shortlisted_candidates.csv",
    )
    parser.add_argument(
        "--test_results_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\test_results.csv",
    )
    parser.add_argument(
        "--out_final_rankings_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\final_rankings.csv",
    )
    parser.add_argument(
        "--out_qualified_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\qualified_candidates.csv",
    )
    parser.add_argument("--threshold", type=float, default=60.0, help="Cutoff on test performance (0-100).")
    parser.add_argument("--w_test_la", type=float, default=0.4)
    parser.add_argument("--w_test_code", type=float, default=0.6)
    parser.add_argument("--w_pipeline", type=float, default=0.7, help="Weight of Phase 6 overall in final score.")
    parser.add_argument("--w_test", type=float, default=0.3, help="Weight of test performance in final score.")
    args = parser.parse_args()

    if abs((args.w_test_la + args.w_test_code) - 1.0) > 1e-6:
        raise ValueError("w_test_la + w_test_code must sum to 1.0")
    if abs((args.w_pipeline + args.w_test) - 1.0) > 1e-6:
        raise ValueError("w_pipeline + w_test must sum to 1.0")

    shortlist = pd.read_csv(args.shortlist_csv)
    tests = pd.read_csv(args.test_results_csv)

    # Merge on Candidate ID (more robust for this dataset than email).
    merged = shortlist.merge(tests, on="Candidate ID", how="inner")

    merged["test_la"] = pd.to_numeric(merged["test_la"], errors="coerce").fillna(0.0)
    merged["test_code"] = pd.to_numeric(merged["test_code"], errors="coerce").fillna(0.0)

    # Assume both tests are already on 0..100 scale.
    merged["test_performance_score"] = (
        args.w_test_la * merged["test_la"] + args.w_test_code * merged["test_code"]
    ).round(1)

    merged["final_score"] = (
        args.w_pipeline * merged["overall_score"] + args.w_test * merged["test_performance_score"]
    ).round(1)

    merged = merged.sort_values("final_score", ascending=False)

    # Pandas merge may create Email_x/Email_y. Normalize for output.
    if "Email" not in merged.columns:
        if "Email_x" in merged.columns:
            merged = merged.rename(columns={"Email_x": "Email"})
        elif "Email_y" in merged.columns:
            merged = merged.rename(columns={"Email_y": "Email"})

    merged[
        [
            "Candidate ID",
            "Name",
            "Email",
            "overall_score",
            "test_la",
            "test_code",
            "test_performance_score",
            "final_score",
        ]
    ].to_csv(args.out_final_rankings_csv, index=False)

    qualified = merged[merged["test_performance_score"] >= args.threshold]
    qualified[["Candidate ID", "Name", "Email", "final_score", "test_performance_score"]].to_csv(
        args.out_qualified_csv, index=False
    )

    print(f"Wrote final rankings: {args.out_final_rankings_csv} ({len(merged)} candidates)")
    print(f"Qualified for interviews (test >= {args.threshold}): {args.out_qualified_csv} ({len(qualified)} candidates)")


if __name__ == "__main__":
    main()

