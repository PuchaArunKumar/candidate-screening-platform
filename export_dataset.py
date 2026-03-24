import argparse
from pathlib import Path

import pandas as pd


def export(xlsx_path: Path, out_dir: Path) -> None:
    df = pd.read_excel(xlsx_path)

    required = [
        "name",
        "email",
        "college",
        "branch",
        "cgpa",
        "best_ai_project",
        "research_work",
        "github",
        "resume",
        "test_la",
        "test_code",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in XLSX: {missing}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Phase 2 expected fields (assignment names)
    candidates = df[
        [
            "s_no",
            "name",
            "email",
            "college",
            "branch",
            "cgpa",
            "best_ai_project",
            "research_work",
            "github",
            "resume",
        ]
    ].copy()
    candidates = candidates.rename(
        columns={
            "s_no": "Candidate ID",
            "name": "Name",
            "email": "Email",
            "college": "College",
            "branch": "Branch",
            "cgpa": "CGPA",
            "best_ai_project": "Best AI Project",
            "research_work": "Research Work",
            "github": "GitHub Profile",
            "resume": "Resume Link",
        }
    )
    candidates.to_csv(out_dir / "candidates.csv", index=False)

    # Phase 8 expected fields
    test_results = df[["s_no", "email", "test_la", "test_code"]].copy()
    test_results = test_results.rename(
        columns={
            "s_no": "Candidate ID",
            "email": "Email",
            "test_la": "test_la",
            "test_code": "test_code",
        }
    )
    # Ensure numeric consistency for later scoring.
    test_results["test_la"] = pd.to_numeric(test_results["test_la"], errors="coerce")
    test_results["test_code"] = pd.to_numeric(test_results["test_code"], errors="coerce")
    test_results.to_csv(out_dir / "test_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export XLSX dataset into assignment CSVs.")
    parser.add_argument(
        "--xlsx",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\AI Assignment\candidate_dataset.xlsx",
        help="Path to candidate_dataset.xlsx",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data",
        help="Output directory for CSV files.",
    )
    args = parser.parse_args()
    export(Path(args.xlsx), Path(args.out))


if __name__ == "__main__":
    main()

