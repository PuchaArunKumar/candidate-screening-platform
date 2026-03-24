import argparse
import secrets
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 7: prepare automated emails with test links (demo mode).")
    parser.add_argument(
        "--shortlist_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\shortlisted_candidates.csv",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=r"c:\Users\arunk\OneDrive\Desktop\files\candidate_screening_platform\data\emails_to_send.csv",
    )
    parser.add_argument(
        "--test_link_base",
        type=str,
        default="https://your-app-domain.com/tests/take?token=",
        help="Base URL used to build the per-candidate test link.",
    )
    parser.add_argument(
        "--email_subject",
        type=str,
        default="Your Technical Test Link",
    )
    args = parser.parse_args()

    shortlist = pd.read_csv(args.shortlist_csv)
    required = {"Candidate ID", "Email", "overall_score", "Name"}
    missing = [c for c in required if c not in shortlist.columns]
    if missing:
        # Some datasets might omit Name; keep it flexible.
        missing_core = [c for c in {"Candidate ID", "Email", "overall_score"} if c not in shortlist.columns]
        if missing_core:
            raise ValueError(f"shortlisted CSV missing required columns: {missing_core}")

    rows = []
    for _, row in shortlist.iterrows():
        email = str(row["Email"]).strip()
        candidate_id = row["Candidate ID"]
        token = secrets.token_urlsafe(24)
        test_link = args.test_link_base + token

        name = str(row["Name"]).strip() if "Name" in row else ""
        subject = args.email_subject
        body = (
            f"Hi {name or 'there'},\n\n"
            "Thanks for your interest. Here is your link to complete the technical test.\n"
            f"Test link: {test_link}\n\n"
            "Best regards,\nRecruitment Team"
        )
        rows.append({"Candidate ID": candidate_id, "Email": email, "subject": subject, "body": body, "test_link": test_link})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"Wrote {len(rows)} email rows to {args.out_csv}")


if __name__ == "__main__":
    main()

