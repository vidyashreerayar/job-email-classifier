from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "labeled_emails.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "human_labeled_emails.csv"


def load_emails():
    return pd.read_csv(INPUT_FILE)


def save_progress(df):
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )


def display_email(row, number, total):
    print("\n" + "=" * 80)
    print(f"EMAIL {number} / {total}")
    print("=" * 80)

    print(f"\nSubject:\n{row['subject']}")

    print("\nBody:")
    print("-" * 80)

    body = str(row["body"])

    # Limit the initial display so very long emails do not overwhelm
    # the annotation interface.
    print(body[:5000])

    if len(body) > 5000:
        print("\n[Body truncated for display]")

    print("-" * 80)
    print(f"\nSuggested label: {row['suggested_label']}")

    print(
        "\nChoose label:"
        "\n  P = JOB_POSITIVE"
        "\n  R = JOB_REJECTION"
        "\n  O = JOB_OTHER"
        "\n  S = SKIP"
        "\n  Q = QUIT"
    )


def annotate_emails(df):
    if "human_label" not in df.columns:
        df["human_label"] = ""

    total = len(df)

    for index, row in df.iterrows():

        if pd.notna(row["human_label"]) and str(
            row["human_label"]
        ).strip():
            continue

        display_email(
            row,
            index + 1,
            total,
        )

        while True:
            choice = input("\nYour choice: ").strip().upper()

            if choice == "P":
                df.at[index, "human_label"] = "JOB_POSITIVE"
                break

            if choice == "R":
                df.at[index, "human_label"] = "JOB_REJECTION"
                break

            if choice == "O":
                df.at[index, "human_label"] = "JOB_OTHER"
                break

            if choice == "S":
                print("Skipped.")
                break

            if choice == "Q":
                save_progress(df)
                print(f"\nProgress saved to: {OUTPUT_FILE}")
                return df

            print(
                "Invalid choice. "
                "Use P, R, O, S, or Q."
            )

        save_progress(df)

    return df


def main():
    df = load_emails()

    print(f"Loaded emails: {len(df)}")
    print(f"Output file: {OUTPUT_FILE}")

    df = annotate_emails(df)

    print("\nAnnotation complete.")
    print("\nHuman label distribution:")
    print(
        df["human_label"]
        .replace("", pd.NA)
        .dropna()
        .value_counts()
    )


if __name__ == "__main__":
    main()