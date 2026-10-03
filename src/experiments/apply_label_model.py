from pathlib import Path

import pandas as pd
from snorkel.labeling.model import LabelModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "weak_labels.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "model_labeled_emails.csv"
)


# -------------------------------------------------------------------
# Label mapping
# -------------------------------------------------------------------

ABSTAIN = -1

LABEL_MAP = {
    "JOB_POSITIVE": 0,
    "JOB_REJECTION": 1,
    "JOB_OTHER": 2,
}

REVERSE_LABEL_MAP = {
    -1: "ABSTAIN",
    0: "JOB_POSITIVE",
    1: "JOB_REJECTION",
    2: "JOB_OTHER",
}


# -------------------------------------------------------------------
# Load labeling-function outputs
# -------------------------------------------------------------------

def load_labeling_function_outputs():
    df = pd.read_csv(INPUT_FILE)

    lf_columns = [
        column
        for column in df.columns
        if column.startswith("lf_")
    ]

    if not lf_columns:
        raise ValueError(
            "No labeling-function columns found."
        )

    return df, lf_columns


# -------------------------------------------------------------------
# Convert LF labels to Snorkel integer format
# -------------------------------------------------------------------

def convert_labels(df, lf_columns):
    label_matrix = df[lf_columns].copy()

    for column in lf_columns:
        label_matrix[column] = (
            label_matrix[column]
            .map(LABEL_MAP)
            .fillna(ABSTAIN)
            .astype(int)
        )

    return label_matrix.values


# -------------------------------------------------------------------
# Train label model
# -------------------------------------------------------------------

def train_label_model(label_matrix):
    label_model = LabelModel(
        cardinality=len(LABEL_MAP),
        verbose=False,
    )

    label_model.fit(
        L_train=label_matrix,
        n_epochs=500,
        seed=42,
    )

    return label_model


# -------------------------------------------------------------------
# Generate weak labels
# -------------------------------------------------------------------

def generate_labels(label_model, label_matrix):
    predicted_labels = label_model.predict(
        L=label_matrix
    )

    predicted_probabilities = label_model.predict_proba(
        L=label_matrix
    )

    return predicted_labels, predicted_probabilities


# -------------------------------------------------------------------
# Save results
# -------------------------------------------------------------------

def save_results(
    df,
    predicted_labels,
    predicted_probabilities,
):
    result = df.copy()

    result["weak_label"] = [
        REVERSE_LABEL_MAP[label]
        for label in predicted_labels
    ]

    result["label_confidence"] = (
        predicted_probabilities.max(axis=1)
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    return result


# -------------------------------------------------------------------
# Analyze results
# -------------------------------------------------------------------

def analyze_results(result):
    print("\n" + "=" * 80)
    print("WEAK LABEL DISTRIBUTION")
    print("=" * 80)

    print(
        result["weak_label"]
        .value_counts()
        .to_string()
    )

    print("\n" + "=" * 80)
    print("CONFIDENCE DISTRIBUTION")
    print("=" * 80)

    print(
        result["label_confidence"]
        .describe()
        .to_string()
    )

    print("\n" + "=" * 80)
    print("LOW-CONFIDENCE LABELS")
    print("=" * 80)

    low_confidence = (
        result[
            result["label_confidence"] < 0.60
        ][
            [
                "email_id",
                "subject",
                "weak_label",
                "label_confidence",
            ]
        ]
        .sort_values("label_confidence")
    )

    print(
        low_confidence
        .head(20)
        .to_string(index=False)
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    df, lf_columns = (
        load_labeling_function_outputs()
    )

    print(f"Emails loaded: {len(df)}")

    print("\nLabeling functions:")
    for column in lf_columns:
        print(f"- {column}")

    label_matrix = convert_labels(
        df,
        lf_columns,
    )

    print(
        f"\nLabel matrix shape: "
        f"{label_matrix.shape}"
    )

    label_model = train_label_model(
        label_matrix
    )

    predicted_labels, predicted_probabilities = (
        generate_labels(
            label_model,
            label_matrix,
        )
    )

    result = save_results(
        df,
        predicted_labels,
        predicted_probabilities,
    )

    print(
        f"\nSaved model-generated labels to:"
        f"\n{OUTPUT_FILE}"
    )

    analyze_results(result)


if __name__ == "__main__":
    main()