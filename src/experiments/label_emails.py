from pathlib import Path
import re

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "emails.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "weak_labels.csv"


# Weak-supervision labels
JOB_POSITIVE = "JOB_POSITIVE"
JOB_REJECTION = "JOB_REJECTION"
JOB_OTHER = "JOB_OTHER"
ABSTAIN = "ABSTAIN"


def normalize_text(subject, body):
    text = f"{subject} {body}".lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text, patterns):
    return any(
        re.search(pattern, text, flags=re.IGNORECASE)
        for pattern in patterns
    )


# -------------------------------------------------------------------
# Labeling Function 1: explicit rejection language
# -------------------------------------------------------------------

def lf_explicit_rejection(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English - explicit rejection
        r"\bwe will not be moving forward\b",
        r"\bwe're not moving forward\b",
        r"\bnot moving forward with your application\b",
        r"\bnot progressing further\b",
        r"\bwill not proceed with your application\b",
        r"\bnot proceed with your application\b",
        r"\bnot selected\b",
        r"\bapplication.*unsuccessful\b",
        r"\bdecline your application\b",
        r"\breject your application\b",
        r"\bwe decided not to\b",
        r"\bwe have decided not to\b",
        r"\bmove forward with other candidates\b",
        r"\bmove forward with another candidate\b",
        r"\bdecided to move forward with another candidate\b",
        r"\bdecided to move forward with other candidates\b",
        r"\bnot take your application further\b",
        r"\bnot able to take your application further\b",
        r"\bnot able to take your application any further\b",
        r"\bwill not continue with your application\b",
        r"\bhiring manager has decided to move forward\b",
        r"\bwe have decided to move forward with another candidate\b",
        r"\bregrettably.*not able to take your application further\b",
        r"\bregrettably.*not able to take your application any further\b",

        # German - explicit rejection
        r"\babsage\b",
        r"\babgelehnt\b",
        r"\bnicht berücksichtigt\b",
        r"\bnicht beruecksichtigt\b",
        r"\bnicht weiterverfolgen\b",
    ]

    if contains_any(text, patterns):
        return JOB_REJECTION

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 2: explicit interview / assessment progression
# -------------------------------------------------------------------

def lf_interview_or_assessment(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\binterview invitation\b",
        r"\binvite you.*interview\b",
        r"\binvited.*interview\b",
        r"\bschedule.*interview\b",
        r"\btechnical interview\b",
        r"\bphone interview\b",
        r"\binterview.*schedule\b",
        r"\bwould like to speak with you\b",
        r"\bproceed to the next stage\b",
        r"\bproceed to the next step\b",
        r"\bnext step.*interview\b",
        r"\bnext stage.*interview\b",
        r"\bassessment invitation\b",
        r"\binvite you.*assessment\b",
        r"\binvited.*assessment\b",

        # German
        r"\bvorstellungsgespräch.*einladung\b",
        r"\beinladung.*vorstellungsgespräch\b",
        r"\bzum vorstellungsgespräch eingeladen\b",
        r"\bvorstellungsgespräch.*vereinbaren\b",
        r"\bvorstellungsgespräch.*termin\b",
        r"\beinladung.*assessment\b",
        r"\bzum assessment eingeladen\b",
        r"\bnächsten schritt.*interview\b",
        r"\bnaechsten schritt.*interview\b",
    ]

    if contains_any(text, patterns):
        return JOB_POSITIVE

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 3: application acknowledgement
# -------------------------------------------------------------------

def lf_application_acknowledgement(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\bthank you for your application\b",
        r"\bthank you for applying\b",
        r"\bthanks for applying\b",
        r"\bapplication received\b",
        r"\bapplication has been received\b",
        r"\bwe received your application\b",
        r"\bwe have received your application\b",
        r"\bbeing reviewed\b",
        r"\bcarefully reviewing\b",
        r"\breview your application\b",
        r"\bwe will review your documents\b",
        r"\bwe will review your application\b",
        r"\bwe are reviewing your application\b",
        r"\bwe will carefully review\b",
        r"\bbeing considered\b",
        r"\bbeing processed\b",
        r"\bbeing examined\b",
        r"\bbeing evaluated\b",
        r"\bbeing considered for the position\b",
        r"\bwill contact you soon\b",
        r"\bcontact you shortly\b",
        r"\bcontact you with the next steps\b",

        # German
        r"\beingangsbestätigung\b",
        r"\bdanke für ihre bewerbung\b",
        r"\bvielen dank für ihre bewerbung\b",
        r"\bvielen dank für dein interesse\b",
        r"\bdanke für dein interesse\b",
        r"\bbewerbung erhalten\b",
        r"\bbewerbung eingegangen\b",
        r"\bunterlagen.*prüfen\b",
        r"\bunterlagen.*pruefen\b",
        r"\bunterlagen.*sorgfältig prüfen\b",
        r"\bunterlagen.*sorgfaeltig pruefen\b",
        r"\bwerden.*geprüft\b",
        r"\bwerden.*geprueft\b",
        r"\bsorgfältig geprüft\b",
        r"\bsorgfaeltig geprueft\b",
        r"\bwir prüfen ihre unterlagen\b",
        r"\bwir werden ihre unterlagen\b",
        r"\bmit den nächsten schritten\b",
        r"\bmit den naechsten schritten\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 4: job alert / recruiter marketing
# -------------------------------------------------------------------

def lf_job_alert_or_marketing(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\bjob alert\b",
        r"\bjob alerts\b",
        r"\bnew jobs\b",
        r"\bnew job\b",
        r"\bnew position\b",
        r"\bnew positions\b",
        r"\bjob recommendations\b",
        r"\brecommended jobs\b",
        r"\bcareer opportunities\b",
        r"\bcareers newsletter\b",
        r"\bjobs newsletter\b",
        r"\bheadhunter\b",

        # German
        r"\bkarrieremöglichkeit\b",
        r"\bkarrieremoeglichkeit\b",
        r"\bgespeicherte suche\b",
        r"\bstellenangebote\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 5: career portal / account notification
# -------------------------------------------------------------------

def lf_career_portal_notification(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        r"\bnew password\b",
        r"\bpassword reset\b",
        r"\bforgot your password\b",
        r"\bapplicant profile\b",
        r"\bbewerberprofil\b",
        r"\bcareer portal\b",
        r"\bkarriereportal\b",
        r"\btalent community\b",
        r"\bprofile was created\b",
        r"\bprofil.*erstellt\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 6: generic job-application context
# -------------------------------------------------------------------

def lf_job_application_context(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        r"\bapplication\b",
        r"\byour application\b",
        r"\byour candidacy\b",
        r"\bcandidacy\b",
        r"\bapplying\b",
        r"\bapplied\b",
        r"\bbewerbung\b",
        r"\bbewerben\b",
        r"\bbewerber\b",
    ]

    # This LF intentionally abstains.
    # It identifies job-application context but does not provide
    # enough information to distinguish the three target classes.

    if contains_any(text, patterns):
        return ABSTAIN

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 7: job recommendations
# -------------------------------------------------------------------

def lf_job_recommendation(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\bjob recommendation\b",
        r"\bjob recommendations\b",
        r"\brecommended jobs\b",
        r"\bthis job is a potential match\b",
        r"\bpotential match\b",
        r"\bjobs you might like\b",
        r"\bjobs you may be interested in\b",
        r"\bjobs you don't want to miss\b",
        r"\bjobs you do not want to miss\b",

        # German
        r"\bjobs,? die du nicht verpassen willst\b",
        r"\bjobs,? die sie nicht verpassen wollen\b",
        r"\bpassende stellen\b",
        r"\bpassende jobs\b",
        r"\bstellen.*vorschläge\b",
        r"\bjob.*vorschläge\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 8: talent network / recruiter database
# -------------------------------------------------------------------

def lf_talent_network_or_recruiter(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\btalent network\b",
        r"\btalent community\b",
        r"\brecruiting database\b",
        r"\bcandidate database\b",
        r"\bcandidate pool\b",
        r"\bwe're pleased you're interested\b",
        r"\bwe are pleased you are interested\b",
        r"\bjoin our talent\b",

        # German
        r"\bkarrierenetzwerk\b",
        r"\btalentnetzwerk\b",
        r"\bkandidatendatenbank\b",
        r"\bbewerberdatenbank\b",
        r"\brecruiting-kandidatendatenbank\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# Labeling Function 9: candidate account / portal notification
# -------------------------------------------------------------------

def lf_candidate_account_notification(subject, body):
    text = normalize_text(subject, body)

    patterns = [
        # English
        r"\bcreate password\b",
        r"\bactivate your account\b",
        r"\bactivate your profile\b",
        r"\bverify your candidate account\b",
        r"\bcandidate account\b",
        r"\bapplicant account\b",
        r"\bcandidate portal\b",
        r"\bapplicant portal\b",
        r"\brecruiting portal\b",
        r"\baccess.*candidate portal\b",

        # German
        r"\beinmalkennwort\b",
        r"\beinmalpasswort\b",
        r"\bverifizierung des kandidatenkontos\b",
        r"\bkandidatenkonto\b",
        r"\bbewerberkonto\b",
        r"\bzugang zum bewerbermanagement\b",
        r"\bzugang.*bewerber\b",
        r"\baktivieren sie ihren account\b",
        r"\baktivieren sie ihr konto\b",
    ]

    if contains_any(text, patterns):
        return JOB_OTHER

    return ABSTAIN


# -------------------------------------------------------------------
# All labeling functions
# -------------------------------------------------------------------

LABELING_FUNCTIONS = {
    "lf_explicit_rejection": lf_explicit_rejection,
    "lf_interview_or_assessment": lf_interview_or_assessment,
    "lf_application_acknowledgement": lf_application_acknowledgement,
    "lf_job_alert_or_marketing": lf_job_alert_or_marketing,
    "lf_career_portal_notification": lf_career_portal_notification,
    "lf_job_application_context": lf_job_application_context,
    "lf_job_recommendation": lf_job_recommendation,
    "lf_talent_network_or_recruiter": lf_talent_network_or_recruiter,
    "lf_candidate_account_notification": lf_candidate_account_notification,
}


# -------------------------------------------------------------------
# Apply labeling functions
# -------------------------------------------------------------------

def apply_labeling_functions(df):
    result = df.copy()

    # First generate the independent LF outputs.
    for name, function in LABELING_FUNCTIONS.items():
        result[name] = result.apply(
            lambda row: function(
                row["subject"],
                row["body"],
            ),
            axis=1,
        )

    # ---------------------------------------------------------------
    # Resolve semantic precedence.
    #
    # Explicit rejection is stronger evidence than generic
    # application-acknowledgement, job-alert, or career-portal
    # language. Rejection emails frequently contain those phrases.
    # ---------------------------------------------------------------

    rejection_mask = (
        result["lf_explicit_rejection"] == JOB_REJECTION
    )

    # Rejection > application acknowledgement
    result.loc[
        rejection_mask,
        "lf_application_acknowledgement",
    ] = ABSTAIN

    # Rejection > job alert / marketing
    result.loc[
        rejection_mask,
        "lf_job_alert_or_marketing",
    ] = ABSTAIN

    # Rejection > career portal notification
    result.loc[
        rejection_mask,
        "lf_career_portal_notification",
    ] = ABSTAIN

    # ---------------------------------------------------------------
    # Explicit interview / assessment progression is stronger than
    # generic application acknowledgement.
    # ---------------------------------------------------------------

    positive_mask = (
        result["lf_interview_or_assessment"] == JOB_POSITIVE
    )

    result.loc[
        positive_mask,
        "lf_application_acknowledgement",
    ] = ABSTAIN

    return result


# -------------------------------------------------------------------
# Analyze labeling-function coverage
# -------------------------------------------------------------------

def analyze_coverage(df):
    lf_columns = list(LABELING_FUNCTIONS.keys())

    print("\n" + "=" * 80)
    print("LABELING FUNCTION COVERAGE")
    print("=" * 80)

    for column in lf_columns:
        non_abstain = (df[column] != ABSTAIN).sum()
        coverage = non_abstain / len(df)

        print(
            f"{column}: "
            f"{non_abstain}/{len(df)} "
            f"({coverage:.1%})"
        )


# -------------------------------------------------------------------
# Analyze conflicts between labeling functions
# -------------------------------------------------------------------

def analyze_conflicts(df):
    lf_columns = list(LABELING_FUNCTIONS.keys())

    label_columns = df[lf_columns]

    non_abstain_counts = (
        label_columns != ABSTAIN
    ).sum(axis=1)

    conflicting_rows = []

    for index, row in label_columns.iterrows():
        labels = set(
            value
            for value in row
            if value != ABSTAIN
        )

        if len(labels) > 1:
            conflicting_rows.append(index)

    print("\n" + "=" * 80)
    print("LABELING FUNCTION CONFLICTS")
    print("=" * 80)

    print(
        f"Emails with at least one LF vote: "
        f"{(non_abstain_counts > 0).sum()}"
    )

    print(
        f"Emails with no LF votes: "
        f"{(non_abstain_counts == 0).sum()}"
    )

    print(
        f"Emails with multiple LF votes: "
        f"{(non_abstain_counts > 1).sum()}"
    )

    print(
        f"Emails with conflicting labels: "
        f"{len(conflicting_rows)}"
    )

    if conflicting_rows:
        print("\nExample conflicts:")

        for index in conflicting_rows[:10]:
            print("\n" + "-" * 80)
            print(f"Email ID: {df.loc[index, 'email_id']}")
            print(f"Subject: {df.loc[index, 'subject']}")

            for column in lf_columns:
                value = df.loc[index, column]

                if value != ABSTAIN:
                    print(f"{column}: {value}")


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    result = apply_labeling_functions(df)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Processed emails: {len(result)}")
    print(f"Saved weak labels to: {OUTPUT_FILE}")

    analyze_coverage(result)
    analyze_conflicts(result)


if __name__ == "__main__":
    main()