from pathlib import Path
import base64
import csv
import time

import pandas as pd
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS_FILE = PROJECT_ROOT / "credentials.json"
TOKEN_FILE = PROJECT_ROOT / "token.json"


def authenticate_gmail():
    credentials = None

    if TOKEN_FILE.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_FILE,
            SCOPES,
        )

    if credentials is None or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            credentials.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                CREDENTIALS_FILE,
                SCOPES,
            )
            credentials = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(credentials.to_json())

    return credentials


def get_gmail_service():
    credentials = authenticate_gmail()
    return build("gmail", "v1", credentials=credentials)


def get_message_count(service):
    response = service.users().messages().list(
        userId="me",
        maxResults=1,
    ).execute()

    return response.get("resultSizeEstimate", 0)


def get_message(service, message_id, max_retries=5):
    for attempt in range(max_retries):
        try:
            return service.users().messages().get(
                userId="me",
                id=message_id,
                format="full",
            ).execute()

        except HttpError as error:
            if error.resp.status in (429, 403):
                wait_time = 2 ** attempt

                print(
                    f"Rate limit reached. "
                    f"Waiting {wait_time} seconds..."
                )

                time.sleep(wait_time)

            else:
                raise

    raise RuntimeError(
        f"Failed to retrieve message {message_id} "
        f"after {max_retries} retries."
    )


def decode_message_body(payload):
    data = payload.get("body", {}).get("data")

    if data:
        return base64.urlsafe_b64decode(data).decode(
            "utf-8",
            errors="replace",
        )

    for part in payload.get("parts", []):
        body = decode_message_body(part)

        if body:
            return body

    return ""


def get_message_record(service, message_id):
    message = get_message(service, message_id)

    headers = {
        header["name"]: header["value"]
        for header in message["payload"].get("headers", [])
    }

    body = decode_message_body(message["payload"])

    return {
        "email_id": message["id"],
        "subject": headers.get("Subject", ""),
        "body": body,
        "label": "",
    }


def list_message_ids(service, max_results=100, label_ids=None):
    message_ids = []
    page_token = None

    while True:
        response = service.users().messages().list(
            userId="me",
            maxResults=max_results,
            pageToken=page_token,
            labelIds=label_ids,
        ).execute()

        message_ids.extend(
            message["id"]
            for message in response.get("messages", [])
        )

        page_token = response.get("nextPageToken")

        if not page_token:
            break

    return message_ids


def save_records_to_csv(records, output_file):
    fieldnames = [
        "email_id",
        "subject",
        "body",
        "label",
    ]

    with open(
        output_file,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(records)


def download_inbox_emails(
    service,
    output_file,
    batch_size=20,
    delay_between_messages=0.5,
):
    output_file = Path(output_file)

    existing_ids = set()

    if output_file.exists() and output_file.stat().st_size > 0:
        existing_df = pd.read_csv(output_file)

        if "email_id" in existing_df.columns:
            existing_ids = set(
                existing_df["email_id"]
                .dropna()
                .astype(str)
            )

    message_ids = list_message_ids(
        service,
        label_ids=["INBOX"],
    )

    remaining_ids = [
        message_id
        for message_id in message_ids
        if message_id not in existing_ids
    ]

    print(f"Total Inbox messages: {len(message_ids)}")
    print(f"Already downloaded: {len(existing_ids)}")
    print(f"Remaining: {len(remaining_ids)}")

    fieldnames = [
        "email_id",
        "subject",
        "body",
        "label",
    ]

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    file_exists = (
        output_file.exists()
        and output_file.stat().st_size > 0
    )

    with open(
        output_file,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        if not file_exists:
            writer.writeheader()

        for batch_start in range(
            0,
            len(remaining_ids),
            batch_size,
        ):
            batch_ids = remaining_ids[
                batch_start:batch_start + batch_size
            ]

            print(
                f"Downloading batch "
                f"{batch_start + 1}-"
                f"{batch_start + len(batch_ids)} "
                f"of {len(remaining_ids)}..."
            )

            for message_id in batch_ids:
                record = get_message_record(
                    service,
                    message_id,
                )

                writer.writerow(record)
                file.flush()

                time.sleep(delay_between_messages)

            print(
                f"Batch complete. "
                f"Saved {len(batch_ids)} messages."
            )