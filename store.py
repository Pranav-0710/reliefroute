"""Shared state storage: Cloud Firestore when configured, otherwise a local JSON file.

Firestore is reached through its REST API with a service-account token (google-auth already comes
with the Gemini SDK), so no extra client library is needed. Each state document lives at
<collection>/<name> with one string field, "json". Every write is conditional on the document's
updateTime from the read, so concurrent server instances never overwrite each other: on a
conflict the read-modify-write is retried.

Configure with FIREBASE_SERVICE_ACCOUNT = the service-account key JSON (raw or base64).
"""
import base64
import json
import os
import threading
import time

import httpx

API = "https://firestore.googleapis.com/v1"
SCOPES = ["https://www.googleapis.com/auth/datastore"]
_file_lock = threading.Lock()
_auth_lock = threading.Lock()
_creds = {}  # service-account email -> google.oauth2 credentials


class Conflict(Exception):
    """Someone else wrote the document between our read and our write."""


def _service_account():
    raw = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()
    if not raw:
        return None
    try:
        info = json.loads(raw if raw.startswith("{") else base64.b64decode(raw))
    except ValueError:
        return None
    return info if info.get("type") == "service_account" and info.get("project_id") else None


def backend():
    return "firestore" if _service_account() else "file"


def _token(info):
    from google.auth.transport.requests import Request
    from google.oauth2 import service_account

    key = info["client_email"]
    with _auth_lock:
        creds = _creds.get(key)
        if creds is None:
            creds = _creds[key] = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
        if not creds.valid:
            creds.refresh(Request())
        return creds.token


def _doc_url(info, name):
    db = os.getenv("FIRESTORE_DATABASE", "(default)")
    coll = os.getenv("FIRESTORE_COLLECTION", "reliefroute")
    return f"{API}/projects/{info['project_id']}/databases/{db}/documents/{coll}/{name}"


# ---- Firestore -----------------------------------------------------------------------
def _fs_read(info, name):
    r = httpx.get(_doc_url(info, name), headers={"Authorization": f"Bearer {_token(info)}"}, timeout=10)
    if r.status_code == 404:
        return None, None
    r.raise_for_status()
    doc = r.json()
    return json.loads(doc["fields"]["json"]["stringValue"]), doc["updateTime"]


def _fs_write(info, name, data, update_time):
    pre = {"currentDocument.updateTime": update_time} if update_time else {"currentDocument.exists": "false"}
    r = httpx.patch(_doc_url(info, name), params=pre, timeout=10,
                    headers={"Authorization": f"Bearer {_token(info)}"},
                    json={"fields": {"json": {"stringValue": json.dumps(data)}}})
    if r.status_code in (400, 409) and any(s in r.text for s in ("FAILED_PRECONDITION", "ALREADY_EXISTS", "ABORTED")):
        raise Conflict(name)
    r.raise_for_status()


# ---- local file ------------------------------------------------------------------------
def _file_read(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _file_write(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)


# ---- public API -------------------------------------------------------------------------
def read(name, path, default):
    """Current state of document `name` (or default() if it does not exist yet)."""
    info = _service_account()
    data = _fs_read(info, name)[0] if info else _file_read(path)
    return data if isinstance(data, dict) else default()


def update(name, path, default, mutate, retries=6):
    """Atomically read-modify-write document `name`. mutate(state) edits state in place; its
    return value is returned. Retries on concurrent writes; the file backend uses a process lock."""
    info = _service_account()
    if not info:
        with _file_lock:
            data = _file_read(path)
            data = data if isinstance(data, dict) else default()
            result = mutate(data)
            _file_write(path, data)
            return result
    for attempt in range(retries):
        data, stamp = _fs_read(info, name)
        data = data if isinstance(data, dict) else default()
        result = mutate(data)
        try:
            _fs_write(info, name, data, stamp)
            return result
        except Conflict:
            time.sleep(0.05 * (attempt + 1))
    raise RuntimeError(f"could not update {name}: too many concurrent writes")
