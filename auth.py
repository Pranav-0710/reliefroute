"""Coordinator sign-in with Firebase Authentication (Google accounts), verified on the server.

Off until configured; then only signed-in coordinators can dispatch, edit depots and fleet, restock
or reset. Field reporters and drivers never sign in: reporters use the open /report page, drivers
the secret link of their truck.

    COORDINATOR_EMAILS    comma-separated Google accounts allowed to coordinate
    FIREBASE_WEB_API_KEY  the web app's apiKey (public by design), or the firebaseConfig snippet it is in
    FIREBASE_PROJECT_ID   optional; defaults to the project of FIREBASE_SERVICE_ACCOUNT
"""
import os
import re
import threading
import time

import store

_verified = {}  # ID token -> (email, expiry); saves fetching Google's signing keys on every request
_lock = threading.Lock()


class Denied(Exception):
    pass


def project_id():
    return os.getenv("FIREBASE_PROJECT_ID") or (store._service_account() or {}).get("project_id")


def allowed():
    return {e.strip().lower() for e in os.getenv("COORDINATOR_EMAILS", "").split(",") if e.strip()}


def enabled():
    return bool(allowed() and os.getenv("FIREBASE_WEB_API_KEY") and project_id())


def _web_setting(name):
    """A field of the Firebase web config. FIREBASE_WEB_API_KEY may hold the bare apiKey or the whole
    firebaseConfig snippet copied from the Firebase console; both work."""
    raw = os.getenv("FIREBASE_WEB_API_KEY", "").strip()
    m = re.search(rf"""["']?{name}["']?\s*:\s*["']([^"']+)["']""", raw)
    if m:
        return m.group(1)
    return raw if name == "apiKey" and "{" not in raw else None


def web_config():
    """What the browser needs to start Firebase sign-in, or None when sign-in is off."""
    if not enabled():
        return None
    pid = project_id()
    return {"apiKey": _web_setting("apiKey"), "projectId": pid,
            "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN") or _web_setting("authDomain") or f"{pid}.firebaseapp.com"}


def _verify(token):
    from google.auth.transport.requests import Request
    from google.oauth2 import id_token

    return id_token.verify_firebase_token(token, Request(), audience=project_id())


def coordinator(authorization):
    """Email of the signed-in coordinator; None when sign-in is off (open console). Raises Denied."""
    if not enabled():
        return None
    token = authorization[7:].strip() if authorization.lower().startswith("bearer ") else ""
    if not token:
        raise Denied("Sign in as a coordinator to do this.")
    now = time.time()
    with _lock:
        hit = _verified.get(token)
    if hit and hit[1] > now:
        email = hit[0]
    else:
        try:
            claims = _verify(token)
        except Exception as e:
            raise Denied(f"Sign-in expired or invalid ({str(e)[:60]}). Sign in again.")
        email = str(claims.get("email", "")).lower()
        if not claims.get("email_verified"):
            raise Denied("This Google account's email is not verified.")
        with _lock:
            for t in [t for t, (_, exp) in _verified.items() if exp <= now]:
                del _verified[t]
            _verified[token] = (email, float(claims.get("exp", now + 300)))
    if email not in allowed():
        raise Denied(f"{email} is not on the coordinator list.")
    return email
