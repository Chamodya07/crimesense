from __future__ import annotations

from dataclasses import dataclass

import streamlit as st


LOGIN_PAGE = "streamlit_app.py"
DEFAULT_PAGE = "pages/dashboard.py"
AUTH_USER_KEY = "auth_user"
AUTH_NOTICE_KEY = "auth_notice"
AUTH_FLAG_KEY = "authenticated"
AUTH_USERNAME_KEY = "user"


class AuthConfigError(Exception):
    pass


@dataclass(frozen=True)
class AuthUser:
    username: str
    display_name: str
    role: str


def _init_auth_state() -> None:
    if AUTH_FLAG_KEY not in st.session_state:
        st.session_state[AUTH_FLAG_KEY] = False
    if AUTH_USERNAME_KEY not in st.session_state:
        st.session_state[AUTH_USERNAME_KEY] = None


def _normalize_username(username: str) -> str:
    return str(username or "").strip().lower()


def _normalize_role(role: str) -> str:
    normalized = str(role or "").strip().lower()
    if normalized not in {"user", "admin"}:
        raise ValueError("Account type must be either 'user' or 'admin'.")
    return normalized


def _load_bcrypt():
    try:
        import bcrypt
    except ImportError as exc:
        raise AuthConfigError(
            "bcrypt is not installed. Add it to requirements and install dependencies."
        ) from exc
    return bcrypt


def _load_firestore():
    from services.firebase_service import FirebaseConfigError, init_firebase

    try:
        return init_firebase()
    except FirebaseConfigError as exc:
        raise AuthConfigError(str(exc)) from exc


def _get_firestore_user(username: str) -> dict | None:
    document_id = str(username or "").strip()
    if not document_id:
        return None

    db = _load_firestore()
    snapshot = db.collection("users").document(document_id).get()
    if not snapshot.exists:
        return None
    return {"id": snapshot.id, **(snapshot.to_dict() or {})}


def auth_user_count() -> int:
    try:
        user_count = sum(1 for _ in _load_firestore().collection("users").stream())
    except AuthConfigError as exc:
        raise AuthConfigError(str(exc)) from exc

    if user_count == 0:
        raise AuthConfigError("No auth users found in Firestore collection 'users'.")
    return user_count


def _password_matches_hash(candidate: str, password_hash: str) -> bool:
    return verify_password(candidate, password_hash)


def hash_password(pw: str) -> str:
    if not pw:
        raise ValueError("Password is required.")

    bcrypt = _load_bcrypt()
    return bcrypt.hashpw(pw.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(pw: str, password_hash: str) -> bool:
    if not pw or not password_hash:
        return False

    bcrypt = _load_bcrypt()
    try:
        return bool(bcrypt.checkpw(pw.encode("utf-8"), password_hash.encode("utf-8")))
    except ValueError:
        return False


def authenticate_user(username: str, password: str) -> AuthUser | None:
    document_id = str(username or "").strip()
    if not document_id or not password:
        return None

    firestore_user = _get_firestore_user(document_id)
    if firestore_user is None:
        return None

    if _password_matches_hash(password, str(firestore_user.get("password_hash", "")).strip()):
        stored_username = str(firestore_user.get("username") or firestore_user.get("id") or "").strip()
        return AuthUser(
            username=stored_username or str(username).strip(),
            display_name=stored_username or str(username).strip(),
            role=str(firestore_user.get("role", "user")).strip() or "user",
        )
    return None


def user_exists(username: str) -> bool:
    document_id = str(username or "").strip()
    if not document_id:
        return False

    db = _load_firestore()
    return bool(db.collection("users").document(document_id).get().exists)


def create_user(username: str, password: str, role: str = "user") -> str:
    document_id = str(username or "").strip()
    if not document_id:
        raise ValueError("Username is required.")
    if len(password or "") < 8:
        raise ValueError("Password must be at least 8 characters long.")
    if user_exists(document_id):
        raise ValueError("User already exists")
    account_role = _normalize_role(role)

    db = _load_firestore()

    try:
        from firebase_admin import firestore
    except ImportError as exc:
        raise AuthConfigError(
            "firebase-admin is not installed. Add it to requirements and install dependencies."
        ) from exc

    password_hash = hash_password(password)
    payload = {
        "username": document_id,
        "password_hash": password_hash,
        "created_at": firestore.SERVER_TIMESTAMP,
        "role": account_role,
    }

    try:
        db.collection("users").document(document_id).create(payload)
    except Exception as exc:
        if user_exists(document_id):
            raise ValueError("User already exists") from exc
        raise
    return document_id


def verify_user(username: str, password: str) -> bool:
    return authenticate_user(username, password) is not None


def login_user(user: AuthUser) -> None:
    _init_auth_state()
    st.session_state[AUTH_USER_KEY] = {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }
    st.session_state[AUTH_FLAG_KEY] = True
    st.session_state[AUTH_USERNAME_KEY] = user.username
    st.session_state.pop(AUTH_NOTICE_KEY, None)
    try:
        from services.audit_service import log_event

        log_event("login", page="auth")
    except Exception:
        pass


def logout_user(notice: str = "Signed out.") -> None:
    try:
        from services.audit_service import log_event

        log_event("logout", page="auth")
    except Exception:
        pass

    _init_auth_state()
    st.session_state.pop(AUTH_USER_KEY, None)
    st.session_state[AUTH_FLAG_KEY] = False
    st.session_state[AUTH_USERNAME_KEY] = None
    st.session_state[AUTH_NOTICE_KEY] = notice
    st.rerun()


def current_user() -> AuthUser | None:
    _init_auth_state()
    if not st.session_state.get(AUTH_FLAG_KEY):
        return None

    payload = st.session_state.get(AUTH_USER_KEY)
    if not isinstance(payload, dict):
        return None

    username = str(payload.get("username", "")).strip()
    if not username:
        return None

    st.session_state[AUTH_USERNAME_KEY] = username
    return AuthUser(
        username=username,
        display_name=str(payload.get("display_name", username)).strip() or username,
        role=str(payload.get("role", "user")).strip() or "user",
    )


def is_authenticated() -> bool:
    _init_auth_state()
    return bool(st.session_state.get(AUTH_FLAG_KEY)) and current_user() is not None


def pop_notice() -> str | None:
    notice = st.session_state.get(AUTH_NOTICE_KEY)
    if notice:
        st.session_state.pop(AUTH_NOTICE_KEY, None)
        return str(notice)
    return None


def require_auth() -> AuthUser:
    user = current_user()
    if user is not None:
        return user
    st.session_state[AUTH_NOTICE_KEY] = "Please sign in to continue."
    st.switch_page(LOGIN_PAGE)
    st.stop()


def render_auth_status(key_suffix: str) -> AuthUser:
    user = require_auth()
    info_col, action_col = st.columns([6, 1])
    with info_col:
        st.caption(f"Signed in as {user.display_name} ({user.role})")
    with action_col:
        if st.button("Logout", key=f"logout_{key_suffix}", use_container_width=True):
            logout_user()
            st.stop()
    return user


__all__ = [
    "AuthConfigError",
    "AuthUser",
    "LOGIN_PAGE",
    "DEFAULT_PAGE",
    "AUTH_USER_KEY",
    "AUTH_NOTICE_KEY",
    "auth_user_count",
    "authenticate_user",
    "hash_password",
    "verify_password",
    "user_exists",
    "create_user",
    "verify_user",
    "login_user",
    "logout_user",
    "current_user",
    "is_authenticated",
    "pop_notice",
    "require_auth",
    "render_auth_status",
]


if __name__ == "__main__":
    print("exports:", "create_user" in globals(), "user_exists" in globals())
