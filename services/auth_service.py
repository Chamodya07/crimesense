from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass
from typing import Any

import streamlit as st


LOGIN_PAGE = "streamlit_app.py"
DEFAULT_PAGE = "pages/dashboard.py"
AUTH_USER_KEY = "auth_user"
AUTH_NOTICE_KEY = "auth_notice"


class AuthConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AuthUser:
    username: str
    display_name: str
    role: str


def _read_secret(container: Any, key: str, default: Any = None) -> Any:
    if container is None:
        return default
    try:
        return container[key]
    except Exception:
        return getattr(container, key, default)


def _configured_users() -> list[dict[str, str]]:
    auth_section = _read_secret(st.secrets, "auth")
    raw_users = _read_secret(auth_section, "users", [])
    users: list[dict[str, str]] = []

    for raw in raw_users or []:
        username = str(_read_secret(raw, "username", "")).strip()
        if not username:
            continue
        users.append(
            {
                "username": username,
                "display_name": str(
                    _read_secret(raw, "display_name", _read_secret(raw, "name", username))
                ).strip()
                or username,
                "role": str(_read_secret(raw, "role", "user")).strip() or "user",
                "password": str(_read_secret(raw, "password", "")).strip(),
                "password_sha256": str(_read_secret(raw, "password_sha256", "")).strip().lower(),
            }
        )

    if not users:
        raise AuthConfigError(
            "No auth users configured. Add [auth] users to .streamlit/secrets.toml."
        )

    return users


def auth_user_count() -> int:
    return len(_configured_users())


def _password_matches(candidate: str, configured: dict[str, str]) -> bool:
    plain = configured.get("password", "")
    hashed = configured.get("password_sha256", "")
    if hashed:
        candidate_hash = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        return hmac.compare_digest(candidate_hash, hashed)
    if plain:
        return hmac.compare_digest(candidate, plain)
    return False


def authenticate_user(username: str, password: str) -> AuthUser | None:
    normalized = username.strip().lower()
    if not normalized or not password:
        return None

    for user in _configured_users():
        if user["username"].strip().lower() != normalized:
            continue
        if _password_matches(password, user):
            return AuthUser(
                username=user["username"],
                display_name=user["display_name"],
                role=user["role"],
            )
    return None


def login_user(user: AuthUser) -> None:
    st.session_state[AUTH_USER_KEY] = {
        "username": user.username,
        "display_name": user.display_name,
        "role": user.role,
    }
    st.session_state.pop(AUTH_NOTICE_KEY, None)


def logout_user(notice: str = "Signed out.") -> None:
    st.session_state.pop(AUTH_USER_KEY, None)
    st.session_state[AUTH_NOTICE_KEY] = notice


def current_user() -> AuthUser | None:
    payload = st.session_state.get(AUTH_USER_KEY)
    if not isinstance(payload, dict):
        return None
    username = str(payload.get("username", "")).strip()
    if not username:
        return None
    return AuthUser(
        username=username,
        display_name=str(payload.get("display_name", username)).strip() or username,
        role=str(payload.get("role", "user")).strip() or "user",
    )


def is_authenticated() -> bool:
    return current_user() is not None


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
            st.switch_page(LOGIN_PAGE)
            st.stop()
    return user
