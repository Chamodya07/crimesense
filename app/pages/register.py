from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Ensure project root is on path for backend imports
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import (
    AuthConfigError,
    LOGIN_PAGE,
    create_user,
    render_auth_status,
    require_auth,
    user_exists,
)
from services.firebase_service import FirebaseConfigError, init_firebase


ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


def inject_styles() -> None:
    if CSS_FILE.exists():
        st.markdown(f"<style>{CSS_FILE.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _firebase_ready() -> bool:
    try:
        init_firebase()
    except FirebaseConfigError as err:
        st.error(str(err))
        return False
    return True


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Register",
        layout="wide",
    )
    inject_styles()
    current = require_auth()

    st.title("Create User")
    render_auth_status("register")

    if current.role.strip().lower() != "admin":
        st.error("Only admin accounts can create users.")
        st.stop()

    if not _firebase_ready():
        st.info("Registration is unavailable until Firebase is configured.")
        st.stop()

    with st.form("register_form", clear_on_submit=True):
        username = st.text_input("Username")
        account_type = st.selectbox("Account Type", ["user", "admin"], index=0)
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Create User", use_container_width=True)

    if not submitted:
        return

    normalized_username = username.strip()
    if not normalized_username:
        st.error("Username is required.")
        return
    if len(password) < 8:
        st.error("Password must be at least 8 characters long.")
        return
    if password != confirm_password:
        st.error("Passwords do not match.")
        return

    try:
        if user_exists(normalized_username):
            st.error("Username already exists.")
            return
        create_user(normalized_username, password, role=account_type)
    except (AuthConfigError, FirebaseConfigError, ValueError) as err:
        st.error(str(err))
        return

    st.success(f"{account_type.title()} account created successfully.")
    st.info("The new account can now sign in from the login page.")
    if st.button("Go to Login", use_container_width=False):
        st.switch_page(LOGIN_PAGE)


if __name__ == "__main__":
    main()
