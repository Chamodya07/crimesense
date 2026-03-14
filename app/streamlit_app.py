from __future__ import annotations

import base64
import sys
from pathlib import Path

import streamlit as st


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from services.auth_service import (
    DEFAULT_PAGE,
    AuthConfigError,
    authenticate_user,
    auth_user_count,
    is_authenticated,
    login_user,
    pop_notice,
)


ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
CSS_FILE = ASSETS_DIR / "styles.css"


def _encode_image(path: Path) -> tuple[str, str]:
    """Return base64 string and mime type for the chosen image file."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode(), mime


def inject_styles() -> None:
    """Load CSS and override the hero background with the local asset if present."""
    css = CSS_FILE.read_text(encoding="utf-8") if CSS_FILE.exists() else ""

    preferred = ASSETS_DIR / "board.jpg"
    fallback = ASSETS_DIR / "board-placeholder.png"
    image_path = preferred if preferred.exists() else fallback

    encoded, mime = _encode_image(image_path)
    hero_override = (
        ".crime-hero::before {"
        f"background-image: linear-gradient(180deg, rgba(10,0,0,0.65) 0%, rgba(0,0,0,0.92) 80%), "
        f"url('data:{mime};base64,{encoded}');"
        "}"
    )

    st.markdown(f"<style>{css}\n{hero_override}</style>", unsafe_allow_html=True)


def hide_sidebar_for_login() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="stSidebarNav"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_welcome() -> None:
    st.markdown(
        """
        <section class="crime-hero splash-card">
            <div>
                <div class="crime-title">Crime Sense</div>
                <div class="crime-sub">AI-assisted offender profile predictions</div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_login_card() -> None:
    st.markdown('<div id="login-card"></div>', unsafe_allow_html=True)
    with st.form("login_panel_form", clear_on_submit=False):
        username = st.text_input("", placeholder="username", label_visibility="collapsed", key="user_main")
        password = st.text_input("", placeholder="password", type="password", label_visibility="collapsed", key="pass_main")
        clicked = st.form_submit_button("Log In", use_container_width=True)

        if clicked:
            if not username or not password:
                st.error("Please enter username and password.")
                return

            try:
                user = authenticate_user(username, password)
            except AuthConfigError as err:
                st.error(str(err))
                return

            if user is None:
                st.error("Invalid username or password.")
                return

            login_user(user)
            st.switch_page(DEFAULT_PAGE)


def main() -> None:
    st.set_page_config(
        page_title="Crime Sense | Prototype",
        page_icon="🕵️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    if is_authenticated():
        st.switch_page(DEFAULT_PAGE)
        st.stop()

    inject_styles()
    hide_sidebar_for_login()
    render_welcome()
    st.markdown("<div class='login-wrapper-gap'></div>", unsafe_allow_html=True)
    notice = pop_notice()
    if notice:
        st.info(notice)
    try:
        st.caption(f"{auth_user_count()} login account(s) configured for this app.")
    except AuthConfigError as err:
        st.error(str(err))
    render_login_card()


if __name__ == "__main__":
    main()
