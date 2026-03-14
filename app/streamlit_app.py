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
LOGIN_CSS_FILE = ASSETS_DIR / "login.css"


def _encode_image(path: Path) -> tuple[str, str]:
    """Return base64 string and mime type for the chosen image file."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return base64.b64encode(path.read_bytes()).decode(), mime


def inject_styles() -> None:
    """Load CSS and override login page backgrounds with the local asset if present."""
    css = CSS_FILE.read_text(encoding="utf-8") if CSS_FILE.exists() else ""
    login_css = LOGIN_CSS_FILE.read_text(encoding="utf-8") if LOGIN_CSS_FILE.exists() else ""

    preferred = ASSETS_DIR / "board.jpg"
    fallback = ASSETS_DIR / "board-placeholder.png"
    image_path = preferred if preferred.exists() else fallback

    encoded, mime = _encode_image(image_path)
    login_override = f":root {{ --login-bg-image: url('data:{mime};base64,{encoded}'); }}"

    st.markdown(f"<style>{css}\n{login_css}\n{login_override}</style>", unsafe_allow_html=True)


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


def render_login_card(notice: str | None, account_count: int | None, config_error: str | None) -> None:
    left, center, right = st.columns([0.8, 1.45, 0.8])
    with center:
        with st.form("login_panel_form", clear_on_submit=False):
            st.markdown(
                """
                <div class="login-head">
                    <div class="crime-title">Crime Sense</div>
                    <div class="crime-sub">AI-assisted offender profile predictions</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if notice:
                st.info(notice)
            if config_error:
                st.error(config_error)
            # elif account_count is not None:
            #     st.caption(f"{account_count} login account(s) configured for this app.")

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
    notice = pop_notice()
    account_count: int | None = None
    config_error: str | None = None
    try:
        account_count = auth_user_count()
    except AuthConfigError as err:
        config_error = str(err)
    render_login_card(notice, account_count, config_error)


if __name__ == "__main__":
    main()
