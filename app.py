"""Launcher: runs the app with app/pages/ (your existing UI)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", "--", *sys.argv[1:]],
        cwd=root,
    )
