"""WSGI entry point for Parkinson's Disease Detection System.

Uses Waitress as the production WSGI server (works on Windows, Linux, and Mac).
Run with: python wsgi.py   or   waitress-serve --host=0.0.0.0 --port=8000 wsgi:app
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root so it works regardless of current working directory.
_project_root = Path(__file__).resolve().parent
load_dotenv(_project_root / ".env")

from webapp.app import create_app

# Create the Flask application instance
app = create_app()

if __name__ == "__main__":
    import waitress

    port = int(os.environ.get("PORT", 8001))
    print(f"Starting Waitress server at http://0.0.0.0:{port}")
    print("Press Ctrl+C to stop.")
    waitress.serve(app, host="0.0.0.0", port=port, threads=4)
