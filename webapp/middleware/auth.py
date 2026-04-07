"""JWT authentication middleware for protecting Flask routes."""

import os

import jwt
from flask import request, jsonify, g

from webapp.models.user import find_user_by_id


def _get_jwt_secret():
    """Return the JWT secret key from the environment."""
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    return secret


def _decode_token():
    """
    Extract and decode the JWT from the Authorization header.

    Returns:
        dict: The decoded token payload.

    Raises:
        ValueError: If the header is missing, malformed, or the token
            is invalid / expired.
    """
    auth_header = request.headers.get("Authorization", "")

    if not auth_header:
        raise ValueError("Authorization header is missing")

    parts = auth_header.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise ValueError("Authorization header must be: Bearer <token>")

    token = parts[1]

    try:
        payload = jwt.decode(token, _get_jwt_secret(), algorithms=["HS256"])
        return payload
    except jwt.ExpiredSignatureError:
        raise ValueError("Token has expired")
    except jwt.InvalidTokenError as exc:
        raise ValueError(f"Invalid token: {exc}")


def enforce_auth():
    """
    A ``before_request`` handler that protects **API** routes while
    leaving browser-navigated template pages and static assets public.

    With JWT-based auth the browser cannot attach an ``Authorization``
    header during normal page navigation, so only ``/api/*`` endpoints
    are gated (except the public ones listed below).

    Public (no token required):
        - All non-API routes (template pages, static files, favicon, etc.)
        - GET  /api/health
        - GET  /api/training_report
        - POST /api/auth/register
        - POST /api/auth/login
        - POST /api/auth/logout

    Protected (valid JWT required):
        - All other ``/api/*`` endpoints (detect, upload, model_info,
          process_combined_video, auth/me, etc.)
    """
    # Allow static file requests through.
    if request.endpoint and request.endpoint == "static":
        return None

    # Only enforce JWT on API routes.  Template pages served by the
    # browser (/, /detect, /about, etc.) are not API calls and
    # cannot carry an Authorization header.
    if not request.path.startswith("/api/"):
        return None

    # Public API endpoints that don't require a token.
    public_api_endpoints = {
        "detect.health_check",              # GET  /api/health
        "training_report.get_training_report",  # GET  /api/training_report
        "auth.register",                    # POST /api/auth/register
        "auth.login",                       # POST /api/auth/login
        "auth.logout",                      # POST /api/auth/logout
    }

    endpoint = request.endpoint
    if endpoint in public_api_endpoints:
        return None

    # --- Require authentication for all other API routes ---
    try:
        payload = _decode_token()
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 401

    user_id = payload.get("sub")
    if not user_id:
        return jsonify({"success": False, "error": "Invalid token payload"}), 401

    user = find_user_by_id(user_id)
    if user is None:
        return jsonify({"success": False, "error": "User not found"}), 401

    # Store the user for downstream handlers.
    g.current_user = user
    return None
