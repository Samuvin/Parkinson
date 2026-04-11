"""Authentication API endpoints: register, login, logout, and current user."""

import os
import re
from datetime import datetime, timezone, timedelta

import jwt
from flask import Blueprint, request, jsonify, g

from webapp.models.user import (
    create_user,
    find_user_by_email,
    verify_password,
)

auth_bp = Blueprint("auth", __name__)

# Minimum password length enforced on registration.
MIN_PASSWORD_LENGTH = 8

# Token lifetime (24 hours).
TOKEN_EXPIRY_HOURS = 24

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")


def _get_jwt_secret():
    """Return the JWT secret key from the environment."""
    secret = os.environ.get("JWT_SECRET_KEY")
    if not secret:
        raise RuntimeError(
            "JWT_SECRET_KEY environment variable is not set. "
            "Please add it to your .env file."
        )
    return secret


def _generate_token(user_id, email):
    """
    Generate a signed JWT for the given user.

    The token contains the user's id and email and expires after
    ``TOKEN_EXPIRY_HOURS`` hours.
    """
    payload = {
        "sub": str(user_id),
        "email": email,
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc) + timedelta(hours=TOKEN_EXPIRY_HOURS),
    }
    return jwt.encode(payload, _get_jwt_secret(), algorithm="HS256")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@auth_bp.route("/register", methods=["POST"])
def register():
    """
    Register a new user.

    Expects JSON:
        { "email": "...", "password": "..." }

    Returns 201 on success with the created user info.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    # --- validation ---
    if not email:
        return jsonify({"success": False, "error": "Email is required"}), 400

    if not _EMAIL_RE.match(email):
        return jsonify({"success": False, "error": "Invalid email format"}), 400

    if len(password) < MIN_PASSWORD_LENGTH:
        return jsonify({
            "success": False,
            "error": f"Password must be at least {MIN_PASSWORD_LENGTH} characters",
        }), 400

    # --- create user ---
    try:
        user = create_user(email, password)
    except ValueError as exc:
        return jsonify({"success": False, "error": str(exc)}), 409

    return jsonify({"success": True, "user": user}), 201


@auth_bp.route("/login", methods=["POST"])
def login():
    """
    Authenticate a user and return a JWT.

    Expects JSON:
        { "email": "...", "password": "..." }

    Returns 200 with an access token on success.
    """
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"success": False, "error": "Request body must be JSON"}), 400

    email = (data.get("email") or "").strip()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({
            "success": False,
            "error": "Email and password are required",
        }), 400

    user_doc = find_user_by_email(email)
    if user_doc is None or not verify_password(user_doc, password):
        return jsonify({
            "success": False,
            "error": "Invalid email or password",
        }), 401

    token = _generate_token(user_doc["_id"], user_doc["email"])

    return jsonify({
        "success": True,
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": TOKEN_EXPIRY_HOURS * 3600,
    }), 200


@auth_bp.route("/logout", methods=["POST"])
def logout():
    """
    Logout the current user.

    With pure JWT authentication the server does not maintain session state,
    so the client should simply discard the token.  This endpoint exists for
    API completeness and can be extended with a token-blacklist later.
    """
    return jsonify({"success": True, "message": "Logged out successfully"}), 200


@auth_bp.route("/me", methods=["GET"])
def me():
    """
    Return the profile of the currently authenticated user.

    The ``before_request`` hook in ``app.py`` will have already verified
    the token and placed the user in ``g.current_user``.
    """
    user = getattr(g, "current_user", None)
    if user is None:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    return jsonify({
        "success": True,
        "user": {
            "_id": str(user["_id"]),
            "email": user["email"],
            "created_at": user.get("created_at", "").isoformat()
            if hasattr(user.get("created_at", ""), "isoformat")
            else str(user.get("created_at", "")),
        },
    }), 200
