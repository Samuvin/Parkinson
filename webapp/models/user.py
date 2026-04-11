"""User model operations for MongoDB."""

from datetime import datetime, timezone

import bcrypt
from bson.objectid import ObjectId
from pymongo.errors import DuplicateKeyError

from webapp.db import get_db

def _users_collection():
    """Return the ``users`` collection and ensure indexes exist."""
    db = get_db()
    collection = db["users"]
    # Ensure a unique index on the email field (idempotent call).
    collection.create_index("email", unique=True)
    return collection


def create_user(email, password):
    """
    Create a new user with a hashed password.

    Args:
        email: The user's email address (must be unique).
        password: The plaintext password to hash.

    Returns:
        dict: The created user document (without the password hash).

    Raises:
        ValueError: If a user with the given email already exists.
    """
    email = email.strip().lower()
    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    user_doc = {
        "email": email,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
    }

    try:
        result = _users_collection().insert_one(user_doc)
        user_doc["_id"] = result.inserted_id
    except DuplicateKeyError:
        raise ValueError(f"A user with email '{email}' already exists")

    # Return a safe copy without the hash.
    return {
        "_id": str(user_doc["_id"]),
        "email": user_doc["email"],
        "created_at": user_doc["created_at"].isoformat(),
    }


def find_user_by_email(email):
    """
    Find a user by email address.

    Args:
        email: The email to look up (case-insensitive).

    Returns:
        dict or None: The full user document, or ``None`` if not found.
    """
    return _users_collection().find_one({"email": email.strip().lower()})


def find_user_by_id(user_id):
    """
    Find a user by their MongoDB ObjectId.

    Args:
        user_id: A string or ``ObjectId`` representing the user's ``_id``.

    Returns:
        dict or None: The full user document, or ``None`` if not found.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)
    return _users_collection().find_one({"_id": user_id})


def verify_password(user_doc, password):
    """
    Verify a plaintext password against the stored hash.

    Args:
        user_doc: The user document containing ``password_hash``.
        password: The plaintext password to check.

    Returns:
        bool: ``True`` if the password matches, ``False`` otherwise.
    """
    stored_hash = user_doc.get("password_hash", "")
    return bcrypt.checkpw(
        password.encode("utf-8"), stored_hash.encode("utf-8")
    )
