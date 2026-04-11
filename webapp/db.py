"""MongoDB connection helper for the application."""

import os

import certifi
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

_client = None
_db = None


def get_db():
    """
    Return the MongoDB database instance, initialising the connection lazily.

    The connection URI is read from the MONGODB_URI environment variable.
    The database name can be overridden via MONGODB_DB_NAME (defaults to
    ``parkinsons_prediction``).

    Returns:
        pymongo.database.Database: The application database.

    Raises:
        RuntimeError: If MONGODB_URI is not set or the connection fails.
    """
    global _client, _db

    if _db is not None:
        return _db

    uri = os.environ.get("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI environment variable is not set. "
            "Please add it to your .env file."
        )

    db_name = os.environ.get("MONGODB_DB_NAME", "parkinsons_prediction")

    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000, tlsCAFile=certifi.where())
        # Verify that the server is reachable.
        _client.admin.command("ping")
        _db = _client[db_name]
        return _db
    except ConnectionFailure as exc:
        _client = None
        _db = None
        raise RuntimeError(
            f"Could not connect to MongoDB at {uri}: {exc}"
        ) from exc


def close_db():
    """Close the MongoDB client connection if it is open."""
    global _client, _db

    if _client is not None:
        _client.close()
        _client = None
        _db = None
