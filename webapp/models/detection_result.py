"""Persisted detection result operations (MongoDB).

Documents still use fields ``prediction`` and ``prediction_label`` and live in the
``predictions`` collection so existing databases keep working.
"""

from datetime import datetime, timezone
from typing import Optional, Dict, List, Any

from bson.objectid import ObjectId

from webapp.db import get_db

def _predictions_collection():
    """Return the ``predictions`` collection and ensure indexes exist."""
    db = get_db()
    collection = db["predictions"]
    collection.create_index("user_id")
    collection.create_index([("user_id", 1), ("created_at", -1)])
    return collection


def save_detection(user_id: str, result_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save a detection result to the database.

    Args:
        user_id: The user's MongoDB ObjectId (as string or ObjectId).
        result_data: Same shape as before (``prediction``, ``prediction_label``, etc.).

    Returns:
        dict: The saved document with _id.
    """
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    doc = {
        "user_id": user_id,
        "prediction": result_data.get("prediction", 0),
        "prediction_label": result_data.get("prediction_label", "Unknown"),
        "confidence": result_data.get("confidence", 0.0),
        "probabilities": result_data.get("probabilities", {}),
        "modalities_used": result_data.get("modalities_used", []),
        "model_type": result_data.get("model_type", "sklearn"),
        "created_at": datetime.now(timezone.utc),
    }

    result = _predictions_collection().insert_one(doc)
    doc["_id"] = result.inserted_id

    doc["_id"] = str(doc["_id"])
    doc["user_id"] = str(doc["user_id"])
    created_at_iso = doc["created_at"].isoformat()
    if not created_at_iso.endswith("Z") and not ("+" in created_at_iso or created_at_iso.count("-") > 2):
        created_at_iso = created_at_iso + "Z"
    doc["created_at"] = created_at_iso

    return doc


def find_by_user_id(user_id: str, limit: Optional[int] = None, skip: int = 0) -> List[Dict[str, Any]]:
    """List stored results for a user (newest first)."""
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    query = {"user_id": user_id}
    cursor = _predictions_collection().find(query).sort("created_at", -1)

    if skip > 0:
        cursor = cursor.skip(skip)
    if limit is not None:
        cursor = cursor.limit(limit)

    results = list(cursor)

    for doc in results:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            created_at_iso = doc["created_at"].isoformat()
            if not created_at_iso.endswith("Z") and not ("+" in created_at_iso or created_at_iso.count("-") > 2):
                created_at_iso = created_at_iso + "Z"
            doc["created_at"] = created_at_iso

    return results


def search_detections(user_id: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Search stored results with filters (same query shape as before)."""
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    query = {"user_id": user_id}

    date_filter = {}
    if filters.get("date_from"):
        try:
            date_from = datetime.fromisoformat(filters["date_from"].replace("Z", "+00:00"))
            date_filter["$gte"] = date_from
        except (ValueError, AttributeError):
            pass

    if filters.get("date_to"):
        try:
            date_to = datetime.fromisoformat(filters["date_to"].replace("Z", "+00:00"))
            date_filter["$lte"] = date_to
        except (ValueError, AttributeError):
            pass

    if date_filter:
        query["created_at"] = date_filter

    if filters.get("result"):
        query["prediction_label"] = filters["result"]

    confidence_filter = {}
    if filters.get("min_confidence") is not None:
        confidence_filter["$gte"] = float(filters["min_confidence"])
    if filters.get("max_confidence") is not None:
        confidence_filter["$lte"] = float(filters["max_confidence"])

    if confidence_filter:
        query["confidence"] = confidence_filter

    limit = filters.get("limit", 50)
    skip = filters.get("skip", 0)

    cursor = _predictions_collection().find(query).sort("created_at", -1)

    if skip > 0:
        cursor = cursor.skip(skip)
    if limit is not None:
        cursor = cursor.limit(limit)

    results = list(cursor)

    for doc in results:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()

    return results


def find_by_id(result_id: str, user_id: str) -> Optional[Dict[str, Any]]:
    """Find one result by id, scoped to user."""
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    try:
        result_id_obj = ObjectId(result_id)
    except Exception:
        return None

    doc = _predictions_collection().find_one({
        "_id": result_id_obj,
        "user_id": user_id
    })

    if doc:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
        if isinstance(doc.get("created_at"), datetime):
            created_at_iso = doc["created_at"].isoformat()
            if not created_at_iso.endswith("Z") and not ("+" in created_at_iso or created_at_iso.count("-") > 2):
                created_at_iso = created_at_iso + "Z"
            doc["created_at"] = created_at_iso

    return doc


def count_by_user_id(user_id: str) -> int:
    """Count stored results for a user."""
    if isinstance(user_id, str):
        user_id = ObjectId(user_id)

    return _predictions_collection().count_documents({"user_id": user_id})
