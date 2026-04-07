"""Public API: full training metrics + config feature summary (read-only, no secrets)."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Blueprint, jsonify

from webapp.services.training_report import load_training_report

logger = logging.getLogger(__name__)

training_report_bp = Blueprint("training_report", __name__)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@training_report_bp.route("/training_report", methods=["GET"])
def get_training_report():
    """
    JSON report: pipeline description, feature lists from config YAML,
    resolved CSV paths, and full ``dl_model_metrics.json`` when present.

    Public (no JWT) — no user data, only repository training artifacts.
    """
    try:
        payload = load_training_report(_project_root())
        return jsonify(payload), 200
    except Exception as e:
        logger.exception("training_report failed")
        return jsonify({"success": False, "error": str(e)}), 500
