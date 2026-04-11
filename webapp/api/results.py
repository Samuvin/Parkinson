"""Results API endpoints for retrieving detection history."""

from flask import Blueprint, request, jsonify, g

from webapp.models.detection_result import (
    find_by_user_id,
    search_detections,
    find_by_id,
    count_by_user_id
)

results_bp = Blueprint('results', __name__)


@results_bp.route('/results', methods=['GET'])
def get_results():
    """
    Get user's detection results with optional filters.

    Query parameters:
        - limit: Maximum number of results (default: 50)
        - skip: Number of results to skip (default: 0)
        - date_from: ISO datetime string (inclusive)
        - date_to: ISO datetime string (inclusive)
        - result: "Healthy" or "Parkinson's Disease"
        - min_confidence: float (0.0-1.0)
        - max_confidence: float (0.0-1.0)

    Returns:
        JSON response with results list and metadata.
    """
    try:
        if not hasattr(g, 'current_user') or not g.current_user:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401

        user_id = str(g.current_user['_id'])

        # Build filters from query parameters
        filters = {
            'limit': request.args.get('limit', type=int, default=50),
            'skip': request.args.get('skip', type=int, default=0),
        }

        # Optional filters
        if request.args.get('date_from'):
            filters['date_from'] = request.args.get('date_from')
        if request.args.get('date_to'):
            filters['date_to'] = request.args.get('date_to')
        if request.args.get('result'):
            filters['result'] = request.args.get('result')
        if request.args.get('min_confidence') is not None:
            filters['min_confidence'] = request.args.get('min_confidence', type=float)
        if request.args.get('max_confidence') is not None:
            filters['max_confidence'] = request.args.get('max_confidence', type=float)

        results = search_detections(user_id, filters)
        total_count = count_by_user_id(user_id)

        return jsonify({
            'success': True,
            'data': {
                'results': results,
                'total': total_count,
                'returned': len(results),
                'skip': filters['skip'],
                'limit': filters['limit']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@results_bp.route('/results/<result_id>', methods=['GET'])
def get_result_by_id(result_id):
    """
    Get a specific stored result by ID.

    Args:
        result_id: MongoDB ObjectId of the result.

    Returns:
        JSON response with the result document.
    """
    try:
        if not hasattr(g, 'current_user') or not g.current_user:
            return jsonify({
                'success': False,
                'error': 'Authentication required'
            }), 401

        user_id = str(g.current_user['_id'])
        result = find_by_id(result_id, user_id)

        if not result:
            return jsonify({
                'success': False,
                'error': 'Result not found'
            }), 404

        return jsonify({
            'success': True,
            'data': result
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
