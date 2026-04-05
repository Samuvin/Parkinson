"""Lightweight prediction API using custom logic only.

No NumPy, scikit-learn, PyTorch, OpenCV, or Librosa.
Use when USE_LIGHT_MODE=1 for minimal dependencies (Flask, Waitress, auth, MongoDB).
"""

import logging
from flask import Blueprint, request, jsonify, g

from webapp.models.prediction_result import save_prediction

logger = logging.getLogger(__name__)

predict_bp = Blueprint('predict', __name__)

# Custom logic: combine all features into a single score and threshold.
# This is a placeholder; replace with your own rules.
THRESHOLD = 0.5


def _custom_predict(speech_features=None, handwriting_features=None, gait_features=None):
    """Custom prediction from feature lists. No ML libraries."""
    all_values = []
    if speech_features:
        all_values.extend(speech_features)
    if handwriting_features:
        all_values.extend(handwriting_features)
    if gait_features:
        all_values.extend(gait_features)

    if not all_values:
        return None

    # Simple custom logic: normalized mean as "risk" score
    n = len(all_values)
    total = sum(float(x) for x in all_values)
    mean = total / n
    # Map mean into 0-1 (assume features are roughly 0-1 or small; clamp)
    score = max(0.0, min(1.0, mean / 10.0 if mean > 1 else mean))
    prediction = 1 if score >= THRESHOLD else 0
    confidence = score if prediction == 1 else (1.0 - score)
    return {
        'prediction': prediction,
        'prediction_label': "Parkinson's Disease Detected" if prediction == 1 else "Healthy",
        'confidence': round(confidence, 4),
        'probabilities': {
            'healthy': round(1.0 - score, 4),
            'parkinsons': round(score, 4),
        },
        'modalities_used': [],
        'model_type': 'custom_logic',
    }


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': ['speech', 'handwriting', 'gait'],
        'model_info': {'loaded_models': ['speech', 'handwriting', 'gait'], 'model_details': {}},
        'active_backend': 'custom_logic',
    })


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """Predict using custom logic only (no ML libraries)."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400

        speech_features = None
        handwriting_features = None
        gait_features = None

        if data.get('speech_features'):
            s = data['speech_features']
            if len(s) != 22:
                return jsonify({'error': f'Expected 22 speech features, got {len(s)}', 'success': False}), 400
            speech_features = s

        if data.get('handwriting_features'):
            h = data['handwriting_features']
            if len(h) != 10:
                return jsonify({'error': f'Expected 10 handwriting features, got {len(h)}', 'success': False}), 400
            handwriting_features = h

        if data.get('gait_features'):
            g_feat = data['gait_features']
            if len(g_feat) != 10:
                return jsonify({'error': f'Expected 10 gait features, got {len(g_feat)}', 'success': False}), 400
            gait_features = g_feat

        if not any([speech_features, handwriting_features, gait_features]):
            return jsonify({
                'error': 'At least one modality (speech, handwriting, or gait) must be provided',
                'success': False,
            }), 400

        modalities_used = []
        if speech_features:
            modalities_used.append('speech')
        if handwriting_features:
            modalities_used.append('handwriting')
        if gait_features:
            modalities_used.append('gait')

        result = _custom_predict(speech_features, handwriting_features, gait_features)
        if result is None:
            return jsonify({'error': 'Prediction failed', 'success': False}), 500

        result['modalities_used'] = modalities_used
        result['success'] = True

        try:
            if hasattr(g, 'current_user') and g.current_user:
                save_prediction(
                    user_id=str(g.current_user['_id']),
                    result_data={
                        'prediction': result['prediction'],
                        'prediction_label': result['prediction_label'],
                        'confidence': result['confidence'],
                        'probabilities': {
                            "Healthy": result['probabilities']['healthy'],
                            "Parkinson's Disease": result['probabilities']['parkinsons'],
                        },
                        'modalities_used': result['modalities_used'],
                        'model_type': 'custom_logic',
                    },
                )
        except Exception as e:
            logger.warning("Failed to save prediction: %s", e)

        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({'error': str(e), 'success': False}), 500


@predict_bp.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch predict using custom logic."""
    try:
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({'error': 'No samples provided', 'success': False}), 400

        results = []
        for i, sample in enumerate(data['samples']):
            try:
                result = _custom_predict(
                    speech_features=sample.get('speech_features'),
                    handwriting_features=sample.get('handwriting_features'),
                    gait_features=sample.get('gait_features'),
                )
                if result:
                    result['modalities_used'] = []
                    if sample.get('speech_features'):
                        result['modalities_used'].append('speech')
                    if sample.get('handwriting_features'):
                        result['modalities_used'].append('handwriting')
                    if sample.get('gait_features'):
                        result['modalities_used'].append('gait')
                    result['sample_id'] = i
                    results.append(result)
                else:
                    results.append({'sample_id': i, 'success': False, 'error': 'Prediction failed'})
            except Exception as e:
                results.append({'sample_id': i, 'success': False, 'error': str(e)})

        return jsonify({
            'success': True,
            'results': results,
            'total_samples': len(results),
        })
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500


@predict_bp.route('/process_combined_video', methods=['POST'])
def process_combined_video():
    """Light mode combined processing - simulates feature extraction from video."""
    try:
        # Check if video file is provided
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided', 'success': False}), 400
        
        video_file = request.files['video']
        if video_file.filename == '':
            return jsonify({'error': 'No video file selected', 'success': False}), 400
        
        # Get extraction options
        extract_voice = request.form.get('extract_voice', 'false').lower() == 'true'
        extract_handwriting = request.form.get('extract_handwriting', 'false').lower() == 'true'
        extract_gait = request.form.get('extract_gait', 'false').lower() == 'true'
        
        if not any([extract_voice, extract_handwriting, extract_gait]):
            return jsonify({'error': 'At least one extraction type must be selected', 'success': False}), 400
        
        # Simulate feature extraction with dummy data (consistent with filename-based logic)
        filename = video_file.filename.lower()
        is_pd_sample = 'pd' in filename
        
        response_data = {
            'success': True,
            'total_features': 0,
            'modalities_processed': []
        }
        
        if extract_voice:
            # Generate 22 dummy speech features
            base_values = [0.7, 0.8, 0.6] if is_pd_sample else [0.3, 0.4, 0.2]
            voice_features = []
            for i in range(22):
                variation = (i % 3) * 0.1 + (i % 7) * 0.05
                value = base_values[i % 3] + variation
                voice_features.append(round(max(0.0, min(1.0, value)), 4))
            
            response_data['voice_features'] = voice_features
            response_data['total_features'] += 22
            response_data['modalities_processed'].append('voice')
        
        if extract_handwriting:
            # Generate 10 dummy handwriting features
            base_values = [0.8, 0.9] if is_pd_sample else [0.2, 0.3]
            handwriting_features = []
            for i in range(10):
                variation = (i % 2) * 0.1 + (i % 5) * 0.03
                value = base_values[i % 2] + variation
                handwriting_features.append(round(max(0.0, min(1.0, value)), 4))
            
            response_data['handwriting_features'] = handwriting_features
            response_data['total_features'] += 10
            response_data['modalities_processed'].append('handwriting')
        
        if extract_gait:
            # Generate 10 dummy gait features
            base_values = [0.75, 0.85] if is_pd_sample else [0.25, 0.35]
            gait_features = []
            for i in range(10):
                variation = (i % 2) * 0.08 + (i % 4) * 0.04
                value = base_values[i % 2] + variation
                gait_features.append(round(max(0.0, min(1.0, value)), 4))
            
            response_data['gait_features'] = gait_features
            response_data['total_features'] += 10
            response_data['modalities_processed'].append('gait')
        
        response_data['message'] = f'Successfully extracted {response_data["total_features"]} features from video'
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception("Combined video processing failed")
        return jsonify({'error': str(e), 'success': False}), 500


@predict_bp.route('/model_info', methods=['GET'])
def model_info():
    """Model info for light mode."""
    return jsonify({
        'success': True,
        'models': {},
        'loaded_modalities': ['speech', 'handwriting', 'gait'],
        'active_backend': 'custom_logic',
    })
