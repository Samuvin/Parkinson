"""API endpoint for processing combined video with multiple modalities."""

import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from flask import Blueprint, request, jsonify

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from utils.audio_processing import extract_speech_features
from utils.video_processing import extract_gait_features
from utils.image_processing import extract_handwriting_features

logger = logging.getLogger(__name__)

combined_bp = Blueprint('combined', __name__)


@combined_bp.route('/process_combined_video', methods=['POST'])
def process_combined_video():
    """Process uploaded video file and extract selected modalities.

    Expected: multipart/form-data with:
      - ``video`` file
      - ``extract_voice`` (boolean)
      - ``extract_handwriting`` (boolean)
      - ``extract_gait`` (boolean)

    Returns JSON with extracted features for selected modalities.
    """
    try:
        if 'video' not in request.files:
            return jsonify({
                'error': 'No video file provided',
                'success': False
            }), 400
        
        video_file = request.files['video']
        
        if video_file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'success': False
            }), 400
        
        extract_voice = request.form.get('extract_voice', 'false').lower() == 'true'
        extract_handwriting = request.form.get('extract_handwriting', 'false').lower() == 'true'
        extract_gait = request.form.get('extract_gait', 'false').lower() == 'true'
        
        logger.info(
            "Combined video processing: file=%s, voice=%s, handwriting=%s, gait=%s",
            video_file.filename, extract_voice, extract_handwriting, extract_gait,
        )
        
        # Check if this is a demo/dummy file (no real files uploaded)
        is_demo_file = (video_file.filename.startswith('demo_') or 
                       video_file.content_length < 100)  # Very small file indicates dummy content
        
        if is_demo_file:
            # Handle demo mode with mock feature generation
            logger.info("Processing demo file, using mock feature generation")
            return _process_demo_combined(video_file.filename, extract_voice, extract_handwriting, extract_gait)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            video_file.save(tmp_file.name)
            tmp_path = tmp_file.name
        
        try:
            response_data = {
                'success': True,
                'voice_features': None,
                'handwriting_features': None,
                'gait_features': None,
                'total_features': 0
            }
            
            if extract_voice:
                try:
                    # Extract audio from video and get speech features
                    audio_path = _extract_audio_from_video(tmp_path)
                    if audio_path:
                        features_dict = extract_speech_features(audio_path)
                        from utils.audio_processing import features_dict_to_array
                        voice_features = features_dict_to_array(features_dict)
                        response_data['voice_features'] = voice_features.tolist()
                        response_data['total_features'] += len(voice_features)
                        logger.info("Extracted %d voice features from video", len(voice_features))
                        # Clean up temporary audio file
                        if os.path.exists(audio_path):
                            os.unlink(audio_path)
                    else:
                        logger.warning("Could not extract audio from video")
                except (RuntimeError, ValueError) as e:
                    logger.error("Error extracting voice from video: %s", e)
                    # Don't fail entire request, just skip voice features
                except Exception as e:
                    logger.warning("Unexpected error extracting voice: %s", e)
            
            if extract_handwriting:
                try:
                    # Extract a frame from video and analyze for handwriting features
                    # Note: Handwriting analysis from video frames is limited
                    # Real handwriting analysis requires pen digitizer data
                    frame_path = _extract_frame_from_video(tmp_path)
                    if frame_path:
                        features_dict = extract_handwriting_features(frame_path)
                        from utils.image_processing import features_dict_to_array
                        handwriting_features = features_dict_to_array(features_dict)
                        response_data['handwriting_features'] = handwriting_features.tolist()
                        response_data['total_features'] += len(handwriting_features)
                        logger.info("Extracted %d handwriting features from video frame", len(handwriting_features))
                        # Clean up temporary frame
                        if os.path.exists(frame_path):
                            os.unlink(frame_path)
                    else:
                        logger.warning("Could not extract frame from video for handwriting analysis")
                except (RuntimeError, ValueError) as e:
                    logger.error("Error extracting handwriting from video: %s", e)
                    # Don't fail entire request, just skip handwriting features
                except Exception as e:
                    logger.warning("Unexpected error extracting handwriting: %s", e)
            
            if extract_gait:
                try:
                    # Extract gait features from video (real extraction)
                    features_dict = extract_gait_features(tmp_path)
                    from utils.video_processing import features_dict_to_array
                    gait_features = features_dict_to_array(features_dict)
                    response_data['gait_features'] = gait_features.tolist()
                    response_data['total_features'] += len(gait_features)
                    logger.info("Extracted %d gait features from video", len(gait_features))
                except (RuntimeError, ValueError) as e:
                    logger.error("Error extracting gait from video: %s", e)
                    # Don't fail entire request, just skip gait features
                except Exception as e:
                    logger.warning("Unexpected error extracting gait: %s", e)
            
            if response_data['total_features'] == 0:
                return jsonify({
                    'error': 'No features could be extracted from the video',
                    'success': False,
                    'note': 'Please ensure the video contains the selected assessment types'
                }), 400
            
            logger.info("Total features extracted: %d", response_data['total_features'])
            return jsonify(response_data)
        
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    except Exception as e:
        logger.exception("Combined video processing failed")
        return jsonify({
            'error': 'An error occurred while processing the video. Please ensure the file is valid and try again.',
            'success': False
        }), 500


def _extract_frame_from_video(video_path: str) -> str:
    """Extract a frame from video file for handwriting analysis.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to temporary image file, or None if extraction fails
    """
    try:
        import cv2
        import tempfile
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None
        
        # Get middle frame (more likely to have content)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = total_frames // 2 if total_frames > 0 else 0
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret or frame is None:
            return None
        
        # Save frame as temporary image
        frame_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
        frame_path = frame_file.name
        frame_file.close()
        
        cv2.imwrite(frame_path, frame)
        
        if os.path.exists(frame_path):
            logger.info("Successfully extracted frame from video")
            return frame_path
        return None
        
    except Exception as e:
        logger.warning("Error extracting frame from video: %s", e)
        if 'frame_path' in locals() and os.path.exists(frame_path):
            os.unlink(frame_path)
        return None


def _extract_audio_from_video(video_path: str) -> str:
    """Extract audio track from video file and save as temporary WAV file.
    
    Args:
        video_path: Path to video file
        
    Returns:
        Path to temporary audio file, or None if extraction fails
    """
    try:
        import subprocess
        import tempfile
        
        # Create temporary audio file
        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        audio_path = audio_file.name
        audio_file.close()
        
        # Use ffmpeg to extract audio
        # ffmpeg -i video.mp4 -vn -acodec pcm_s16le -ar 44100 -ac 1 audio.wav
        cmd = [
            'ffmpeg',
            '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # PCM 16-bit little-endian
            '-ar', '44100',  # Sample rate
            '-ac', '1',  # Mono
            '-y',  # Overwrite output file
            audio_path
        ]
        
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30
        )
        
        if result.returncode == 0 and os.path.exists(audio_path):
            logger.info("Successfully extracted audio from video")
            return audio_path
        else:
            logger.warning("ffmpeg failed to extract audio: %s", result.stderr.decode())
            if os.path.exists(audio_path):
                os.unlink(audio_path)
            return None
            
    except FileNotFoundError:
        logger.warning("ffmpeg not found. Cannot extract audio from video.")
        return None
    except Exception as e:
        logger.warning("Error extracting audio from video: %s", e)
        if 'audio_path' in locals() and os.path.exists(audio_path):
            os.unlink(audio_path)
        return None


def _process_demo_combined(filename, extract_voice, extract_handwriting, extract_gait):
    """Process demo/dummy files with mock feature generation.
    
    This mimics the light mode behavior for when no real files are uploaded
    but the system is running in full mode.
    """
    try:
        # Simulate feature extraction with dummy data (consistent with filename-based logic)
        filename_lower = filename.lower()
        is_pd_sample = 'pd' in filename_lower
        
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
            logger.info("Generated 22 demo voice features")
        
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
            logger.info("Generated 10 demo handwriting features")
        
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
            logger.info("Generated 10 demo gait features")
        
        response_data['message'] = f'Successfully extracted {response_data["total_features"]} features from demo video'
        logger.info("Demo combined processing complete: %d total features", response_data['total_features'])
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.exception("Demo combined processing failed")
        return jsonify({'error': str(e), 'success': False}), 500
