"""File upload API endpoints for audio, images, and video.

Handles file uploads and extracts features automatically.
"""

import logging
import os
import sys
import tempfile
from pathlib import Path

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.audio_processing import extract_speech_features, features_dict_to_array as audio_to_array
from common.image_processing import extract_handwriting_features, features_dict_to_array as image_to_array
from common.video_processing import extract_gait_features, features_dict_to_array as video_to_array

logger = logging.getLogger(__name__)

upload_bp = Blueprint('upload', __name__)

# Allowed file extensions
ALLOWED_AUDIO = {'wav', 'mp3', 'ogg', 'm4a', 'flac'}
ALLOWED_IMAGE = {'jpg', 'jpeg', 'png', 'bmp', 'tiff'}
ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def allowed_file(filename, allowed_extensions):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Magic-byte signatures for supported formats.
# Each entry maps to a list of (offset, bytes) tuples — all must match.
_MAGIC_SIGNATURES = {
    # Audio
    'wav':  [(0, b'RIFF'), (8, b'WAVE')],
    'mp3':  [(0, b'\xff\xfb'), (0, b'\xff\xf3'), (0, b'\xff\xf2'), (0, b'ID3')],
    'ogg':  [(0, b'OggS')],
    'flac': [(0, b'fLaC')],
    'm4a':  [(4, b'ftyp')],
    # Image
    'jpg':  [(0, b'\xff\xd8\xff')],
    'jpeg': [(0, b'\xff\xd8\xff')],
    'png':  [(0, b'\x89PNG\r\n\x1a\n')],
    'bmp':  [(0, b'BM')],
    'tiff': [(0, b'II*\x00'), (0, b'MM\x00*')],
    # Video
    'mp4':  [(4, b'ftyp')],
    'avi':  [(0, b'RIFF'), (8, b'AVI ')],
    'mov':  [(4, b'ftyp'), (4, b'moov'), (4, b'free'), (4, b'mdat')],
    'mkv':  [(0, b'\x1a\x45\xdf\xa3')],
}


def validate_magic_bytes(file_path: str, extension: str) -> bool:
    """Return True if the file's magic bytes match the expected format.

    For extensions with multiple possible signatures (e.g. MP3) any single
    matching signature is sufficient.  Returns True for unknown extensions so
    the function is non-breaking when a new extension is added without a
    corresponding signature entry.
    """
    ext = extension.lower().lstrip('.')
    signatures = _MAGIC_SIGNATURES.get(ext)
    if not signatures:
        return True

    try:
        with open(file_path, 'rb') as fh:
            header = fh.read(16)
    except OSError:
        return False

    # For formats with multiple alternative signatures (mp3, mov) treat each
    # tuple in the list as one candidate — return True if any candidate matches.
    if ext in ('mp3', 'mov'):
        for offset, magic in signatures:
            if len(header) >= offset + len(magic) and header[offset:offset + len(magic)] == magic:
                return True
        return False

    # For all other formats every (offset, magic) tuple must match.
    for offset, magic in signatures:
        if len(header) < offset + len(magic) or header[offset:offset + len(magic)] != magic:
            return False
    return True


@upload_bp.route('/audio', methods=['POST'])
def upload_audio():
    """Upload audio file and extract 22 speech features."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_AUDIO):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_AUDIO)}'
            }), 400
        
        filename = secure_filename(file.filename)
        
        # Validate file size before saving
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            return jsonify({
                'success': False,
                'error': 'The uploaded file is empty. Please upload a valid audio file.'
            }), 400
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return jsonify({
                'success': False,
                'error': f'File too large ({file_size / 1024 / 1024:.1f}MB). Maximum size is 100MB.'
            }), 400
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        ext = Path(filename).suffix.lstrip('.')
        if not validate_magic_bytes(tmp_path, ext):
            os.unlink(tmp_path)
            return jsonify({
                'success': False,
                'error': (
                    f'The uploaded file does not appear to be a valid {ext.upper()} file. '
                    'Please ensure the file is not renamed or corrupted and re-upload.'
                )
            }), 400

        try:
            logger.info("Extracting speech features from: %s (size: %d bytes)", tmp_path, file_size)
            features_dict = extract_speech_features(tmp_path)
            features_array = audio_to_array(features_dict)
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'speech',
                'message': 'Audio features extracted successfully'
            })
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except RuntimeError as e:
        error_msg = str(e)
        logger.error("Audio extraction failed: %s", error_msg)
        # Provide more specific error messages
        if 'format' in error_msg.lower() or 'not supported' in error_msg.lower() or 'corrupted' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'The audio file format is not supported or the file is corrupted. Please upload a valid WAV, MP3, OGG, FLAC, or M4A file.'
            }), 400
        elif 'empty' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'The uploaded file is empty. Please upload a valid audio file.'
            }), 400
        elif 'too large' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': error_msg
            }), 400
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to extract features from audio file. Please ensure the file is a valid audio format and contains speech data.'
            }), 400
    except Exception as e:
        logger.exception("Error processing audio")
        error_msg = str(e)
        # Check for timeout-related errors
        if 'timeout' in error_msg.lower() or 'worker' in error_msg.lower():
            return jsonify({
                'success': False,
                'error': 'Audio processing timed out. The file may be too large or in an unsupported format. Please try a smaller file or convert it to WAV format.'
            }), 500
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the audio file. Please ensure the file is a valid audio format and try again.'
        }), 500


@upload_bp.route('/handwriting', methods=['POST'])
def upload_handwriting():
    """Upload handwriting image and extract 10 features."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_IMAGE):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE)}'
            }), 400
        
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        ext = Path(filename).suffix.lstrip('.')
        if not validate_magic_bytes(tmp_path, ext):
            os.unlink(tmp_path)
            return jsonify({
                'success': False,
                'error': (
                    f'The uploaded file does not appear to be a valid {ext.upper()} file. '
                    'Please ensure the file is not renamed or corrupted and re-upload.'
                )
            }), 400

        try:
            features_dict = extract_handwriting_features(tmp_path)
            features_array = image_to_array(features_dict)
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'handwriting',
                'message': 'Handwriting features extracted successfully',
                'note': 'Optional features that enhance prediction accuracy when combined with speech'
            })
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except (ValueError, RuntimeError) as e:
        logger.error("Handwriting extraction failed: %s", str(e))
        return jsonify({
            'success': False,
            'error': 'Failed to extract features from handwriting image. Please ensure the image is clear and contains handwriting or drawing.'
        }), 400
    except Exception as e:
        logger.exception("Error processing handwriting")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the handwriting image. Please try again or contact support if the issue persists.'
        }), 500


@upload_bp.route('/gait', methods=['POST'])
def upload_gait():
    """Upload walking video and extract 10 gait features."""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename, ALLOWED_VIDEO):
            return jsonify({
                'success': False,
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO)}'
            }), 400
        
        filename = secure_filename(file.filename)
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(filename).suffix) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        ext = Path(filename).suffix.lstrip('.')
        if not validate_magic_bytes(tmp_path, ext):
            os.unlink(tmp_path)
            return jsonify({
                'success': False,
                'error': (
                    f'The uploaded file does not appear to be a valid {ext.upper()} file. '
                    'Please ensure the file is not renamed or corrupted and re-upload.'
                )
            }), 400

        try:
            features_dict = extract_gait_features(tmp_path)
            features_array = video_to_array(features_dict)
            os.unlink(tmp_path)
            
            return jsonify({
                'success': True,
                'features': features_array,
                'feature_count': len(features_array),
                'feature_names': list(features_dict.keys()),
                'modality': 'gait',
                'message': 'Gait features extracted successfully',
                'note': 'Optional features that enhance prediction accuracy when combined with speech'
            })
        except Exception as e:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise e
    
    except (ValueError, RuntimeError) as e:
        logger.error("Gait extraction failed: %s", str(e))
        return jsonify({
            'success': False,
            'error': 'Failed to extract features from gait video. Please ensure the video is valid and contains clear walking/gait movement.'
        }), 400
    except Exception as e:
        logger.exception("Error processing gait video")
        return jsonify({
            'success': False,
            'error': 'An error occurred while processing the gait video. Please try again or contact support if the issue persists.'
        }), 500


