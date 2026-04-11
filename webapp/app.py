"""Flask web application for Parkinson's Disease detection."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify

from webapp.services.training_report import load_training_report
from flask_cors import CORS

from src.utils.config import Config
from webapp.api.detect import detect_bp, get_manager

from webapp.api.auth import auth_bp
from webapp.middleware.auth import enforce_auth


def create_app(config_path=None):
    """
    Create and configure the Flask application.

    Args:
        config_path: Path to configuration file

    Returns:
        Configured Flask application
    """
    app = Flask(__name__)

    Config(config_path)

    # Application configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for video uploads
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', '')
    app.config['MONGODB_URI'] = os.environ.get('MONGODB_URI', '')
    app.config['DEBUG'] = False
    app.config['TESTING'] = False

    # Enable CORS
    CORS(app)

    # Register blueprints
    app.register_blueprint(detect_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')

    from webapp.api.file_upload import upload_bp
    app.register_blueprint(upload_bp, url_prefix='/api/upload')
    from webapp.api.combined_processing import combined_bp
    app.register_blueprint(combined_bp, url_prefix='/api')

    try:
        manager = get_manager()
        loaded_models = manager.get_loaded_modalities()
    except Exception:
        pass

    from webapp.api.results import results_bp
    app.register_blueprint(results_bp, url_prefix='/api')

    from webapp.api.training_report import training_report_bp
    app.register_blueprint(training_report_bp, url_prefix='/api')

    # Enforce JWT authentication on all routes except public ones.
    app.before_request(enforce_auth)

    # Routes
    @app.route('/')
    def index():
        """Home page."""
        return render_template('index.html')

    @app.route('/login')
    def login_page():
        """Login / Register page."""
        return render_template('login.html')

    @app.route('/detect')
    def detect_page():
        """Detection page."""
        return render_template('detect.html')

    @app.route('/results')
    def results():
        """Results page."""
        return render_template('results.html')

    # Error handlers
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 errors."""
        return render_template('error.html', error_code=404, error_message='Page not found'), 404

    @app.errorhandler(500)
    def internal_error(error):
        """Handle 500 errors."""
        return render_template('error.html', error_code=500, error_message='Internal server error'), 500

    @app.errorhandler(413)
    def too_large(error):
        """Handle file too large errors."""
        return jsonify({'error': 'File too large', 'success': False}), 413

    return app


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Parkinson's Disease Detection System")
    print("="*60)
    print("\nRun the app with:  python wsgi.py")
    print("Or use the start script:  start.bat (Windows)  or  ./start.sh (Linux/Mac)")
    print("="*60 + "\n")
