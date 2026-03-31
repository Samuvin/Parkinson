"""Flask web application for Parkinson's Disease detection."""

import sys
import os
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from flask import Flask, render_template, jsonify
from flask_cors import CORS

# Light mode: no NumPy/sklearn/PyTorch/OpenCV/Librosa. Use custom logic only.
USE_LIGHT_MODE = os.environ.get('USE_LIGHT_MODE', '').strip().lower() in ('1', 'true', 'yes')

if USE_LIGHT_MODE:
    from webapp.api.predict_light import predict_bp
else:
    from src.utils.config import Config
    from webapp.api.predict import predict_bp, get_manager

from webapp.api.auth import auth_bp
from webapp.middleware.auth import enforce_auth


def setup_logging(app):
    """Configure production logging."""
    if not app.debug:
        # Create logs directory
        logs_dir = project_root / 'logs'
        logs_dir.mkdir(exist_ok=True)
        
        # File handler
        file_handler = RotatingFileHandler(
            logs_dir / 'app.log',
            maxBytes=10240000,  # 10MB
            backupCount=10
        )
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
        ))
        file_handler.setLevel(logging.INFO)
        app.logger.addHandler(file_handler)
        
        app.logger.setLevel(logging.INFO)
        app.logger.info('Parkinson\'s Detection System startup')


def create_app(config_path=None):
    """
    Create and configure the Flask application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configured Flask application
    """
    app = Flask(__name__)
    
    if not USE_LIGHT_MODE:
        Config(config_path)
    
    # Application configuration
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for video uploads
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(24).hex())
    app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', '')
    app.config['MONGODB_URI'] = os.environ.get('MONGODB_URI', '')
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    
    # Setup logging
    setup_logging(app)
    
    if USE_LIGHT_MODE:
        app.logger.info("Running in LIGHT mode (custom logic only; no ML libraries)")
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(predict_bp, url_prefix='/api')
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    
    if not USE_LIGHT_MODE:
        # Upload and combined need OpenCV, Librosa, etc.
        from webapp.api.file_upload import upload_bp
        app.register_blueprint(upload_bp, url_prefix='/api/upload')
        from webapp.api.combined_processing import combined_bp
        app.register_blueprint(combined_bp, url_prefix='/api')
        
        # Load models on startup
        try:
            manager = get_manager()
            loaded_models = manager.get_loaded_modalities()
            app.logger.info("Models loaded on startup: %s", ', '.join(loaded_models))
        except Exception as e:
            app.logger.warning("Could not load models on startup: %s", e)
    
    # Results (MongoDB) works in both modes
    from webapp.api.results import results_bp
    app.register_blueprint(results_bp, url_prefix='/api')
    
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
    
    @app.route('/predict_page')
    def predict_page():
        """Detection page."""
        return render_template('predict.html')
    
    @app.route('/about')
    def about():
        """About page."""
        return render_template('about.html')
    
    @app.route('/results')
    def results():
        """Results page."""
        return render_template('results.html')
    
    @app.route('/model_images/<path:filename>')
    def model_images(filename):
        """Serve model performance images."""
        from flask import send_from_directory
        import os
        models_dir = os.path.join(app.root_path, '..', 'models')
        return send_from_directory(models_dir, filename)
    
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

