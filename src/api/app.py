from flask import Flask
from .routes import api_blueprint
from src.database.db_config import Config
from src.database.models import db
import os

def create_app():
    """
    Creates and configures the Flask application.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static')
    )
    
    # Load configuration and initialize the database
    app.config.from_object(Config)
    db.init_app(app)
    
    # Create database tables if they don't exist
    with app.app_context():
        db.create_all()

    # Register the blueprint that contains our API routes
    app.register_blueprint(api_blueprint)

    return app