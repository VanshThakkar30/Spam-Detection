from flask import Flask
from .routes import api_blueprint
from src.database.db_config import Config
from src.database.models import db
import os


def create_app():
    """Creates and configures the Flask application."""
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static'),
        static_url_path='/static'
    )

    # --- Initialize Database ---
    app.config.from_object(Config)
    db.init_app(app)

    with app.app_context():
        # Create database tables if they don't exist
        db.create_all()

    app.register_blueprint(api_blueprint)
    return app