from flask import Flask
from .routes import api_blueprint
import os


def create_app():
    """
    Creates and configures the Flask application.
    """
    # --- Tell Flask where to find the templates and static folders ---
    # os.path.abspath finds the absolute path of the project
    # '..' moves up one directory from the current file's location (api -> src)
    # another '..' moves up again (src -> project root)
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    app = Flask(
        __name__,
        template_folder=os.path.join(project_root, 'templates'),
        static_folder=os.path.join(project_root, 'static'),
        static_url_path = '/static'
    )

    # Register the blueprint that contains our API routes
    app.register_blueprint(api_blueprint)

    return app