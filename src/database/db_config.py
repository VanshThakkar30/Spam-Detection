import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

class Config:
    """Database and cache configuration."""
    # --- PostgreSQL Database Configuration ---
    # Example: 'postgresql://user:password@host:port/dbname'
    # For local development, we'll use a simple SQLite database for ease of use.
    # When deploying, you would change this to your PostgreSQL URI.
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL') or \
        'sqlite:///' + os.path.join(BASE_DIR, 'spam_app.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # --- Redis Cache Configuration ---
    # Example: 'redis://:password@host:port/0'
    REDIS_URL = os.environ.get('REDIS_URL') or 'redis://localhost:6379/0'