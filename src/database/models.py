from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

# Create a single SQLAlchemy instance
db = SQLAlchemy()

class PredictionLog(db.Model):
    """Table to store prediction logs."""
    id = db.Column(db.Integer, primary_key=True)
    message = db.Column(db.String, nullable=False)
    prediction = db.Column(db.String(50), nullable=False)
    confidence_spam = db.Column(db.Float, nullable=False)
    confidence_ham = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Log {self.id}: {self.prediction}>'

class UserFeedback(db.Model):
    """Table to store user feedback on predictions."""
    id = db.Column(db.Integer, primary_key=True)
    log_id = db.Column(db.Integer, db.ForeignKey('prediction_log.id'), nullable=False)
    is_correct = db.Column(db.Boolean, nullable=False) # True if user agrees, False if not
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Establish relationship to PredictionLog
    log = db.relationship('PredictionLog', backref=db.backref('feedback', lazy=True))