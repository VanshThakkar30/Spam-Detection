
from flask import Blueprint, request, jsonify, render_template
from .predictor import predict_message
from src.database.models import db, UserFeedback, PredictionLog

# Create a Blueprint
api_blueprint = Blueprint('api', __name__)

# --- NEW: Add a route for the homepage ---
@api_blueprint.route('/')
def home():
    """Serves the frontend application."""
    return render_template('index.html')

@api_blueprint.route('/health', methods=['GET'])
def health_check():
    """A simple health check endpoint."""
    return jsonify({"status": "ok"}), 200

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    """Receives a message and returns a spam/ham prediction."""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "Invalid input: 'message' key not found"}), 400

    try:
        message = data['message']
        result = predict_message(message)
        return jsonify(result)
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        # Generic error handler
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@api_blueprint.route('/feedback', methods=['POST'])
def feedback():
    """Receives user feedback on a prediction."""
    data = request.get_json()
    log_id = data.get('log_id')
    is_correct = data.get('is_correct')

    if not log_id or is_correct is None:
        return jsonify({"error": "Missing log_id or is_correct flag"}), 400

    # Verify the prediction log exists
    log = PredictionLog.query.get(log_id)
    if not log:
        return jsonify({"error": "Prediction log not found"}), 404

    try:
        new_feedback = UserFeedback(log_id=log_id, is_correct=is_correct)
        db.session.add(new_feedback)
        db.session.commit()
        return jsonify({"status": "ok", "message": "Feedback received"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": f"Database error: {str(e)}"}), 500