
from flask import Blueprint, request, jsonify, render_template
from .predictor import predict_message

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