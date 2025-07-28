from src.api.app import create_app

# Create the Flask app instance
app = create_app()

if __name__ == '__main__':
    # The host='0.0.0.0' makes the API accessible from other devices on the same network.
    # The debug=True flag enables auto-reloading when you save changes.
    app.run(debug=True)