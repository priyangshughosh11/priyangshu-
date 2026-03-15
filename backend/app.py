"""Flask application entry point."""
from flask import Flask, jsonify
from flask_cors import CORS

from config import Config
from routes.stock_routes import stock_bp
from routes.prediction_routes import prediction_bp


def create_app(config_class=Config) -> Flask:
    """Application factory.

    Args:
        config_class: Configuration class with app settings.

    Returns:
        Configured Flask application instance.
    """
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Allow requests from the React dev-server (localhost:3000)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # Register blueprints
    app.register_blueprint(stock_bp)
    app.register_blueprint(prediction_bp)

    @app.route("/api/health")
    def health():
        """Health-check endpoint."""
        return jsonify({"status": "ok"})

    return app


if __name__ == "__main__":
    application = create_app()
    application.run(host="0.0.0.0", port=5000, debug=Config.DEBUG)
