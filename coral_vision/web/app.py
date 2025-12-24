"""Flask web application factory for Coral Vision."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

try:
    import eventlet  # noqa: F401
except ImportError:
    eventlet = None  # type: ignore

try:
    import gevent  # noqa: F401
except ImportError:
    gevent = None  # type: ignore

from flask import Flask, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO

from coral_vision.config import Paths, get_data_dir_from_env
from coral_vision.core.edgetpu import check_edgetpu_status
from coral_vision.core.logger import get_logger
from coral_vision.core.storage_pgvector import get_storage_backend_from_env
from coral_vision.web.api import register_api_routes

logger = get_logger("web")

# Global SocketIO instance (will be initialized in create_app)
socketio: Optional[SocketIO] = None


def create_app(
    data_dir: Optional[Path] = None,
    use_edgetpu: bool = False,
) -> Flask:
    """Create and configure the Flask application.

    Args:
        data_dir: Path to data directory containing models.
        use_edgetpu: Whether to use Edge TPU acceleration.

    Returns:
        Configured Flask application instance.
    """
    if data_dir is None:
        data_dir = get_data_dir_from_env()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size
    app.config["DATA_DIR"] = data_dir

    paths = Paths(data_dir=data_dir)

    # Verify Edge TPU availability if requested
    if use_edgetpu:
        check_edgetpu_status(use_edgetpu, paths.models_dir)

    app.config["USE_EDGETPU"] = use_edgetpu

    # Initialize pgvector storage backend
    storage = get_storage_backend_from_env()

    # Define allowed file extensions
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "webp"}

    # Get API key from environment (required)
    api_key = os.getenv("API_KEY")
    if not api_key:
        raise ValueError(
            "API_KEY environment variable is required. "
            "Set it in docker-compose.yml or as an environment variable."
        )
    app.config["API_KEY"] = api_key
    logger.info("API key authentication enabled")

    # Initialize rate limiter
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri=os.getenv(
            "REDIS_URL"
        ),  # Optional Redis for distributed rate limiting
    )
    app.config["LIMITER"] = limiter

    # Initialize SocketIO
    # Try to use eventlet for better WebSocket support, fallback to threading
    global socketio

    # Get allowed origins from environment or default to localhost
    # For production, set ALLOWED_ORIGINS env var with comma-separated origins
    # Example: ALLOWED_ORIGINS=https://coral-vision.barfamily.co.il,https://www.example.com
    allowed_origins_str = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"
    )
    allowed_origins = [
        origin.strip() for origin in allowed_origins_str.split(",") if origin.strip()
    ]

    # If ALLOWED_ORIGINS is set to "*", allow all origins (not recommended for production)
    if allowed_origins == ["*"]:
        allowed_origins = "*"
        logger.warning(
            "ALLOWED_ORIGINS is set to '*' - allowing all origins. Not recommended for production!"
        )

    if eventlet is not None:
        async_mode = "eventlet"
    elif gevent is not None:
        async_mode = "gevent"
    else:
        async_mode = "threading"

    socketio = SocketIO(
        app,
        cors_allowed_origins=allowed_origins,
        async_mode=async_mode,
        logger=False,
        engineio_logger=False,
    )
    logger.info(
        f"SocketIO initialized with async_mode={async_mode}, allowed_origins={allowed_origins}"
    )

    # Register API routes (including WebSocket handlers)
    register_api_routes(
        app=app,
        socketio=socketio,
        paths=paths,
        storage=storage,
        use_edgetpu=app.config["USE_EDGETPU"],
        allowed_extensions=ALLOWED_EXTENSIONS,
        api_key=api_key,
        limiter=limiter,
    )

    # Register template routes
    @app.route("/")
    def index() -> str:
        """Homepage with product information and getting started guide."""
        return render_template(
            "index.html",
            use_edgetpu=app.config["USE_EDGETPU"],
        )

    @app.route("/docs")
    def swagger_ui() -> str:
        """Serve Swagger UI for API documentation."""
        return render_template("docs.html")

    @app.route("/openapi.json")
    def openapi_spec() -> Any:
        """Serve the OpenAPI specification."""
        openapi_path = Path(__file__).parent.parent.parent / "openapi.json"
        if openapi_path.exists():
            return jsonify(json.loads(openapi_path.read_text()))
        return jsonify({"error": "OpenAPI spec not found"}), 404

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict[str, Any], int]:
        """Health check endpoint with dependency checks."""
        from coral_vision.core.edgetpu import verify_edgetpu_availability

        checks: dict[str, Any] = {}
        all_healthy = True

        # Check database connection
        try:
            storage.load_people_index()  # Simple query to verify connection
            checks["database"] = {"status": "healthy", "message": "Connected"}
        except Exception as e:
            logger.error(f"Database health check failed: {e}", exc_info=True)
            checks["database"] = {"status": "unhealthy", "message": str(e)}
            all_healthy = False

        # Check Edge TPU status
        if app.config["USE_EDGETPU"]:
            try:
                is_available, warning = verify_edgetpu_availability(paths.models_dir)
                if is_available and not warning:
                    checks["edgetpu"] = {"status": "healthy", "message": "Available"}
                elif is_available and warning:
                    checks["edgetpu"] = {"status": "degraded", "message": warning}
                    all_healthy = False
                else:
                    checks["edgetpu"] = {
                        "status": "unhealthy",
                        "message": warning or "Not available",
                    }
                    all_healthy = False
            except Exception as e:
                logger.error(f"Edge TPU health check failed: {e}", exc_info=True)
                checks["edgetpu"] = {"status": "unhealthy", "message": str(e)}
                all_healthy = False
        else:
            checks["edgetpu"] = {
                "status": "disabled",
                "message": "Edge TPU not enabled",
            }

        # Check model files exist
        try:
            from coral_vision.config import resolve_model_paths

            model_paths = resolve_model_paths(paths)
            # Check both CPU and Edge TPU models (at least one set should exist)
            detector_ok = model_paths.detector_cpu.exists() or (
                app.config["USE_EDGETPU"] and model_paths.detector_edgetpu.exists()
            )
            embedder_ok = model_paths.embedder_cpu.exists() or (
                app.config["USE_EDGETPU"] and model_paths.embedder_edgetpu.exists()
            )

            models_ok = detector_ok and embedder_ok
            if models_ok:
                checks["models"] = {
                    "status": "healthy",
                    "message": "All required models found",
                    "detector_cpu": str(model_paths.detector_cpu),
                    "detector_edgetpu": str(model_paths.detector_edgetpu),
                    "embedder_cpu": str(model_paths.embedder_cpu),
                    "embedder_edgetpu": str(model_paths.embedder_edgetpu),
                }
            else:
                missing = []
                if not detector_ok:
                    missing.append("detection")
                if not embedder_ok:
                    missing.append("embedding")
                checks["models"] = {
                    "status": "unhealthy",
                    "message": f"Missing models: {', '.join(missing)}",
                }
                all_healthy = False
        except Exception as e:
            logger.error(f"Models health check failed: {e}", exc_info=True)
            checks["models"] = {"status": "unhealthy", "message": str(e)}
            all_healthy = False

        # Determine overall status
        if all_healthy:
            status = "healthy"
            http_status = 200
        else:
            # Check if critical components are down
            if checks.get("database", {}).get("status") == "unhealthy":
                status = "unhealthy"
                http_status = 503  # Service Unavailable
            else:
                status = "degraded"
                http_status = 200  # Still operational but degraded

        return (
            jsonify(
                {
                    "status": status,
                    "use_edgetpu": app.config["USE_EDGETPU"],
                    "storage": "pgvector",
                    "checks": checks,
                }
            ),
            http_status,
        )

    # Register error handlers
    @app.errorhandler(404)
    def not_found(error: Any) -> tuple[dict[str, str], int]:
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(413)
    def request_entity_too_large(error: Any) -> tuple[dict[str, str], int]:
        return jsonify({"error": "File too large. Maximum size is 16MB"}), 413

    return app


def run_server(
    host: str = "0.0.0.0",
    port: int = 5000,
    data_dir: Optional[Path] = None,
    use_edgetpu: bool = False,
    debug: bool = False,
    ssl_cert: Optional[str] = None,
    ssl_key: Optional[str] = None,
) -> None:
    """Run the Flask development server.

    Args:
        host: Host to bind to.
        port: Port to bind to.
        data_dir: Data directory path.
        use_edgetpu: Whether to use Edge TPU.
        debug: Enable debug mode.
        ssl_cert: Path to SSL certificate file (for HTTPS).
        ssl_key: Path to SSL private key file (for HTTPS).
    """
    if data_dir is None:
        data_dir = get_data_dir_from_env()
    app = create_app(data_dir=data_dir, use_edgetpu=use_edgetpu)

    # Get the SocketIO instance
    global socketio
    if socketio is None:
        raise RuntimeError("SocketIO not initialized")

    ssl_context = None
    if ssl_cert and ssl_key:
        ssl_context = (ssl_cert, ssl_key)
        logger.info(
            f"Starting HTTPS server with WebSocket support on https://{host}:{port}"
        )
    else:
        logger.info(
            f"Starting HTTP server with WebSocket support on http://{host}:{port}"
        )
        logger.info(
            "Note: For camera access from remote devices, HTTPS is recommended."
        )
        logger.info("Generate SSL certs with: ./generate-local-ssl.sh")

    # Use eventlet/gevent if available for proper WebSocket support
    # Otherwise fall back to threading (may have limitations)
    if eventlet is not None:
        # eventlet monkey patching is handled by flask-socketio
        # SSL is handled differently with eventlet
        run_kwargs = {
            "host": host,
            "port": port,
            "debug": debug,
            "use_reloader": False,
        }
        if ssl_cert and ssl_key:
            run_kwargs["certfile"] = ssl_cert
            run_kwargs["keyfile"] = ssl_key
        socketio.run(app, **run_kwargs)
    elif gevent is not None:
        run_kwargs = {
            "host": host,
            "port": port,
            "debug": debug,
            "use_reloader": False,
        }
        if ssl_cert and ssl_key:
            run_kwargs["certfile"] = ssl_cert
            run_kwargs["keyfile"] = ssl_key
        socketio.run(app, **run_kwargs)
    else:
        # Fallback to threading mode (may have WebSocket limitations)
        run_kwargs = {
            "host": host,
            "port": port,
            "debug": debug,
            "allow_unsafe_werkzeug": True,
        }
        if ssl_context:
            run_kwargs["ssl_context"] = ssl_context
        socketio.run(app, **run_kwargs)
