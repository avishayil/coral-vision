"""Flask API routes for Coral Vision."""
from __future__ import annotations

import base64
import tempfile
import threading
from pathlib import Path
from typing import Any, Optional

from flask import Blueprint, Flask, Response, jsonify, request
from flask_limiter import Limiter
from flask_socketio import SocketIO, disconnect, emit
from werkzeug.utils import secure_filename

from coral_vision.config import Paths
from coral_vision.core.exceptions import (
    DatabaseError,
    RecognitionError,
    ValidationError,
)
from coral_vision.core.file_validation import validate_image_file
from coral_vision.core.logger import get_logger
from coral_vision.core.pipeline_manager import VideoPipelineManager
from coral_vision.core.recognition import EmbeddingDB
from coral_vision.core.storage_pgvector import PgVectorStorageBackend
from coral_vision.core.validation import (
    validate_person_id,
    validate_person_name,
    validate_threshold,
)
from coral_vision.pipelines.enroll import enroll_person
from coral_vision.pipelines.recognize import recognize_folder
from coral_vision.pipelines.video_recognize import VideoRecognitionPipeline

logger = get_logger("api")

# Create API blueprints for versioning
api_bp = Blueprint("api", __name__, url_prefix="/api")
api_v1_bp = Blueprint("api_v1", __name__, url_prefix="/api/v1")

# API key for authentication (mandatory)
_api_key: Optional[str] = None


def _check_api_key() -> Optional[tuple[dict[str, str], int]]:
    """Check if API key is valid for the current request.

    Returns:
        None if authentication passes, or (error_response, status_code) if it fails.
    """
    # API key must be configured
    if not _api_key:
        return jsonify({"error": "API key not configured on server"}), 500

    # Get API key from request
    api_key = None

    # Check X-API-Key header first
    api_key = request.headers.get("X-API-Key")

    # Fall back to Authorization header (Bearer token)
    if not api_key:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            api_key = auth_header[7:]  # Remove "Bearer " prefix

    # Validate API key
    if not api_key or api_key != _api_key:
        return jsonify({"error": "Invalid or missing API key"}), 401

    return None


def set_api_key(api_key: str) -> None:
    """Set the API key for authentication.

    Args:
        api_key: API key string (required).
    """
    global _api_key
    if not api_key:
        raise ValueError("API key is required and cannot be empty")
    _api_key = api_key


# Pipeline manager for better state management
_pipeline_manager: Optional["VideoPipelineManager"] = None
_pipeline_manager_lock = threading.Lock()

# WebSocket streaming state
_active_streams: dict[str, bool] = {}  # Track active streams by session ID
_stream_threads: dict[str, threading.Thread] = {}  # Track streaming threads


def _get_pipeline_manager() -> "VideoPipelineManager":
    """Get or create pipeline manager instance.

    Returns:
        Video pipeline manager instance.
    """
    global _pipeline_manager
    with _pipeline_manager_lock:
        if _pipeline_manager is None:
            from coral_vision.core.pipeline_manager import VideoPipelineManager

            _pipeline_manager = VideoPipelineManager()
    return _pipeline_manager


def _get_video_pipeline(
    session_id: str,
    paths: Paths,
    use_edgetpu: bool,
    storage: "PgVectorStorageBackend",
    threshold: float = 0.6,
) -> VideoRecognitionPipeline:
    """Get or create video recognition pipeline instance for a session.

    Args:
        session_id: Session identifier.
        paths: Configuration paths object.
        use_edgetpu: Whether to use Edge TPU.
        storage: Storage backend instance.
        threshold: Recognition threshold for the pipeline.

    Returns:
        Video recognition pipeline instance.
    """
    manager = _get_pipeline_manager()
    return manager.get_pipeline(
        session_id=session_id,
        paths=paths,
        use_edgetpu=use_edgetpu,
        storage=storage,
        threshold=threshold,
    )


def register_api_routes(
    app: Flask,
    socketio: SocketIO,
    paths: Paths,
    storage: "PgVectorStorageBackend",
    use_edgetpu: bool,
    allowed_extensions: set[str],
    api_key: str,
    limiter: Limiter | None = None,
) -> None:
    """Register all API routes with the Flask application.

    Args:
        app: Flask application instance.
        paths: Configuration paths object.
        storage: Storage backend instance.
        use_edgetpu: Whether to use Edge TPU.
        allowed_extensions: Set of allowed file extensions.
        api_key: API key for authentication (required).
    """
    # Set API key for authentication
    set_api_key(api_key)

    # Register before_request hook to check API key for all API routes
    @api_bp.before_request
    def require_api_key() -> Optional[tuple[dict[str, str], int]]:
        """Check API key before processing any API request."""
        return _check_api_key()

    @api_v1_bp.before_request
    def require_api_key_v1() -> Optional[tuple[dict[str, str], int]]:
        """Check API key before processing any v1 API request."""
        return _check_api_key()

    # Helper to register route on both blueprints for backward compatibility
    def register_route_on_both(route_path: str, methods: list[str], func: Any) -> None:
        """Register a route on both legacy and v1 blueprints."""
        api_bp.add_url_rule(route_path, view_func=func, methods=methods)
        api_v1_bp.add_url_rule(route_path, view_func=func, methods=methods)

    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed."""
        if not filename:
            return False
        return (
            "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions
        )

    def list_persons() -> tuple[dict[str, Any], int]:
        """List all enrolled persons with pagination support.

        Query parameters:
            page: Page number (default: 1)
            per_page: Items per page (default: 50, max: 100)
        """
        try:
            # Get pagination parameters
            try:
                page = max(1, int(request.args.get("page", 1)))
                per_page = min(100, max(1, int(request.args.get("per_page", 50))))
            except (ValueError, TypeError):
                return jsonify({"error": "Invalid pagination parameters"}), 400

            people_index = storage.load_people_index()
            all_persons = [
                {"person_id": pid, "name": name}
                for pid, name in people_index.items()
                if pid != "unknown"
            ]

            # Calculate pagination
            total = len(all_persons)
            total_pages = (total + per_page - 1) // per_page  # Ceiling division
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page

            persons = all_persons[start_idx:end_idx]

            return (
                jsonify(
                    {
                        "persons": persons,
                        "pagination": {
                            "page": page,
                            "per_page": per_page,
                            "total": total,
                            "total_pages": total_pages,
                            "has_next": page < total_pages,
                            "has_prev": page > 1,
                        },
                    }
                ),
                200,
            )
        except DatabaseError as e:
            logger.error(f"Database error listing persons: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except Exception as e:
            logger.error(f"Error listing persons: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register list_persons on both blueprints
    register_route_on_both("/persons", ["GET"], list_persons)
    if limiter:
        limiter.limit("30 per minute")(list_persons)

    def create_person() -> tuple[dict[str, Any], int]:
        """Register a new person (metadata only, no training yet)."""
        try:
            data = request.get_json(silent=True)
            if not data:
                return jsonify({"error": "Request body must be JSON"}), 400

            person_id = data.get("person_id")
            name = data.get("name")

            if not person_id or not name:
                return (
                    jsonify({"error": "Both 'person_id' and 'name' are required"}),
                    400,
                )

            # Validate person_id and name format
            validate_person_id(person_id)
            validate_person_name(name)

            # Check if person already exists
            existing_name = storage.get_person_name(person_id)
            if existing_name:
                return jsonify({"error": f"Person '{person_id}' already exists"}), 409

            # Register person in storage
            storage.upsert_person(person_id, name)
            logger.info(f"Person registered: {person_id} ({name})")

            return (
                jsonify(
                    {
                        "person_id": person_id,
                        "name": name,
                        "message": "Person registered successfully. Upload training images next.",
                    }
                ),
                201,
            )

        except ValidationError as e:
            logger.warning(f"Validation error creating person: {e}")
            return jsonify({"error": str(e)}), 400
        except DatabaseError as e:
            logger.error(f"Database error creating person: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except Exception as e:
            logger.error(f"Error creating person: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register create_person on both blueprints
    register_route_on_both("/persons", ["POST"], create_person)
    if limiter:
        limiter.limit("10 per minute")(create_person)

    def get_person(person_id: str) -> tuple[dict[str, Any], int]:
        """Get details about a specific person."""
        try:
            name = storage.get_person_name(person_id)
            if not name:
                return jsonify({"error": f"Person '{person_id}' not found"}), 404

            # Get embedding count from storage
            num_embeddings = storage.get_embedding_count(person_id)

            return (
                jsonify(
                    {
                        "person_id": person_id,
                        "name": name,
                        "num_embeddings": num_embeddings,
                    }
                ),
                200,
            )

        except DatabaseError as e:
            logger.error(f"Database error getting person: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except Exception as e:
            logger.error(f"Error getting person: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register get_person on both blueprints
    register_route_on_both("/persons/<person_id>", ["GET"], get_person)
    if limiter:
        limiter.limit("30 per minute")(get_person)

    def delete_person(person_id: str) -> tuple[dict[str, Any], int]:
        """Delete a person and all their training data."""
        try:
            name = storage.get_person_name(person_id)
            if not name:
                return jsonify({"error": f"Person '{person_id}' not found"}), 404

            # Delete from storage backend
            storage.delete_person(person_id)
            logger.info(f"Person deleted: {person_id} ({name})")

            return (
                jsonify({"message": f"Person '{person_id}' deleted successfully"}),
                200,
            )

        except DatabaseError as e:
            logger.error(f"Database error deleting person: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except Exception as e:
            logger.error(f"Error deleting person: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register delete_person on both blueprints
    register_route_on_both("/persons/<person_id>", ["DELETE"], delete_person)
    if limiter:
        limiter.limit("10 per minute")(delete_person)

    def train_person(person_id: str) -> tuple[dict[str, Any], int]:
        """Upload training images for a person and generate embeddings."""
        try:
            # Verify person exists
            name = storage.get_person_name(person_id)
            if not name:
                return jsonify({"error": f"Person '{person_id}' not found"}), 404

            # Get uploaded files
            if "images" not in request.files:
                return (
                    jsonify(
                        {
                            "error": "No images provided. Use 'images' field for file uploads"
                        }
                    ),
                    400,
                )

            files = request.files.getlist("images")
            if not files or all(f.filename == "" for f in files):
                return jsonify({"error": "No images selected"}), 400

            # Validate files comprehensively
            for file in files:
                if not file.filename:
                    continue

                # Basic extension check
                if not allowed_file(file.filename):
                    return (
                        jsonify(
                            {
                                "error": f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
                            }
                        ),
                        400,
                    )

                # Comprehensive image validation
                file_content = file.read()
                file.seek(0)  # Reset file pointer for later use

                is_valid, error_msg = validate_image_file(
                    file_content,
                    file.filename,
                    allowed_extensions,
                    max_size=app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024),
                )

                if not is_valid:
                    logger.warning(
                        f"File validation failed for {file.filename}: {error_msg}"
                    )
                    return jsonify({"error": f"Invalid image file: {error_msg}"}), 400

            # Save files to temp directory and process
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                saved_files = []

                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = tmp_path / filename
                        file.save(str(filepath))
                        saved_files.append(filepath)

                # Process images with enrollment pipeline
                enroll_person(
                    paths=paths,
                    person_id=person_id,
                    name=name,
                    images_path=tmp_path,
                    use_edgetpu=use_edgetpu,
                    min_score=0.95,
                    max_faces=1,
                    keep_copies=True,
                    storage=storage,
                )

            # Get embedding count from storage
            num_embeddings = storage.get_embedding_count(person_id)

            return (
                jsonify(
                    {
                        "person_id": person_id,
                        "name": name,
                        "images_processed": len(saved_files),
                        "total_embeddings": num_embeddings,
                        "message": "Training completed successfully",
                    }
                ),
                200,
            )

        except DatabaseError as e:
            logger.error(f"Database error in train_person: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except RecognitionError as e:
            logger.error(f"Recognition error in train_person: {e}", exc_info=True)
            return jsonify({"error": "Face recognition failed"}), 500
        except Exception as e:
            logger.error(f"Error in train_person: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register train_person on both blueprints
    register_route_on_both("/persons/<person_id>/train", ["POST"], train_person)
    if limiter:
        limiter.limit("5 per minute")(train_person)

    def recognize() -> tuple[dict[str, Any], int]:
        """Recognize faces in uploaded images."""
        try:
            # Get uploaded files
            if "images" not in request.files:
                return (
                    jsonify(
                        {
                            "error": "No images provided. Use 'images' field for file uploads"
                        }
                    ),
                    400,
                )

            files = request.files.getlist("images")
            if not files or all(f.filename == "" for f in files):
                return jsonify({"error": "No images selected"}), 400

            # Get and validate optional parameters
            try:
                threshold = float(request.form.get("threshold", 0.6))
                validate_threshold(threshold)
            except (ValueError, ValidationError) as e:
                return jsonify({"error": f"Invalid threshold: {e}"}), 400

            top_k = int(request.form.get("top_k", 3))
            if top_k < 1 or top_k > 100:
                return jsonify({"error": "top_k must be between 1 and 100"}), 400

            # Validate files
            for file in files:
                if not file.filename:
                    continue

                # Basic extension check
                if not allowed_file(file.filename):
                    return (
                        jsonify(
                            {
                                "error": f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
                            }
                        ),
                        400,
                    )

                # Comprehensive image validation
                file_content = file.read()
                file.seek(0)  # Reset file pointer for later use

                is_valid, error_msg = validate_image_file(
                    file_content,
                    file.filename,
                    allowed_extensions,
                    max_size=app.config.get("MAX_CONTENT_LENGTH", 16 * 1024 * 1024),
                )

                if not is_valid:
                    logger.warning(
                        f"File validation failed for {file.filename}: {error_msg}"
                    )
                    return jsonify({"error": f"Invalid image file: {error_msg}"}), 400

            # Save files to temp directory and process
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                saved_files = []

                for file in files:
                    if file.filename:
                        filename = secure_filename(file.filename)
                        filepath = tmp_path / filename
                        file.save(str(filepath))
                        saved_files.append(filepath)

                # Run recognition pipeline
                result_dict = recognize_folder(
                    paths=paths,
                    input_path=tmp_path,
                    use_edgetpu=use_edgetpu,
                    threshold=threshold,
                    top_k=top_k,
                    per_person_k=20,
                    say=False,
                    storage=storage,
                )

            # The result is already properly formatted
            return jsonify(result_dict), 200

        except RecognitionError as e:
            logger.error(f"Recognition error: {e}", exc_info=True)
            return jsonify({"error": "Face recognition failed"}), 500
        except DatabaseError as e:
            logger.error(f"Database error during recognition: {e}", exc_info=True)
            return jsonify({"error": "Database operation failed"}), 503
        except Exception as e:
            logger.error(f"Error in recognize endpoint: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500

    # Register recognize on both blueprints
    register_route_on_both("/recognize", ["POST"], recognize)
    if limiter:
        limiter.limit("20 per minute")(recognize)

    @api_bp.route("/video_feed")
    @api_v1_bp.route("/video_feed")
    def video_feed() -> Response:
        """Video streaming route with face recognition (server-side camera)."""
        threshold = float(request.args.get("threshold", 0.6))
        # Use a default session ID for server-side camera
        session_id = f"server_camera_{request.remote_addr}"
        pipeline = _get_video_pipeline(
            session_id, paths, use_edgetpu, storage, threshold
        )
        return Response(
            pipeline.generate_frames(camera_index=0, width=640, height=480),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @api_bp.route("/process_frame", methods=["POST"])
    @api_v1_bp.route("/process_frame", methods=["POST"])
    def process_frame() -> tuple[dict[str, Any], int]:
        """Process a single frame for face recognition (client-side camera)."""
        try:
            if "image" not in request.files:
                return jsonify({"error": "No image provided"}), 400

            file = request.files["image"]
            if not file.filename:
                return jsonify({"error": "No image selected"}), 400

            if not allowed_file(file.filename):
                return (
                    jsonify(
                        {
                            "error": f"Invalid file type: {file.filename}. Allowed: {', '.join(allowed_extensions)}"
                        }
                    ),
                    400,
                )

            threshold = float(request.form.get("threshold", 0.6))

            # Save file to temp and process
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                filename = secure_filename(file.filename)
                filepath = tmp_path / filename
                file.save(str(filepath))

                # Run recognition on single image
                result_dict = recognize_folder(
                    paths=paths,
                    input_path=filepath,
                    use_edgetpu=use_edgetpu,
                    threshold=threshold,
                    top_k=3,
                    per_person_k=20,
                    say=False,
                    storage=storage,
                )

            # Return first result (single image)
            if result_dict.get("results"):
                return jsonify(result_dict["results"][0]), 200
            return jsonify({"faces": []}), 200

        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @api_bp.route("/camera/stop", methods=["POST"])
    @api_v1_bp.route("/camera/stop", methods=["POST"])
    def stop_camera() -> tuple[dict[str, Any], int]:
        """Stop and release camera resources."""
        global _video_pipeline
        try:
            # Pipeline will release camera when generator exits
            # This endpoint is mainly for client-side cleanup
            _video_pipeline = None
            return jsonify({"message": "Camera resources released"}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    # Apply rate limiting to endpoints if limiter is provided
    if limiter:
        # Apply rate limits after all routes are defined
        # Note: This must be done before registering the blueprint
        pass  # Rate limits applied via decorators above

    # Register blueprints with app
    # Register v1 (new versioned API) first
    app.register_blueprint(api_v1_bp)
    # Register legacy API (backward compatibility)
    app.register_blueprint(api_bp)

    # WebSocket event handlers
    @socketio.on("connect")
    def handle_connect(auth: Optional[dict[str, Any]] = None) -> None:
        """Handle WebSocket connection."""
        # Check API key from auth data
        api_key_provided = None
        if auth:
            api_key_provided = auth.get("api_key") or auth.get("X-API-Key")

        # Also check query string
        if not api_key_provided:
            api_key_provided = request.args.get("api_key")

        if not api_key_provided or api_key_provided != api_key:
            logger.warning(
                f"WebSocket connection rejected: Invalid or missing API key from {request.remote_addr}"
            )
            disconnect()
            return

        logger.info(
            f"WebSocket client connected: {request.sid} from {request.remote_addr}"
        )
        emit("connected", {"status": "ok"})

    @socketio.on("disconnect")
    def handle_disconnect() -> None:
        """Handle WebSocket disconnection."""
        session_id = request.sid
        logger.info(f"WebSocket client disconnected: {session_id}")

        # Stop any active stream for this session
        if session_id in _active_streams:
            _active_streams[session_id] = False
        if session_id in _stream_threads:
            thread = _stream_threads.pop(session_id)
            if thread.is_alive():
                # Thread will stop on next iteration when _active_streams[session_id] is False
                pass

    @socketio.on("start_video_stream")
    def handle_start_video_stream(data: dict[str, Any]) -> None:
        """Start video streaming via WebSocket (client-side camera)."""
        session_id = request.sid

        # Validate threshold
        try:
            threshold = float(data.get("threshold", 0.6))
            validate_threshold(threshold)
        except (ValueError, ValidationError) as e:
            logger.warning(f"Invalid threshold in start_video_stream: {e}")
            socketio.emit(
                "stream_error", {"error": f"Invalid threshold: {e}"}, room=session_id
            )
            return

        # Check if database has embeddings
        try:
            db = EmbeddingDB.load_from_backend(storage)  # noqa: F841
            total_embeddings = sum(
                storage.get_embedding_count(pid)
                for pid in storage.load_people_index().keys()
                if pid != "unknown"
            )
            logger.info(f"Database loaded: {total_embeddings} total embeddings")
            if total_embeddings == 0:
                logger.warning(
                    "No embeddings found in database. Recognition will not work."
                )
        except DatabaseError as e:
            logger.warning(f"Could not check database: {e}")
        except Exception as e:
            logger.warning(f"Could not check database: {e}", exc_info=True)

        # Mark session as active for processing frames
        _active_streams[session_id] = True

        # Get or create video pipeline for processing frames (per session)
        pipeline = _get_video_pipeline(  # noqa: F841
            session_id, paths, use_edgetpu, storage, threshold
        )

        # Store pipeline reference for this session (simplified - using global for now)
        # In production, you might want a session-to-pipeline mapping
        logger.info(f"WebSocket video stream ready for session: {session_id}")
        socketio.emit("stream_started", {"status": "ok"}, room=session_id)

    @socketio.on("process_frame")
    def handle_process_frame(data: dict[str, Any]) -> None:
        """Process a single frame from client-side camera via WebSocket."""
        session_id = request.sid

        if not _active_streams.get(session_id, False):
            return  # Stream not active for this session

        try:
            # Get frame data (base64 encoded JPEG)
            frame_base64 = data.get("frame")
            if not frame_base64:
                socketio.emit(
                    "recognition_result",
                    {"error": "No frame data provided"},
                    room=session_id,
                )
                return

            threshold = float(data.get("threshold", 0.6))

            # Decode base64 frame and save to temp file
            frame_bytes = base64.b64decode(frame_base64)

            # Save to temp file and use existing recognize_folder function
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(frame_bytes)
                tmp_path = Path(tmp_file.name)

            try:
                # Use existing recognition pipeline
                result_dict = recognize_folder(
                    paths=paths,
                    input_path=tmp_path,
                    use_edgetpu=use_edgetpu,
                    threshold=threshold,
                    top_k=1,
                    per_person_k=20,
                    say=False,
                    storage=storage,
                )

                # Extract faces from result - match frontend expected structure
                faces = []
                if result_dict.get("results"):
                    result = result_dict["results"][0]
                    for face in result.get("faces", []):
                        # Get predicted match (best match, may or may not be accepted)
                        predicted = face.get("predicted")
                        accepted = face.get("accepted", False)
                        matches = face.get("matches", [])

                        # Build face result matching frontend expected structure
                        face_result = {
                            "bbox": {
                                "xmin": face.get("bbox", {}).get("xmin", 0),
                                "ymin": face.get("bbox", {}).get("ymin", 0),
                                "xmax": face.get("bbox", {}).get("xmax", 0),
                                "ymax": face.get("bbox", {}).get("ymax", 0),
                            },
                            "accepted": accepted,
                        }

                        # Add predicted match if available
                        if predicted:
                            face_result["predicted"] = {
                                "name": predicted.get("name", "Unknown"),
                                "distance": predicted.get("distance"),
                                "person_id": predicted.get("person_id"),
                            }

                        # Add matches list for debugging
                        if matches:
                            face_result["matches"] = matches

                        faces.append(face_result)

                        # Debug logging
                        if predicted:
                            logger.debug(
                                f"Face detected: {predicted.get('name', 'Unknown')} "
                                f"(distance: {predicted.get('distance', 0):.4f}, "
                                f"threshold: {threshold}, accepted: {accepted})"
                            )
                        else:
                            logger.debug("Face detected but no match found in database")

                # Send recognition results back to client
                socketio.emit(
                    "recognition_result",
                    {"faces": faces, "timestamp": data.get("timestamp")},
                    room=session_id,
                )

            finally:
                # Clean up temp file
                try:
                    tmp_path.unlink()
                except Exception:
                    pass

        except RecognitionError as e:
            error_msg = str(e)
            logger.error(
                f"Recognition error processing frame: {error_msg}", exc_info=True
            )
            try:
                socketio.emit(
                    "recognition_result",
                    {"error": "Recognition failed", "faces": []},
                    room=session_id,
                )
            except Exception:
                pass
        except DatabaseError as e:
            error_msg = str(e)
            logger.error(f"Database error processing frame: {error_msg}", exc_info=True)
            try:
                socketio.emit(
                    "recognition_result",
                    {"error": "Database unavailable", "faces": []},
                    room=session_id,
                )
            except Exception:
                pass
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Frame processing error: {error_msg}", exc_info=True)
            try:
                socketio.emit(
                    "recognition_result",
                    {"error": error_msg, "faces": []},
                    room=session_id,
                )
            except Exception:
                pass

    @socketio.on("stop_video_stream")
    def handle_stop_video_stream() -> None:
        """Stop video streaming."""
        session_id = request.sid
        if session_id in _active_streams:
            _active_streams[session_id] = False
            logger.info(f"WebSocket video stream stopped for session: {session_id}")

            # Clean up pipeline for this session
            try:
                manager = _get_pipeline_manager()
                # Get threshold from active stream if available, or use default
                threshold = 0.6  # Default, could be stored per session
                manager.remove_pipeline(session_id, threshold)
            except Exception as e:
                logger.warning(
                    f"Error cleaning up pipeline for session {session_id}: {e}"
                )

            emit("stream_stopped", {"status": "ok"})
