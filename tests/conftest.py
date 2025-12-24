"""Pytest fixtures and test configuration for Coral Vision tests."""
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Generator

import psycopg2
import pytest
import requests
from PIL import Image

from coral_vision.core.storage_pgvector import PgVectorStorageBackend
from coral_vision.web.app import create_app


# Docker management fixtures
@pytest.fixture(scope="session")
def docker_compose_project():
    """Manage Docker Compose lifecycle for the test session."""
    project_root = Path(__file__).parent.parent

    # Check if containers are already running
    result = subprocess.run(
        ["docker", "compose", "ps", "-q"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    containers_running = bool(result.stdout.strip())
    started_by_us = False

    if not containers_running:
        print("\n🐳 Starting Docker containers for tests...")
        subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=project_root,
            check=True,
        )
        started_by_us = True
        time.sleep(5)  # Give containers time to start
    else:
        print("\n🐳 Using existing Docker containers...")

    yield project_root

    # Only stop containers if we started them
    if started_by_us:
        print("\n🛑 Stopping Docker containers...")
        subprocess.run(
            ["docker", "compose", "down"],
            cwd=project_root,
            check=False,  # Don't fail if already stopped
        )


@pytest.fixture
def temp_data_dir() -> Generator[Path, None, None]:
    """Create a temporary data directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create required subdirectories (only models now)
        (data_dir / "models").mkdir(parents=True)

        yield data_dir


@pytest.fixture(scope="session")
def docker_db_config():
    """Database configuration for Docker containers."""
    return {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", "5432")),
        "database": os.getenv("DB_NAME", "coral_vision_test"),
        "user": os.getenv("DB_USER", "coral"),
        "password": os.getenv("DB_PASSWORD", "coral"),
    }


@pytest.fixture(scope="session")
def wait_for_db(docker_compose_project, docker_db_config):
    """Wait for PostgreSQL database to be ready."""
    max_retries = 30
    retry_delay = 1

    for i in range(max_retries):
        try:
            conn = psycopg2.connect(**docker_db_config)
            conn.close()
            print("\n✓ Database connection established")
            return True
        except psycopg2.OperationalError as e:
            if i < max_retries - 1:
                if i == 0:  # Only print once
                    print("\n⏳ Waiting for database...")
                time.sleep(retry_delay)
            else:
                raise RuntimeError(
                    f"Could not connect to database after {max_retries} attempts: {e}"
                )


@pytest.fixture(scope="session")
def initialize_test_db(wait_for_db, docker_compose_project, docker_db_config):
    """Initialize test database with schema."""
    print("\n🗄️  Initializing test database...")

    # Connect to main database to create test database
    main_config = docker_db_config.copy()
    main_config["database"] = "coral_vision"

    try:
        conn = psycopg2.connect(**main_config)
        conn.autocommit = True
        with conn.cursor() as cur:
            # Drop and create test database
            cur.execute("DROP DATABASE IF EXISTS coral_vision_test")
            cur.execute("CREATE DATABASE coral_vision_test")
        conn.close()

        # Connect to test database and set up schema
        conn = psycopg2.connect(**docker_db_config)
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS people (
                    person_id VARCHAR(255) PRIMARY KEY,
                    name VARCHAR(255) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    person_id VARCHAR(255) REFERENCES people(person_id) ON DELETE CASCADE,
                    embedding vector(192) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS embeddings_embedding_idx
                ON embeddings USING hnsw (embedding vector_l2_ops)
            """
            )
            cur.execute(
                """
                INSERT INTO people (person_id, name)
                VALUES ('unknown', 'unknown')
                ON CONFLICT DO NOTHING
            """
            )
        conn.commit()
        conn.close()

        print("✓ Test database initialized")
        return True

    except Exception as e:
        print(f"⚠️ Failed to initialize test database: {e}")
        raise


@pytest.fixture(scope="session")
def wait_for_api(docker_compose_project, wait_for_db):
    """Wait for Coral Vision API to be ready."""
    api_url = os.getenv("API_URL", "http://localhost:5000")
    max_retries = 30
    retry_delay = 1

    for i in range(max_retries):
        try:
            response = requests.get(f"{api_url}/health", timeout=2)
            if response.status_code == 200:
                print(f"\n✓ API connection established at {api_url}")
                return api_url
        except requests.RequestException:
            pass

        if i < max_retries - 1:
            if i == 0:  # Only print once
                print("\n⏳ Waiting for API...")
            time.sleep(retry_delay)
        else:
            raise RuntimeError(f"Could not connect to API after {max_retries} attempts")


@pytest.fixture
def storage_backend(initialize_test_db, docker_db_config):
    """Create pgvector storage backend connected to Docker database."""
    # Override config with test database
    os.environ.update(
        {
            "DB_HOST": docker_db_config["host"],
            "DB_PORT": str(docker_db_config["port"]),
            "DB_NAME": docker_db_config["database"],
            "DB_USER": docker_db_config["user"],
            "DB_PASSWORD": docker_db_config["password"],
        }
    )

    storage = PgVectorStorageBackend()
    storage.ensure_initialized()

    yield storage

    # Cleanup: Remove test data
    try:
        with storage._get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM embeddings")
                cur.execute("DELETE FROM people WHERE person_id != 'unknown'")
            conn.commit()
    except Exception as e:
        print(f"\n⚠️ Cleanup warning: {e}")


@pytest.fixture
def app(storage_backend, temp_data_dir: Path):
    """Create Flask app for testing with pgvector backend."""
    app = create_app(data_dir=temp_data_dir, use_edgetpu=False)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create Flask test client."""
    return app.test_client()


@pytest.fixture
def sample_image() -> bytes:
    """Create a sample image for testing."""
    # Create a simple RGB image
    img = Image.new("RGB", (640, 480), color=(73, 109, 137))

    # Save to bytes
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_with_face() -> bytes:
    """Create a sample image with a face-like pattern."""
    # Create image with a simple pattern that might be detected as a face
    img = Image.new("RGB", (640, 480), color=(255, 255, 255))

    # Draw a simple face-like pattern
    from PIL import ImageDraw

    draw = ImageDraw.Draw(img)

    # Face oval
    draw.ellipse([200, 150, 440, 390], fill=(255, 220, 177), outline=(0, 0, 0))

    # Eyes
    draw.ellipse([250, 220, 290, 260], fill=(0, 0, 0))
    draw.ellipse([350, 220, 390, 260], fill=(0, 0, 0))

    # Mouth
    draw.arc([270, 280, 370, 350], 0, 180, fill=(0, 0, 0), width=3)

    # Save to bytes
    from io import BytesIO

    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def registered_person(client):
    """Register a test person."""
    response = client.post(
        "/api/persons",
        json={"person_id": "9999_test_person", "name": "Test Person"},
    )
    assert response.status_code == 201
    return {"person_id": "9999_test_person", "name": "Test Person"}


@pytest.fixture
def docker_client(wait_for_api):
    """Create a test client for Docker-based API."""

    class DockerAPIClient:
        """Test client for Docker API."""

        def __init__(self, base_url):
            self.base_url = base_url

        def get(self, path, **kwargs):
            """Send GET request."""
            return requests.get(f"{self.base_url}{path}", **kwargs)

        def post(self, path, **kwargs):
            """Send POST request."""
            return requests.post(f"{self.base_url}{path}", **kwargs)

        def delete(self, path, **kwargs):
            """Send DELETE request."""
            return requests.delete(f"{self.base_url}{path}", **kwargs)

    return DockerAPIClient(wait_for_api)


@pytest.fixture
def registered_person_docker(docker_client):
    """Register a test person via Docker API."""
    response = docker_client.post(
        "/api/persons",
        json={"person_id": "9999_test_person", "name": "Test Person"},
    )
    assert response.status_code == 201

    yield {"person_id": "9999_test_person", "name": "Test Person"}

    # Cleanup
    try:
        docker_client.delete("/api/persons/9999_test_person")
    except Exception as e:
        print(f"\n⚠️ Cleanup warning: {e}")
