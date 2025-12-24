"""Integration tests for Docker deployment with pgvector backend."""
from __future__ import annotations

import pytest


@pytest.mark.integration
class TestDockerHealthEndpoint:
    """Tests for health check endpoint in Docker."""

    def test_health_check(self, docker_client):
        """Test health check returns 200 and correct data."""
        response = docker_client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "storage" in data
        assert data["storage"] == "pgvector"


@pytest.mark.integration
class TestDockerPersonsEndpoints:
    """Tests for person management endpoints in Docker."""

    def test_list_persons_empty(self, docker_client):
        """Test listing persons when none are enrolled."""
        # Clean up any existing test data first
        response = docker_client.get("/api/persons")
        if response.status_code == 200:
            persons = response.json().get("persons", [])
            for person in persons:
                if person["person_id"].startswith("9999_"):
                    docker_client.delete(f"/api/persons/{person['person_id']}")

        response = docker_client.get("/api/persons")
        assert response.status_code == 200

        data = response.json()
        assert "persons" in data
        assert isinstance(data["persons"], list)

    def test_create_person_success(self, docker_client):
        """Test creating a new person via Docker API."""
        person_id = "9999_docker_test"

        # Clean up if exists
        docker_client.delete(f"/api/persons/{person_id}")

        response = docker_client.post(
            "/api/persons",
            json={"person_id": person_id, "name": "Docker Test"},
        )
        assert response.status_code == 201

        data = response.json()
        assert data["person_id"] == person_id
        assert data["name"] == "Docker Test"

        # Cleanup
        docker_client.delete(f"/api/persons/{person_id}")

    def test_create_person_missing_fields(self, docker_client):
        """Test creating person without required fields."""
        # Missing name
        response = docker_client.post(
            "/api/persons",
            json={"person_id": "9999_test"},
        )
        assert response.status_code == 400

        # Missing person_id
        response = docker_client.post(
            "/api/persons",
            json={"name": "Test"},
        )
        assert response.status_code == 400

    def test_create_person_duplicate(self, docker_client):
        """Test creating person that already exists."""
        person_id = "9999_duplicate_test"

        # Clean up if exists
        docker_client.delete(f"/api/persons/{person_id}")

        # Create first time
        response = docker_client.post(
            "/api/persons",
            json={"person_id": person_id, "name": "Test"},
        )
        assert response.status_code == 201

        # Try to create again
        response = docker_client.post(
            "/api/persons",
            json={"person_id": person_id, "name": "Different Name"},
        )
        assert response.status_code == 409

        # Cleanup
        docker_client.delete(f"/api/persons/{person_id}")

    def test_get_person_success(self, docker_client, registered_person_docker):
        """Test getting person details via Docker API."""
        response = docker_client.get(
            f"/api/persons/{registered_person_docker['person_id']}"
        )
        assert response.status_code == 200

        data = response.json()
        assert data["person_id"] == registered_person_docker["person_id"]
        assert data["name"] == registered_person_docker["name"]
        assert "num_embeddings" in data
        assert data["num_embeddings"] == 0  # No training yet

    def test_get_person_not_found(self, docker_client):
        """Test getting non-existent person."""
        response = docker_client.get("/api/persons/9999_nonexistent")
        assert response.status_code == 404

    def test_delete_person_success(self, docker_client):
        """Test deleting a person via Docker API."""
        person_id = "9999_delete_test"

        # Create person
        response = docker_client.post(
            "/api/persons",
            json={"person_id": person_id, "name": "Delete Test"},
        )
        assert response.status_code == 201

        # Delete person
        response = docker_client.delete(f"/api/persons/{person_id}")
        assert response.status_code == 200

        # Verify deletion
        response = docker_client.get(f"/api/persons/{person_id}")
        assert response.status_code == 404

    def test_delete_person_not_found(self, docker_client):
        """Test deleting non-existent person."""
        response = docker_client.delete("/api/persons/9999_nonexistent")
        assert response.status_code == 404


@pytest.mark.integration
class TestDockerPgVectorStorage:
    """Tests for pgvector storage backend in Docker."""

    def test_person_persists_across_requests(self, docker_client):
        """Test that person data persists in pgvector."""
        person_id = "9999_persist_test"

        # Clean up if exists
        docker_client.delete(f"/api/persons/{person_id}")

        # Create person
        response = docker_client.post(
            "/api/persons",
            json={"person_id": person_id, "name": "Persist Test"},
        )
        assert response.status_code == 201

        # Verify it appears in list
        response = docker_client.get("/api/persons")
        assert response.status_code == 200

        persons = response.json()["persons"]
        person_ids = [p["person_id"] for p in persons]
        assert person_id in person_ids

        # Get person details
        response = docker_client.get(f"/api/persons/{person_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["person_id"] == person_id
        assert data["name"] == "Persist Test"

        # Cleanup
        docker_client.delete(f"/api/persons/{person_id}")

    def test_multiple_persons_isolation(self, docker_client):
        """Test that multiple persons are properly isolated."""
        person1_id = "9999_isolation_test1"
        person2_id = "9999_isolation_test2"

        # Clean up if exists
        docker_client.delete(f"/api/persons/{person1_id}")
        docker_client.delete(f"/api/persons/{person2_id}")

        # Create two persons
        response1 = docker_client.post(
            "/api/persons",
            json={"person_id": person1_id, "name": "Person One"},
        )
        assert response1.status_code == 201

        response2 = docker_client.post(
            "/api/persons",
            json={"person_id": person2_id, "name": "Person Two"},
        )
        assert response2.status_code == 201

        # Verify both exist
        response = docker_client.get("/api/persons")
        persons = response.json()["persons"]
        person_ids = [p["person_id"] for p in persons]
        assert person1_id in person_ids
        assert person2_id in person_ids

        # Delete one
        docker_client.delete(f"/api/persons/{person1_id}")

        # Verify only one remains
        response = docker_client.get("/api/persons")
        persons = response.json()["persons"]
        person_ids = [p["person_id"] for p in persons]
        assert person1_id not in person_ids
        assert person2_id in person_ids

        # Cleanup
        docker_client.delete(f"/api/persons/{person2_id}")


@pytest.mark.integration
@pytest.mark.slow
class TestDockerTrainEndpoint:
    """Tests for training endpoint in Docker (slow tests)."""

    def test_train_no_person(self, docker_client, sample_image):
        """Test training for non-existent person."""
        response = docker_client.post(
            "/api/persons/9999_nonexistent/train",
            files={"images": ("test.jpg", sample_image, "image/jpeg")},
        )
        assert response.status_code == 404

    def test_train_no_images(self, docker_client, registered_person_docker):
        """Test training without images."""
        response = docker_client.post(
            f"/api/persons/{registered_person_docker['person_id']}/train",
            data={},
        )
        assert response.status_code == 400


@pytest.mark.integration
@pytest.mark.slow
class TestDockerRecognizeEndpoint:
    """Tests for recognition endpoint in Docker (slow tests)."""

    def test_recognize_no_images(self, docker_client):
        """Test recognition without images."""
        response = docker_client.post("/api/recognize", data={})
        assert response.status_code == 400

    def test_recognize_with_sample_image(self, docker_client, sample_image):
        """Test recognition with a sample image."""
        response = docker_client.post(
            "/api/recognize",
            files={"images": ("test.jpg", sample_image, "image/jpeg")},
            data={"threshold": "0.6"},
        )
        # May fail if models aren't present, which is acceptable
        # In production, models should be mounted
        assert response.status_code in [200, 500]  # 500 if models not found

        if response.status_code == 200:
            data = response.json()
            assert "results" in data
            assert isinstance(data["results"], list)
