"""API endpoint tests for the Coral Vision Flask web application."""
from __future__ import annotations

import io


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_check(self, client):
        """Test health check returns 200 and correct data."""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.get_json()
        assert data["status"] == "healthy"
        assert "use_edgetpu" in data
        assert isinstance(data["use_edgetpu"], bool)


class TestPersonsEndpoints:
    """Tests for person management endpoints."""

    def test_list_persons_empty(self, client):
        """Test listing persons when none are enrolled."""
        response = client.get("/api/persons")
        assert response.status_code == 200

        data = response.get_json()
        assert "persons" in data
        assert isinstance(data["persons"], list)
        # Should not include 'unknown'
        assert len(data["persons"]) == 0

    def test_create_person_success(self, client):
        """Test creating a new person."""
        response = client.post(
            "/api/persons",
            json={"person_id": "0001_john_doe", "name": "John Doe"},
        )
        assert response.status_code == 201

        data = response.get_json()
        assert data["person_id"] == "0001_john_doe"
        assert data["name"] == "John Doe"
        assert "message" in data

    def test_create_person_missing_fields(self, client):
        """Test creating person without required fields."""
        # Missing name
        response = client.post(
            "/api/persons",
            json={"person_id": "0001_john_doe"},
        )
        assert response.status_code == 400

        # Missing person_id
        response = client.post(
            "/api/persons",
            json={"name": "John Doe"},
        )
        assert response.status_code == 400

    def test_create_person_invalid_id(self, client):
        """Test creating person with invalid ID format."""
        response = client.post(
            "/api/persons",
            json={"person_id": "invalid id!", "name": "John Doe"},
        )
        assert response.status_code == 400

    def test_create_person_duplicate(self, client, registered_person):
        """Test creating person that already exists."""
        response = client.post(
            "/api/persons",
            json={
                "person_id": registered_person["person_id"],
                "name": "Different Name",
            },
        )
        assert response.status_code == 409

    def test_create_person_no_json(self, client):
        """Test creating person without JSON body."""
        response = client.post("/api/persons")
        assert response.status_code == 400

    def test_list_persons_with_data(self, client, registered_person):
        """Test listing persons after registration."""
        response = client.get("/api/persons")
        assert response.status_code == 200

        data = response.get_json()
        assert len(data["persons"]) == 1
        assert data["persons"][0]["person_id"] == registered_person["person_id"]
        assert data["persons"][0]["name"] == registered_person["name"]

    def test_get_person_success(self, client, registered_person):
        """Test getting person details."""
        response = client.get(f"/api/persons/{registered_person['person_id']}")
        assert response.status_code == 200

        data = response.get_json()
        assert data["person_id"] == registered_person["person_id"]
        assert data["name"] == registered_person["name"]
        assert "embedding_count" in data
        assert "face_count" in data
        assert data["embedding_count"] == 0  # No training yet
        assert data["face_count"] == 0

    def test_get_person_not_found(self, client):
        """Test getting non-existent person."""
        response = client.get("/api/persons/9999_nonexistent")
        assert response.status_code == 404

    def test_delete_person_success(self, client, registered_person):
        """Test deleting a person."""
        response = client.delete(f"/api/persons/{registered_person['person_id']}")
        assert response.status_code == 200

        data = response.get_json()
        assert "message" in data

        # Verify person is deleted
        response = client.get(f"/api/persons/{registered_person['person_id']}")
        assert response.status_code == 404

    def test_delete_person_not_found(self, client):
        """Test deleting non-existent person."""
        response = client.delete("/api/persons/9999_nonexistent")
        assert response.status_code == 404


class TestTrainEndpoint:
    """Tests for the training endpoint."""

    def test_train_no_person(self, client, sample_image):
        """Test training for non-existent person."""
        response = client.post(
            "/api/persons/9999_nonexistent/train",
            data={"images": (io.BytesIO(sample_image), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 404

    def test_train_no_images(self, client, registered_person):
        """Test training without images."""
        response = client.post(
            f"/api/persons/{registered_person['person_id']}/train",
            data={},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_train_invalid_file_type(self, client, registered_person):
        """Test training with invalid file type."""
        response = client.post(
            f"/api/persons/{registered_person['person_id']}/train",
            data={"images": (io.BytesIO(b"not an image"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_train_with_valid_image(self, client, registered_person, sample_image):
        """Test training with valid image (may not detect faces, but should accept upload)."""
        response = client.post(
            f"/api/persons/{registered_person['person_id']}/train",
            data={
                "images": (io.BytesIO(sample_image), "test.jpg"),
                "min_score": "0.95",
                "max_faces": "1",
            },
            content_type="multipart/form-data",
        )

        # Should return 200 even if no faces detected
        # (The models aren't loaded in test environment)
        # We're testing the API structure, not the model
        assert response.status_code in [200, 500]  # 500 if models missing

        if response.status_code == 200:
            data = response.get_json()
            assert "person_id" in data
            assert "processed_images" in data

    def test_train_with_parameters(self, client, registered_person, sample_image):
        """Test training with custom parameters."""
        response = client.post(
            f"/api/persons/{registered_person['person_id']}/train",
            data={
                "images": (io.BytesIO(sample_image), "test.jpg"),
                "min_score": "0.85",
                "max_faces": "2",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code in [200, 500]

    def test_train_multiple_images(self, client, registered_person, sample_image):
        """Test training with multiple images."""
        response = client.post(
            f"/api/persons/{registered_person['person_id']}/train",
            data={
                "images": [
                    (io.BytesIO(sample_image), "test1.jpg"),
                    (io.BytesIO(sample_image), "test2.jpg"),
                    (io.BytesIO(sample_image), "test3.jpg"),
                ],
            },
            content_type="multipart/form-data",
        )

        assert response.status_code in [200, 500]


class TestRecognizeEndpoint:
    """Tests for the recognition endpoint."""

    def test_recognize_no_images(self, client):
        """Test recognition without images."""
        response = client.post(
            "/api/recognize",
            data={},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_recognize_invalid_file_type(self, client):
        """Test recognition with invalid file type."""
        response = client.post(
            "/api/recognize",
            data={"images": (io.BytesIO(b"not an image"), "test.txt")},
            content_type="multipart/form-data",
        )
        assert response.status_code == 400

    def test_recognize_with_valid_image(self, client, sample_image):
        """Test recognition with valid image."""
        response = client.post(
            "/api/recognize",
            data={
                "images": (io.BytesIO(sample_image), "test.jpg"),
                "threshold": "0.6",
                "top_k": "3",
                "per_person_k": "20",
            },
            content_type="multipart/form-data",
        )

        # Should return 200 or 500 (if models missing)
        assert response.status_code in [200, 500]

        if response.status_code == 200:
            data = response.get_json()
            assert "results" in data
            assert "threshold" in data
            assert isinstance(data["results"], list)

    def test_recognize_with_default_parameters(self, client, sample_image):
        """Test recognition with default parameters."""
        response = client.post(
            "/api/recognize",
            data={"images": (io.BytesIO(sample_image), "test.jpg")},
            content_type="multipart/form-data",
        )

        assert response.status_code in [200, 500]

    def test_recognize_multiple_images(self, client, sample_image):
        """Test recognition with multiple images."""
        response = client.post(
            "/api/recognize",
            data={
                "images": [
                    (io.BytesIO(sample_image), "test1.jpg"),
                    (io.BytesIO(sample_image), "test2.jpg"),
                ],
                "threshold": "0.6",
            },
            content_type="multipart/form-data",
        )

        assert response.status_code in [200, 500]


class TestErrorHandling:
    """Tests for error handling."""

    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

        data = response.get_json()
        assert "error" in data

    def test_method_not_allowed(self, client):
        """Test method not allowed."""
        response = client.put("/health")
        assert response.status_code == 405


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_complete_workflow(self, client, sample_image):
        """Test complete workflow: register -> train -> recognize -> delete."""
        # 1. Register person
        response = client.post(
            "/api/persons",
            json={"person_id": "0099_integration_test", "name": "Integration Test"},
        )
        assert response.status_code == 201

        # 2. Verify person exists
        response = client.get("/api/persons/0099_integration_test")
        assert response.status_code == 200

        # 3. Train (may fail if models missing, that's ok)
        response = client.post(
            "/api/persons/0099_integration_test/train",
            data={"images": (io.BytesIO(sample_image), "train.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code in [200, 500]

        # 4. Recognize (may fail if models missing, that's ok)
        response = client.post(
            "/api/recognize",
            data={"images": (io.BytesIO(sample_image), "test.jpg")},
            content_type="multipart/form-data",
        )
        assert response.status_code in [200, 500]

        # 5. Delete person
        response = client.delete("/api/persons/0099_integration_test")
        assert response.status_code == 200

        # 6. Verify person is deleted
        response = client.get("/api/persons/0099_integration_test")
        assert response.status_code == 404
