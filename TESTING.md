# Testing Guide - Coral Vision

This guide explains how to run tests for Coral Vision, including integration tests against Docker containers with pgvector.

## Prerequisites

- Docker and docker compose installed
- Poetry installed
- Development dependencies installed: `poetry install --with dev`

## Quick Start

### Run All Tests (Automated)

The simplest way to run all tests including integration tests:

```bash
# Run all tests (will start Docker containers automatically)
poetry run pytest

# Run only integration tests
poetry run pytest -m integration

# Run fast tests (skip slow tests)
poetry run pytest -m "not slow"

# Run with coverage
poetry run pytest --cov=coral_vision --cov-report=html
```

**How it works:**
- Pytest automatically starts Docker containers if not running
- Waits for services to be ready
- Initializes test database
- Runs tests
- Containers keep running for subsequent test runs
- Use `docker compose down` to stop when done

### Run Specific Test Types

```bash
# Unit tests only (no Docker required)
poetry run pytest tests/test_core.py -v

# API tests (uses in-memory fixtures)
poetry run pytest tests/test_api.py -v

# Docker integration tests only
poetry run pytest tests/test_docker_integration.py -v

# Specific test class
poetry run pytest tests/test_docker_integration.py::TestDockerHealthEndpoint -v

# Specific test
poetry run pytest tests/test_docker_integration.py::TestDockerHealthEndpoint::test_health_check -v
```

## Test Markers

Tests are organized with pytest markers:

- `@pytest.mark.integration` - Requires Docker containers (auto-started)
- `@pytest.mark.slow` - Takes longer to run (model inference)

### Running by Marker

```bash
# Only integration tests
poetry run pytest -m integration

# Exclude integration tests (unit tests only)
poetry run pytest -m "not integration"

# Exclude slow tests
poetry run pytest -m "not slow"

# Integration but not slow
poetry run pytest -m "integration and not slow"
```

## Docker Container Management

### Automatic (Recommended)

Pytest handles containers automatically:

```bash
# Containers start automatically when needed
poetry run pytest -m integration

# Run multiple times - containers stay running
poetry run pytest -m integration
poetry run pytest -m integration

# Stop containers when done
docker compose down
```

### Manual Control

If you prefer manual control:

```bash
# Start containers first
docker compose up -d

# Wait a few seconds for startup
sleep 10

# Run tests (will use existing containers)
poetry run pytest

# Stop containers
docker compose down
```

### Troubleshooting Container Issues

```bash
# Check container status
docker compose ps

# View logs
docker compose logs coral-vision
docker compose logs postgres

# Restart containers
docker compose restart

# Full rebuild
docker compose down
docker compose up -d --build
```

## Environment Variables

Tests use these environment variables (configured in pytest.ini):

```bash
DB_HOST=localhost          # PostgreSQL host
DB_PORT=5432               # PostgreSQL port
DB_NAME=coral_vision_test  # Test database name
DB_USER=coral              # Database user
DB_PASSWORD=coral          # Database password
API_URL=http://localhost:5000  # API endpoint
```

Override for custom setup:

```bash
DB_HOST=192.168.1.100 API_URL=http://192.168.1.100:5000 pytest tests/
```

## Test Coverage

Generate coverage report:

```bash
# Run with coverage
poetry run pytest --cov=coral_vision --cov-report=html tests/

# Open report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Continuous Integration

For CI/CD pipelines:

```yaml
# Example GitHub Actions workflow
- name: Start Docker containers
  run: docker-compose up -d

- name: Wait for services
  run: |
    sleep 10
    docker-compose exec -T postgres pg_isready -U coral
    curl --retry 10 --retry-delay 2 http://localhost:5000/health

- name: Run tests
  run: poetry run pytest tests/ -v --cov=coral_vision

- name: Stop containers
  run: docker-compose down
```

## Troubleshooting

### Database Connection Failed

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Check logs
docker-compose logs postgres

# Restart database
docker-compose restart postgres
```

### API Not Responding

```bash
# Check API is running
docker-compose ps coral-vision

# Check logs
docker-compose logs coral-vision

# Restart API
docker-compose restart coral-vision
```

### Tests Timeout

Increase wait times in conftest.py:

```python
max_retries = 60  # Increase from 30
retry_delay = 2   # Increase from 1
```

### Permission Errors

```bash
# Fix data directory permissions
sudo chown -R $USER:$USER ./data

# Rebuild containers
docker-compose down
docker-compose up --build -d
```

### Port Already in Use

```bash
# Change ports in docker-compose.yml
ports:
  - "5432:5432"  # PostgreSQL
  - "5001:5000"  # API

# Update environment variables
export DB_PORT=5432
export API_URL=http://localhost:5001
```

## Test Structure

```
tests/
├── conftest.py                    # Shared fixtures
├── test_api.py                    # Flask API tests (unit)
├── test_core.py                   # Core functionality tests (unit)
└── test_docker_integration.py     # Docker integration tests
```

### Key Fixtures

- `temp_data_dir` - Temporary directory for tests
- `storage_backend` - pgvector storage connected to Docker
- `app` - Flask application instance
- `client` - Flask test client
- `docker_client` - HTTP client for Docker API
- `wait_for_db` - Wait for PostgreSQL to be ready
- `wait_for_api` - Wait for API to be ready
- `registered_person` - Test person for unit tests
- `registered_person_docker` - Test person for integration tests
- `sample_image` - Sample JPEG image for testing

## Best Practices

1. **Isolate tests**: Each test should clean up after itself
2. **Use fixtures**: Reuse common setup with pytest fixtures
3. **Mark appropriately**: Use `@pytest.mark` for test organization
4. **Test real scenarios**: Integration tests should match production use
5. **Keep tests fast**: Mock external services when possible
6. **Clean up data**: Remove test data in teardown

## Writing New Tests

### Unit Test Example

```python
def test_my_feature(client):
    """Test my feature."""
    response = client.get("/api/my-endpoint")
    assert response.status_code == 200
```

### Integration Test Example

```python
@pytest.mark.integration
def test_my_feature_docker(docker_client):
    """Test my feature against Docker."""
    response = docker_client.get("/api/my-endpoint")
    assert response.status_code == 200
```

### Slow Test Example

```python
@pytest.mark.integration
@pytest.mark.slow
def test_model_inference(docker_client, sample_image):
    """Test model inference (slow)."""
    response = docker_client.post(
        "/api/recognize",
        files={"images": ("test.jpg", sample_image, "image/jpeg")}
    )
    assert response.status_code == 200
```

## Summary

- ✅ Use `./run_tests.sh` for automated testing
- ✅ Integration tests verify Docker deployment
- ✅ Tests use real pgvector database
- ✅ Fixtures handle setup/teardown
- ✅ Markers organize test types
- ✅ Coverage reports show test quality

Happy testing! 🧪
