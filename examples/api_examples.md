# Flask Web API Examples

This document provides example API calls for the Coral Vision web API.

## API Documentation

### Viewing the OpenAPI/Swagger Documentation

**🚀 Built-in Swagger UI (Easiest - No Installation Needed!)**

The Flask server includes a built-in Swagger UI. Just start the server and visit:

```bash
# Start the server
coral-vision serve --host 0.0.0.0 --port 5000

# Open your browser to:
# http://localhost:5000/docs
```

**Available endpoints:**
- **`/docs`** - Interactive Swagger UI documentation
- **`/openapi.json`** - OpenAPI specification (JSON)
- **`/`** - API root with links to documentation

---

### Alternative Documentation Viewers

If you want to use external tools, the complete API specification is available in [openapi.json](../openapi.json).

**Option 1: Online Swagger Editor (Easiest)**
1. Go to [https://editor.swagger.io/](https://editor.swagger.io/)
2. Click `File` → `Import file`
3. Upload the `openapi.json` file from the project root
4. Browse the interactive documentation

**Option 2: Swagger UI with Docker**
```bash
# From project root
docker run -p 8080:8080 -e SWAGGER_JSON=/openapi.json -v $(pwd)/openapi.json:/openapi.json swaggerapi/swagger-ui

# Open browser to http://localhost:8080
```

**Option 3: Swagger UI with NPM**
```bash
# Install swagger-ui-watcher globally
npm install -g swagger-ui-watcher

# Serve the OpenAPI spec
swagger-ui-watcher openapi.json

# Opens automatically at http://localhost:8000
```

**Option 4: Add Swagger UI Endpoint to Flask (Recommended for Development)**

Install the Flask-CORS and flasgger packages:
```bash
poetry add flasgger flask-cors
```

Then access the Swagger UI at: `http://localhost:5000/apidocs`

**Option 5: Redoc (Alternative Documentation Viewer)**
```bash
npx @redocly/cli preview-docs openapi.json

# Opens at http://localhost:8080
```

**Option 6: Import to Postman/Insomnia**
- **Postman**: Import → Upload Files → Select `openapi.json`
- **Insomnia**: Create → Import From → File → Select `openapi.json`

## Starting the Server

```bash
# Start with CPU models
coral-vision serve --host 0.0.0.0 --port 5000 --data-dir ./data

# Start with Edge TPU acceleration
coral-vision serve --host 0.0.0.0 --port 5000 --data-dir ./data --use-edgetpu

# Start in debug mode
coral-vision serve --debug
```

## API Endpoints

### Health Check

```bash
curl http://localhost:5000/health
```

**Response:**
```json
{
  "status": "healthy",
  "use_edgetpu": false
}
```

---

### List All Persons

```bash
curl http://localhost:5000/api/persons
```

**Response:**
```json
{
  "persons": [
    {
      "person_id": "0001_alfred_maier",
      "name": "Alfred Maier"
    },
    {
      "person_id": "0002_john_doe",
      "name": "John Doe"
    }
  ]
}
```

---

### Register New Person

```bash
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{
    "person_id": "0003_jane_smith",
    "name": "Jane Smith"
  }'
```

**Response:**
```json
{
  "person_id": "0003_jane_smith",
  "name": "Jane Smith",
  "message": "Person registered successfully. Upload training images next."
}
```

**Error Response (if person exists):**
```json
{
  "error": "Person 0003_jane_smith already exists"
}
```

---

### Get Person Details

```bash
curl http://localhost:5000/api/persons/0001_alfred_maier
```

**Response:**
```json
{
  "person_id": "0001_alfred_maier",
  "name": "Alfred Maier",
  "embedding_count": 25,
  "face_count": 25
}
```

---

### Train Person (Upload Training Images)

```bash
curl -X POST http://localhost:5000/api/persons/0001_alfred_maier/train \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg" \
  -F "min_score=0.95" \
  -F "max_faces=1"
```

**Response:**
```json
{
  "person_id": "0001_alfred_maier",
  "name": "Alfred Maier",
  "use_edgetpu": false,
  "input": "/tmp/tmpxyz123",
  "processed_images": 3,
  "saved_faces": 3,
  "saved_embeddings": 3,
  "skipped_no_face": 0,
  "skipped_low_score": 0,
  "output_dir": "data/scanned_people/0001_alfred_maier"
}
```

**Parameters:**
- `images` (required): One or more image files
- `min_score` (optional): Minimum detection confidence (default: 0.95)
- `max_faces` (optional): Max faces per image to train (default: 1)

---

### Recognize Faces

```bash
curl -X POST http://localhost:5000/api/recognize \
  -F "images=@test_photo.jpg" \
  -F "threshold=0.6" \
  -F "top_k=3" \
  -F "per_person_k=20"
```

**Response:**
```json
{
  "use_edgetpu": false,
  "input": "/tmp/tmpxyz456",
  "threshold": 0.6,
  "top_k": 3,
  "per_person_k": 20,
  "results": [
    {
      "image_path": "/tmp/tmpxyz456/test_photo.jpg",
      "faces": [
        {
          "bbox": {
            "xmin": 150,
            "ymin": 100,
            "xmax": 350,
            "ymax": 300
          },
          "score": 0.98,
          "matches": [
            {
              "person_id": "0001_alfred_maier",
              "name": "Alfred Maier",
              "distance": 0.42
            },
            {
              "person_id": "0002_john_doe",
              "name": "John Doe",
              "distance": 0.78
            }
          ],
          "predicted": {
            "person_id": "0001_alfred_maier",
            "name": "Alfred Maier",
            "distance": 0.42
          },
          "accepted": true,
          "threshold": 0.6
        }
      ]
    }
  ]
}
```

**Parameters:**
- `images` (required): One or more image files
- `threshold` (optional): Recognition threshold (default: 0.6)
- `top_k` (optional): Return top K matches per face (default: 3)
- `per_person_k` (optional): Compare against K embeddings per person (default: 20)

---

### Delete Person

```bash
curl -X DELETE http://localhost:5000/api/persons/0003_jane_smith
```

**Response:**
```json
{
  "message": "Person 0003_jane_smith deleted successfully"
}
```

---

## Python Examples

### Using `requests` library

```python
import requests

# Base URL
BASE_URL = "http://localhost:5000"

# Register a new person
response = requests.post(
    f"{BASE_URL}/api/persons",
    json={
        "person_id": "0004_bob_wilson",
        "name": "Bob Wilson"
    }
)
print(response.json())

# Upload training images
files = [
    ("images", open("photo1.jpg", "rb")),
    ("images", open("photo2.jpg", "rb")),
    ("images", open("photo3.jpg", "rb")),
]
data = {
    "min_score": "0.95",
    "max_faces": "1"
}
response = requests.post(
    f"{BASE_URL}/api/persons/0004_bob_wilson/train",
    files=files,
    data=data
)
print(response.json())

# Recognize faces
files = [("images", open("test.jpg", "rb"))]
data = {
    "threshold": "0.6",
    "top_k": "3"
}
response = requests.post(
    f"{BASE_URL}/api/recognize",
    files=files,
    data=data
)
print(response.json())

# List all persons
response = requests.get(f"{BASE_URL}/api/persons")
print(response.json())

# Get person details
response = requests.get(f"{BASE_URL}/api/persons/0004_bob_wilson")
print(response.json())

# Delete person
response = requests.delete(f"{BASE_URL}/api/persons/0004_bob_wilson")
print(response.json())
```

---

## Complete Workflow Example

### 1. Register a person
```bash
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -d '{"person_id": "0005_alice_jones", "name": "Alice Jones"}'
```

### 2. Upload 20+ training images
```bash
curl -X POST http://localhost:5000/api/persons/0005_alice_jones/train \
  -F "images=@alice_1.jpg" \
  -F "images=@alice_2.jpg" \
  -F "images=@alice_3.jpg" \
  # ... more images
```

### 3. Recognize faces in new images
```bash
curl -X POST http://localhost:5000/api/recognize \
  -F "images=@group_photo.jpg" \
  -F "threshold=0.6"
```

---

## Error Responses

### 400 Bad Request
```json
{
  "error": "Both person_id and name are required"
}
```

### 404 Not Found
```json
{
  "error": "Person 0999_unknown not found"
}
```

### 409 Conflict
```json
{
  "error": "Person 0001_alfred_maier already exists"
}
```

### 413 Payload Too Large
```json
{
  "error": "File too large. Max size is 16MB"
}
```

### 500 Internal Server Error
```json
{
  "error": "Model not found: data/models/detector.tflite"
}
```

---

## Testing with Multiple Images

```bash
# Upload multiple images for training
curl -X POST http://localhost:5000/api/persons/0001_alfred_maier/train \
  -F "images=@./training_photos/img1.jpg" \
  -F "images=@./training_photos/img2.jpg" \
  -F "images=@./training_photos/img3.jpg" \
  -F "images=@./training_photos/img4.jpg" \
  -F "images=@./training_photos/img5.jpg"

# Recognize faces in multiple images
curl -X POST http://localhost:5000/api/recognize \
  -F "images=@./test_photos/test1.jpg" \
  -F "images=@./test_photos/test2.jpg" \
  -F "images=@./test_photos/test3.jpg"
```

---

## Production Deployment

For production, use a production WSGI server like Gunicorn:

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "coral_vision.web.app:create_app()"
```

Or use Docker:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . /app

RUN pip install poetry && poetry install --no-dev

EXPOSE 5000

CMD ["poetry", "run", "coral-vision", "serve", "--host", "0.0.0.0"]
```
