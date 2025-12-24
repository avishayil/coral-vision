# Coral Vision

> Offline face recognition with Google Coral Edge TPU acceleration and pgvector database backend

A high-performance face recognition system using TensorFlow Lite models optimized for Google Coral Edge TPU. Features PostgreSQL with pgvector for scalable, fast vector similarity search.

## Features

- **Face Detection**: SSD MobileNet V2 for accurate face detection
- **Face Embedding**: MobileNet triplet model (192-D vectors)
- **Vector Search**: pgvector with HNSW indexing for O(log n) similarity search
- **Scalable**: PostgreSQL backend handles millions of embeddings
- **Edge TPU Support**: 5× faster inference with Google Coral Edge TPU
- **RESTful API**: Complete HTTP API for integration
- **Interactive UI**: Web interface for enrollment and recognition
- **Docker Ready**: One-command deployment with docker-compose

## Architecture

```
┌─────────────────────┐
│  Flask Web API      │
│  + Interactive UI   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐      ┌──────────────────┐
│  TensorFlow Lite    │      │   PostgreSQL     │
│  Face Detection +   │◄────►│   + pgvector     │
│  Embedding Models   │      │   (HNSW Index)   │
└─────────────────────┘      └──────────────────┘
           │
           ▼
┌─────────────────────┐
│  Google Coral TPU   │
│  (Optional 5× boost)│
└─────────────────────┘
```

## Quick Start with Docker

### Prerequisites
- Docker & Docker Compose
- Model files (see below)

### 1. Automated Setup

```bash
./setup-docker.sh
```

### 2. Manual Setup

```bash
# Start services with API key (required)
API_KEY=your-secret-key-here docker-compose up -d

# Or set in .env file or docker-compose.yml
export API_KEY=your-secret-key-here
docker-compose up -d

# Check status
curl http://localhost:5000/health
```

**Note:** The `API_KEY` environment variable is **required**. The server will not start without it. Set a strong, unique API key for security.

### 3. Access the UI

Open http://localhost:5000 in your browser to:
- **Configure API Key**: Enter your API key in the configuration section at the top of the page
- Enroll people with drag-and-drop images
- Recognize faces in real-time
- View match scores and confidence levels

**Important:** You must enter the same API key that was used to start the server. The API key is stored in your browser's localStorage and automatically included in all API requests.

See [QUICKSTART.md](QUICKSTART.md) for detailed guide.

## Local Development

### Prerequisites

- Python 3.9 (required for TensorFlow Lite compatibility)
- PostgreSQL 16 with pgvector extension
- Google Coral Edge TPU (optional, for acceleration)
- Poetry for dependency management

### Install pgvector on PostgreSQL

```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS with Homebrew
brew install pgvector

# Or use Docker (easier)
docker run -d --name pgvector -p 5432:5432 \
  -e POSTGRES_PASSWORD=yourpassword \
  pgvector/pgvector:pg16
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/avishayil/coral-vision.git
cd coral-vision
```

### Install Python Dependencies

Using Poetry (recommended):

```bash
poetry install
poetry shell
```

Or using pip:

```bash
pip install -e .
```

### Set up Environment

```bash
# Copy environment template
cp .env.example .env

# Edit .env with your database credentials
nano .env
```

### Initialize Database

```bash
# Initialize tables and indexes
coral-vision init
```

### Download Models

Place TensorFlow Lite models in `data/models/`:

- `ssd_mobilenet_v2_face_quant_postprocess.tflite` (CPU face detection)
- `ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite` (Edge TPU face detection)
- `Mobilenet1_triplet1589223569_triplet_quant.tflite` (CPU face embedding)
- `Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite` (Edge TPU face embedding)

## Usage

### Web Interface

```bash
# Start the web server
coral-vision serve --port 5000

# Open http://localhost:5000
```

The web UI provides:
- **Enroll Tab**: Upload images to enroll new people
- **Recognize Tab**: Upload images to identify faces
- **Results**: View match scores and confidence levels

### Command-Line Interface

#### Initialize Database

```bash
coral-vision init
```

#### Enroll a Person

```bash
coral-vision enroll \
  --person-id "0001_alfred_maier" \
  --name "Alfred Maier" \
  --images ./input/alfred \
  --use-edgetpu \
  --min-score 0.95 \
  --max-faces 1
```

**Options:**
- `--person-id`: Unique identifier (e.g., `0001_john_doe`)
- `--name`: Display name
- `--images`: Path to folder/image for enrollment
- `--use-edgetpu`: Use Edge TPU acceleration
- `--min-score`: Minimum detection confidence (default: 0.95)
- `--max-faces`: Max faces per image (default: 1)

#### Recognize Faces

```bash
coral-vision recognize \
  --input ./images_to_recognize \
  --use-edgetpu \
  --threshold 0.6 \
  --top-k 3
  --per-person-k 20 \
  --say \
  --output-json results.json
```

**Options:**
- `--input`: Path to folder or single image file
- `--use-edgetpu`: Use Edge TPU acceleration
- `--threshold`: Recognition threshold - lower is stricter (default: 0.6)
- `--top-k`: Return top K matches per face (default: 3)
- `--per-person-k`: Compare against K best embeddings per person (default: 20)
- `--say`: Use TTS to greet recognized people (optional)
- `--output-json`: Save results to JSON file (optional)

**Example output:**
```json
{
  "use_edgetpu": true,
  "input": "./test_images",
  "threshold": 0.6,
  "results": [
    {
      "image_path": "./test_images/photo1.jpg",
      "faces": [
        {
          "bbox": {"xmin": 120, "ymin": 80, "xmax": 220, "ymax": 180},
          "score": 0.98,
          "matches": [
            {"person_id": "0001_alfred_maier", "name": "Alfred Maier", "distance": 0.42}
          ],
          "predicted": {"person_id": "0001_alfred_maier", "name": "Alfred Maier", "distance": 0.42},
          "accepted": true,
          "threshold": 0.6
        }
      ]
    }
  ]
}
```

### Web Server

Start the REST API server:

```bash
# Set API key (required)
export API_KEY=your-secret-key-here

# Start server (CPU mode)
coral-vision serve --host 0.0.0.0 --port 5000

# Start with Edge TPU acceleration
coral-vision serve --host 0.0.0.0 --port 5000 --use-edgetpu
```

**Note:** The `API_KEY` environment variable is **required**. The server will not start without it.

**API Endpoints:**

| Method | Endpoint | Description | Authentication |
|--------|----------|-------------|----------------|
| GET | `/health` | Health check | None (public) |
| GET | `/` | Web UI | None (public) |
| GET | `/api/persons` | List all enrolled persons | **Required** |
| POST | `/api/persons` | Register new person | **Required** |
| GET | `/api/persons/{id}` | Get person details | **Required** |
| DELETE | `/api/persons/{id}` | Delete person | **Required** |
| POST | `/api/persons/{id}/train` | Upload training images | **Required** |
| POST | `/api/recognize` | Recognize faces in images | **Required** |
| POST | `/api/process_frame` | Process single frame (camera) | **Required** |

**Authentication:**

All `/api/*` endpoints require API key authentication. The homepage (`/`) and health endpoint (`/health`) are publicly accessible.

You can send the API key in two ways:
1. **X-API-Key header** (recommended):
   ```bash
   curl -H "X-API-Key: your-secret-key-here" ...
   ```
2. **Authorization Bearer token**:
   ```bash
   curl -H "Authorization: Bearer your-secret-key-here" ...
   ```

**Example API Usage:**

```bash
# Set your API key
API_KEY="your-secret-key-here"

# Register a person
curl -X POST http://localhost:5000/api/persons \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $API_KEY" \
  -d '{"person_id": "0001_john_doe", "name": "John Doe"}'

# Upload training images
curl -X POST http://localhost:5000/api/persons/0001_john_doe/train \
  -H "X-API-Key: $API_KEY" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg"

# Recognize faces
curl -X POST http://localhost:5000/api/recognize \
  -H "X-API-Key: $API_KEY" \
  -F "images=@test.jpg" \
  -F "threshold=0.6"
```

```bash
# View the OpenAPI spec
cat openapi.json

# Or serve it with a Swagger UI (requires npm)
npx swagger-ui-watcher openapi.json
```

## Troubleshooting

**Server won't start - "API_KEY environment variable is required":**
- Set the `API_KEY` environment variable before starting the server
- For Docker: `API_KEY=your-key docker-compose up -d`
- For local: `export API_KEY=your-key && coral-vision serve`

**API requests return 401 "Invalid or missing API key":**
- Ensure you've entered the API key in the web UI (top of homepage)
- Verify the API key matches the one used to start the server
- Check that you're sending the `X-API-Key` header in API requests
- The API key is stored in browser localStorage - try clearing it and re-entering

**Model not found:**
- Ensure all `.tflite` files are in `data/models/`
- Check filenames match exactly (including `_edgetpu` suffix)

**No face detected:**
- Lower `--min-score` threshold (try 0.85)
- Ensure clear, front-facing faces in images
- Check image quality and lighting

**Poor recognition:**
- Enroll 10-20+ images per person
- Vary angles, lighting, expressions
- Adjust `--threshold` parameter
- Use `--use-edgetpu` for consistency

**Database connection errors:**
- Verify PostgreSQL is running (`docker-compose ps`)
- Check `.env` database credentials
- Ensure pgvector extension is installed

## Documentation

- [TESTING.md](TESTING.md) - Testing guide and instructions

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project uses TensorFlow Lite models. Check individual model licenses.

## Acknowledgments

- @bergerda1 for the base repo - https://github.com/bergerda1/PiFaceRecognition
- Google Coral for Edge TPU support
- TensorFlow team for TFLite Runtime
- PostgreSQL team for pgvector extension
