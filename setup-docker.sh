#!/bin/bash
# Quick setup script for Coral Vision with Docker

set -e

echo "🚀 Coral Vision Docker Setup"
echo "=============================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and docker-compose found"
echo ""

# Check if data directory exists
if [ ! -d "./data/models" ]; then
    echo "📁 Creating data directory structure..."
    mkdir -p data/models
    echo "✅ Created: ./data/models"
    echo ""
fi

# Check if models exist
MODEL_FILES=(
    "ssd_mobilenet_v2_face_quant_postprocess.tflite"
    "ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite"
    "Mobilenet1_triplet1589223569_triplet_quant.tflite"
    "Mobilenet1_triplet1589223569_triplet_quant_edgetpu.tflite"
)

MISSING_MODELS=0
for model in "${MODEL_FILES[@]}"; do
    if [ ! -f "./data/models/$model" ]; then
        echo "⚠️  Missing model: $model"
        MISSING_MODELS=$((MISSING_MODELS + 1))
    fi
done

if [ $MISSING_MODELS -gt 0 ]; then
    echo ""
    echo "⚠️  $MISSING_MODELS model file(s) missing in ./data/models/"
    echo "Please download the required TensorFlow Lite models before starting."
    echo ""
    read -p "Do you want to continue anyway? (y/N): " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ All required model files found"
fi

echo ""
echo "🐳 Starting services with docker-compose..."
echo ""

# Start services
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo ""
    echo "✅ Services started successfully!"
    echo ""
    echo "📊 Service Status:"
    docker-compose ps
    echo ""
    echo "🌐 Web Interface: http://localhost:5000"
    echo "💚 Health Check:  http://localhost:5000/health"
    echo "📖 API Docs:      http://localhost:5000/docs"
    echo "🗄️  Database:      postgresql://coral:coral@localhost:5432/coral_vision"
    echo ""
    echo "📝 View logs with: docker-compose logs -f"
    echo "🛑 Stop services:  docker-compose down"
    echo ""
    echo "Happy face recognition! 😊"
else
    echo ""
    echo "❌ Failed to start services. Check logs:"
    docker-compose logs
    exit 1
fi
