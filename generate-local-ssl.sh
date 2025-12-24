#!/bin/bash
# Generate self-signed SSL certificate for local development

echo "Generating self-signed SSL certificate for local development..."

# Create certs directory if it doesn't exist
mkdir -p certs

# Generate private key
openssl genrsa -out certs/key.pem 2048

# Get local IP address (optional, for mobile access)
LOCAL_IP=$(hostname -I | awk '{print $1}' 2>/dev/null || echo "")

# Generate certificate with localhost and local IP
if [ -n "$LOCAL_IP" ]; then
    openssl req -new -x509 -key certs/key.pem -out certs/cert.pem -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1,IP:$LOCAL_IP"
else
    openssl req -new -x509 -key certs/key.pem -out certs/cert.pem -days 365 \
        -subj "/C=US/ST=State/L=City/O=Organization/CN=localhost" \
        -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1"
fi

echo "✅ SSL certificate generated in certs/ directory"
echo ""
echo "To use HTTPS, update your Flask app to use these certificates:"
echo "  - Certificate: certs/cert.pem"
echo "  - Private Key: certs/key.pem"
echo ""
echo "Or use Flask with SSL context:"
echo "  app.run(ssl_context=('certs/cert.pem', 'certs/key.pem'))"

