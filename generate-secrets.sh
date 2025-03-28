#!/bin/bash

# Create secrets directory if it doesn't exist
mkdir -p secrets

# Generate API key (random string)
echo "touchless_gesture_api_$(openssl rand -hex 8)" > secrets/api_key.txt

# Generate app secret (random string)
echo "app_secret_$(openssl rand -hex 16)" > secrets/app_secret.txt

echo "Secrets generated successfully:"
echo "- API Key: $(cat secrets/api_key.txt)"
echo "- App Secret: $(cat secrets/app_secret.txt)"