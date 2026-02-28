#!/bin/bash
USER="hackathon"
IP="34.29.243.10"
REMOTE_PATH="~/IL_Hackathon"

echo "🌱 Deploying Crop Doctor to VM ($IP)..."
echo "🔑 Password: 9c2264b5 (You will be prompted to enter this)"

# 1. Create directory on VM
ssh $USER@$IP "mkdir -p $REMOTE_PATH"

# 2. Zip and upload current directory (skipping heavy/unnecessary folders)
echo "🚀 Uploading code..."
tar --exclude='.git' --exclude='model_cache' --exclude='venv' --exclude='__pycache__' -czf - . | ssh $USER@$IP "tar -xzf - -C $REMOTE_PATH"

# 3. Build and Run on VM
echo "🐳 Building and running Docker on VM..."
ssh $USER@$IP "cd $REMOTE_PATH && docker-compose down && docker-compose up --build -d"

echo "✅ Done! API available at http://$IP:8000"