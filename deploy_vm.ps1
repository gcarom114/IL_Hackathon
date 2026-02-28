$User = "hackathon"
$Ip = "34.29.243.10"
$RemotePath = "~/IL_Hackathon"

Write-Host "🌱 Deploying Crop Doctor to VM ($Ip)..."
Write-Host "🔑 Password: 9c2264b5 (You will be prompted to enter this)"

# 1. Create directory on VM
ssh $User@$Ip "mkdir -p $RemotePath"

# 2. Zip and upload current directory (skipping heavy/unnecessary folders)
Write-Host "🚀 Uploading code..."
tar --exclude='.git' --exclude='model_cache' --exclude='venv' --exclude='__pycache__' -czf - . | ssh $User@$Ip "tar -xzf - -C $RemotePath"

# 3. Build and Run on VM
Write-Host "🐳 Building and running Docker on VM..."
ssh $User@$Ip "cd $RemotePath && docker-compose down && docker-compose up --build -d"

Write-Host "✅ Done! API available at http://$Ip:8000"