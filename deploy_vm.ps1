$User = "hackathon"
$Ip = "34.29.243.10"
$RemotePath = "~/IL_Hackathon"

Write-Host "Deploying Crop Doctor to VM ($Ip)..."
Write-Host "Password: 9c2264b5 (enter when prompted)"

# 1. Create directory on VM
ssh "$User@$Ip" "mkdir -p $RemotePath"

# 2. Upload code (skipping heavy folders)
Write-Host "Uploading code..."
$excludes = "--exclude=.git --exclude=model_cache --exclude=venv --exclude=__pycache__"
cmd /c "tar $excludes -czf - . | ssh $User@$Ip ""tar -xzf - -C $RemotePath"""

# 3. Build and run Docker on VM
Write-Host "Building and running Docker on VM..."
ssh "$User@$Ip" "cd $RemotePath && docker-compose down && docker-compose up --build -d"

Write-Host "Done! App available at http://$Ip`:8000"
