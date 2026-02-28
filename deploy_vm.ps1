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

# 3. Start server on VM in background
Write-Host "Starting server on VM..."
ssh "$User@$Ip" "cd $RemotePath/app && nohup /home/hackathon/.local/bin/uvicorn api:app --host 0.0.0.0 --port 8000 > ~/server.log 2>&1 &"

# 4. Open SSH tunnel - access app at http://localhost:8000
Write-Host ""
Write-Host "App is running! Opening tunnel..."
Write-Host "Open http://localhost:8000 in your browser"
Write-Host "Press Ctrl+C to close the tunnel"
Write-Host ""
ssh -L 8000:localhost:8000 "$User@$Ip"
