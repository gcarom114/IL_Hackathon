$User = "hackathon"
$Ip = "34.29.243.10"

Write-Host "Opening tunnel to VM..."
Write-Host "Open http://localhost:8000 in your browser"
Write-Host "Press Ctrl+C to close"
Write-Host ""
ssh -L 8000:localhost:8000 "$User@$Ip"
