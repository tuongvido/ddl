# Startup Script for Harmful Content Detection System
# PowerShell script to start all services

Write-Host "[*] Starting Harmful Content Detection System..." -ForegroundColor Green

# Enable BuildKit for faster Docker builds with cache
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Step 1: Start Docker services
Write-Host "`n[*] Starting Docker services..." -ForegroundColor Cyan
Set-Location docker
docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start Docker services" -ForegroundColor Red
    exit 1
}

Write-Host "[OK] Docker services started" -ForegroundColor Green

# Wait for services to be ready
Write-Host "`n[*] Waiting for services to be ready (30 seconds)..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

# Step 2: Check if Python virtual environment exists
Set-Location ..
if (-not (Test-Path "venv")) {
    Write-Host "`n[*] Creating Python virtual environment..." -ForegroundColor Cyan
    python -m venv venv
}

# Activate virtual environment and install dependencies
Write-Host "`n[*] Installing Python dependencies..." -ForegroundColor Cyan
& .\venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Step 3: Display instructions
Write-Host "`n[OK] System is ready!" -ForegroundColor Green
Write-Host "`n[INFO] Next steps:" -ForegroundColor Cyan
Write-Host "1. Place your test video in the data/ folder" -ForegroundColor White
Write-Host "2. Open 3 new PowerShell terminals and run:" -ForegroundColor White
Write-Host "   Terminal 1: cd src; python ./src/producer.py --video ./data/v001_converted.avi --loop" -ForegroundColor Yellow
Write-Host "   Terminal 2: cd src; python ./src/consumer_video.py" -ForegroundColor Yellow
Write-Host "   Terminal 3: cd src; python ./src/consumer_audio.py" -ForegroundColor Yellow
Write-Host "3. Open another terminal for dashboard:" -ForegroundColor White
Write-Host "   Terminal 4: cd src; streamlit run dashboard.py" -ForegroundColor Yellow
Write-Host "`n[INFO] Access points:" -ForegroundColor Cyan
Write-Host "   - Dashboard: http://localhost:8501" -ForegroundColor White
Write-Host "   - Airflow: http://localhost:8080 (admin/admin)" -ForegroundColor White
Write-Host "`n[NOTE] To stop all services, run: .\shutdown.ps1 or cd docker; docker-compose down" -ForegroundColor Yellow
