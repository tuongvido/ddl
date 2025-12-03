# Script to rebuild Airflow with dependencies and restart
Write-Host "[*] Rebuilding Airflow containers with dependencies..." -ForegroundColor Cyan

cd docker

# Enable BuildKit for faster builds with cache
$env:DOCKER_BUILDKIT=1
$env:COMPOSE_DOCKER_CLI_BUILD=1

# Stop existing containers
Write-Host "[*] Stopping existing Airflow containers..." -ForegroundColor Yellow
docker-compose stop airflow-webserver airflow-scheduler airflow-worker

# Remove old containers
Write-Host "[*] Removing old containers..." -ForegroundColor Yellow
docker-compose rm -f airflow-webserver airflow-scheduler airflow-worker

# Build new image with dependencies (BuildKit will use cache mount)
Write-Host "[*] Building custom Airflow image with cache..." -ForegroundColor Cyan
docker-compose build airflow-webserver

# Start services
Write-Host "[*] Starting Airflow services..." -ForegroundColor Green
docker-compose up -d airflow-webserver airflow-scheduler airflow-worker

Write-Host ""
Write-Host "[OK] Rebuild complete!" -ForegroundColor Green
Write-Host ""
Write-Host "Wait 30 seconds for services to start, then:" -ForegroundColor Cyan
Write-Host "  1. Open Airflow UI: http://localhost:8080" -ForegroundColor White
Write-Host "  2. Find DAG: run_streaming_pipeline" -ForegroundColor White
Write-Host "  3. Click Trigger to start Producer + Consumers" -ForegroundColor White
Write-Host ""
