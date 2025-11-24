# Shutdown Script for Harmful Content Detection System

Write-Host "[*] Stopping Harmful Content Detection System..." -ForegroundColor Yellow

# Stop Docker services
Write-Host "`n[*] Stopping Docker services..." -ForegroundColor Cyan
Set-Location docker
docker-compose down

if ($LASTEXITCODE -eq 0) {
    Write-Host "[OK] Docker services stopped" -ForegroundColor Green
} else {
    Write-Host "[WARNING] Some services may not have stopped cleanly" -ForegroundColor Yellow
}

Write-Host "`n[OK] System shutdown complete" -ForegroundColor Green
Write-Host "[INFO] To start again, run: .\startup.ps1" -ForegroundColor Cyan
