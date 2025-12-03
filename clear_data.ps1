# Script to clear all cached data and reset system
Write-Host "[*] Clearing all cached data..." -ForegroundColor Cyan

# 1. Clear MongoDB collections
Write-Host "`n[*] Clearing MongoDB data..." -ForegroundColor Yellow
docker exec mongodb mongosh -u admin -p admin123 --authenticationDatabase admin livestream_detection --eval "db.video_detections.drop({}); db.audio_detections.drop({}); db.alerts.drop({}); print('MongoDB collections cleared successfully');"

# 2. Delete and recreate Kafka topics
Write-Host "`n[*] Resetting Kafka topics..." -ForegroundColor Yellow

# Delete topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --delete --topic livestream-video 2>$null
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --delete --topic livestream-audio 2>$null

Start-Sleep -Seconds 3

# Recreate topics
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --create --topic livestream-video --partitions 1 --replication-factor 1
docker exec -it kafka kafka-topics --bootstrap-server localhost:9092 --create --topic livestream-audio --partitions 1 --replication-factor 1

Write-Host "`n[OK] All data cleared successfully!" -ForegroundColor Green
Write-Host "`n[INFO] You can now restart producer and consumers with fresh data." -ForegroundColor Cyan
