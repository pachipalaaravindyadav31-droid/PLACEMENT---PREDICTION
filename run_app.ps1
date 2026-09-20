# PlacementPredict Unified Launcher
$pythonExe = "$env:LOCALAPPDATA\Python\bin\python.exe"

if (Test-Path $pythonExe) {
    Write-Host "[*] Launching PlacementPredict using: $pythonExe" -ForegroundColor Cyan
    & $pythonExe app.py
} else {
    Write-Host "[*] Launching PlacementPredict using system python..." -ForegroundColor Cyan
    python app.py
}
