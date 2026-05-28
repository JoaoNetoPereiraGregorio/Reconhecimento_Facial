# ── 1. Define o caminho local ──────────────────────────────────────────────────
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

# ── 2. Abre Chrome em modo kiosk ──────────────────────────────────────────────
$chromePaths = @(
    "C:\Program Files\Google\Chrome\Application\chrome.exe",
    "C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "$env:LOCALAPPDATA\Google\Chrome\Application\chrome.exe"
)

$chrome = $chromePaths | Where-Object { Test-Path $_ } | Select-Object -First 1

if (-not $chrome) {
    Write-Host "Chrome nao encontrado. Abrindo no navegador padrao..." -ForegroundColor Yellow
    Start-Process "http://localhost:8082"
} else {
    Write-Host "Abrindo kiosk: $chrome" -ForegroundColor Cyan
    Start-Process $chrome -ArgumentList `
        "--kiosk", `
        "--no-first-run", `
        "--disable-infobars", `
        "--disable-session-crashed-bubble", `
        "--disable-restore-session-state", `
        "--autoplay-policy=no-user-gesture-required", `
        "http://localhost:8082"
}
