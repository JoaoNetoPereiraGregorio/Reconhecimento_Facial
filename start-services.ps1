# start-services.ps1
$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

# ── 1. Aguarda o Docker Desktop ───────────────────────────────────────────────
Write-Host "Aguardando Docker Desktop iniciar..." -ForegroundColor Yellow

while ($true) {
    $result = docker info 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker pronto!" -ForegroundColor Green
        break
    }
    Start-Sleep -Seconds 3
}

# ── 2. Sobe os containers ─────────────────────────────────────────────────────
Write-Host "Subindo os servicos..." -ForegroundColor Cyan
docker compose up -d

# ── 3. Aguarda NGINX com curl.exe (mais confiavel no Windows) ─────────────────
$url     = "http://localhost:8082/health"
$timeout = 90
$elapsed = 0

Write-Host "Aguardando NGINX em $url ..." -ForegroundColor Yellow

while ($elapsed -lt $timeout) {
    # curl.exe retorna exit code 0 quando a resposta e HTTP 2xx/3xx
    $null = curl.exe -s -o NUL -w "%{http_code}" $url 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Servicos prontos!" -ForegroundColor Green
        break
    }
    Write-Host "  aguardando... ${elapsed}s" -ForegroundColor DarkGray
    Start-Sleep -Seconds 2
    $elapsed += 2
}

if ($elapsed -ge $timeout) {
    Write-Host "Timeout. Abrindo mesmo assim..." -ForegroundColor Yellow
}

# ── 4. Abre Chrome em modo kiosk ──────────────────────────────────────────────
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
