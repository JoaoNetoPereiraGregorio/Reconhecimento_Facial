# stop-services.ps1
# Para todos os containers do projeto de forma limpa.
# Pode ser chamado manualmente ou vinculado a um atalho.

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectDir

Write-Host "Encerrando servicos..." -ForegroundColor Yellow
docker compose down
Write-Host "Servicos encerrados." -ForegroundColor Green
