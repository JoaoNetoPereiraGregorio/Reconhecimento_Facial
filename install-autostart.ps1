# install-autostart.ps1
# Registra o sistema para iniciar automaticamente com o Windows.
# Execute UMA VEZ como Administrador:
#   Right-click → "Run as administrator"

param(
    [string]$ProjectPath = (Split-Path -Parent $MyInvocation.MyCommand.Path)
)

$taskName   = "DocFace - Autostart"
$scriptPath = Join-Path $ProjectPath "start-services.ps1"

if (-not (Test-Path $scriptPath)) {
    Write-Host "ERRO: start-services.ps1 nao encontrado em: $scriptPath" -ForegroundColor Red
    exit 1
}

# Remove tarefa anterior se existir
if (Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue) {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
    Write-Host "Tarefa anterior removida." -ForegroundColor Yellow
}

# Acao: executa o PowerShell com o script de start
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -ExecutionPolicy Bypass -File `"$scriptPath`""

# Gatilho: ao fazer login no Windows (qualquer usuario)
$trigger = New-ScheduledTaskTrigger -AtLogOn

# Configuracoes: roda com privilégios mais altos (necessario para Docker)
$settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 1)

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

Register-ScheduledTask `
    -TaskName  $taskName `
    -Action    $action `
    -Trigger   $trigger `
    -Settings  $settings `
    -Principal $principal `
    -Force

Write-Host ""
Write-Host "Autostart registrado com sucesso!" -ForegroundColor Green
Write-Host "Tarefa: '$taskName'" -ForegroundColor Cyan
Write-Host "Script: $scriptPath" -ForegroundColor Cyan
Write-Host ""
Write-Host "Para remover: Unregister-ScheduledTask -TaskName '$taskName' -Confirm:`$false"
