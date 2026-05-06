param(
    [switch]$SkipOllamaCheck,
    [switch]$SkipFirewall,
    [switch]$Uninstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Runtime data stored in ProgramData so SYSTEM account can read/write it
$DataDir        = "C:\ProgramData\SharedOllama"
$VenvDir        = Join-Path $DataDir ".venv"
$VenvPython     = Join-Path $VenvDir "Scripts\python.exe"
$RepoRoot       = Split-Path -Parent $PSScriptRoot
$RuntimeConfig  = Join-Path $RepoRoot "monitor\runtime_config.json"
$ControlScript  = Join-Path $RepoRoot "scripts\ollama_control_windows.ps1"
$ProxyTaskName  = "SharedOllamaProxy"
$AdminTaskName  = "SharedOllamaAdmin"
$OllamaPort     = 11435   # Ollama backend; proxy exposes 11434 to clients

function Write-Log { param([string]$Message); Write-Host "[sharedollama-setup] $Message" }

function Test-IsAdmin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    ([Security.Principal.WindowsPrincipal]$id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Remove-Task { param([string]$Name)
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
}

# ── Uninstall ────────────────────────────────────────────────────────────────
if ($Uninstall) {
    Write-Log "Uninstalling SharedOllama Windows native services"
    Remove-Task $ProxyTaskName
    Remove-Task $AdminTaskName
    [System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_RUNTIME_CONFIG",       $null, "Machine")
    [System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_OLLAMA_CONTROL_SCRIPT", $null, "Machine")
    Write-Log "Services removed. Venv at $VenvDir was not deleted - remove manually if desired."
    return
}

# ── Admin check ──────────────────────────────────────────────────────────────
if (-not (Test-IsAdmin)) {
    throw "Run this script as Administrator."
}

Write-Log "Repository root : $RepoRoot"
Write-Log "Data directory  : $DataDir"

# ── Ollama ───────────────────────────────────────────────────────────────────
if (-not $SkipOllamaCheck) {
    $ollamaFound = (Get-Command ollama -ErrorAction SilentlyContinue) -or
                   (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"))

    if (-not $ollamaFound) {
        Write-Log "Ollama not found. Attempting install via winget..."
        $winget = Get-Command winget -ErrorAction SilentlyContinue
        if ($winget) {
            & winget install Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
            Write-Log "Ollama installed. It will start automatically at next logon."
        }
        else {
            Write-Log "winget not available. Please install Ollama manually from https://ollama.com/download"
            Write-Log "Re-run this script after installing Ollama."
            exit 1
        }
    }
    else {
        Write-Log "Ollama found"
    }
}

# Set OLLAMA_HOST for the current user so Ollama starts on port 11435 at logon,
# leaving port 11434 free for the proxy.
Write-Log "Setting OLLAMA_HOST=127.0.0.1:$OllamaPort (user environment)"
[System.Environment]::SetEnvironmentVariable("OLLAMA_HOST", "127.0.0.1:$OllamaPort", "User")

# ── Python ───────────────────────────────────────────────────────────────────
$pythonExe = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
if (-not $pythonExe) {
    $pythonExe = Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
}

if (-not $pythonExe) {
    Write-Log "Python not found. Attempting install via winget..."
    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if ($winget) {
        & winget install Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
        $env:PATH = [System.Environment]::GetEnvironmentVariable("PATH", "Machine") + ";" +
                    [System.Environment]::GetEnvironmentVariable("PATH", "User")
        $pythonExe = Get-Command python -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
        if (-not $pythonExe) {
            throw "Python installed but not found in PATH. Open a new terminal and rerun."
        }
    }
    else {
        throw "Python not found and winget is unavailable. Install Python from https://python.org and rerun."
    }
}

Write-Log "Python: $pythonExe"

# ── Virtual environment ───────────────────────────────────────────────────────
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

if (-not (Test-Path $VenvPython)) {
    Write-Log "Creating virtual environment in $VenvDir"
    & $pythonExe -m venv $VenvDir
}
else {
    Write-Log "Virtual environment already exists"
}

Write-Log "Installing Python dependencies"
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $RepoRoot "monitor\requirements.txt") --quiet

# ── Runtime config ────────────────────────────────────────────────────────────
Write-Log "Writing runtime_config.json"
$runtimeConfig = [ordered]@{
    backend_url  = "http://127.0.0.1:$OllamaPort"
    shared_port  = 11434
    ollama_host  = "127.0.0.1"
    ollama_port  = $OllamaPort
    updated_at   = (Get-Date).ToUniversalTime().ToString("o")
}
$runtimeConfig | ConvertTo-Json -Depth 5 | Set-Content -Path $RuntimeConfig -Encoding UTF8

# ── Machine environment variables (read by services running as SYSTEM) ────────
Write-Log "Setting machine environment variables"
[System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_RUNTIME_CONFIG",        $RuntimeConfig,   "Machine")
[System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_OLLAMA_CONTROL_SCRIPT", $ControlScript,   "Machine")

# ── Scheduled tasks (AtStartup, SYSTEM) ──────────────────────────────────────
Write-Log "Registering scheduled tasks for proxy and admin"

$taskSettings = New-ScheduledTaskSettingsSet `
    -RestartCount 10 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)   # 0 = no time limit

$principal = New-ScheduledTaskPrincipal -UserId "SYSTEM" -LogonType ServiceAccount -RunLevel Highest
$trigger   = New-ScheduledTaskTrigger -AtStartup

foreach ($svc in @(
    @{ Name = $ProxyTaskName; Module = "monitor.app";   Desc = "SharedOllama Proxy (port 11434)" },
    @{ Name = $AdminTaskName; Module = "monitor.admin"; Desc = "SharedOllama Admin (port 11444)" }
)) {
    Remove-Task $svc.Name

    $action = New-ScheduledTaskAction `
        -Execute         $VenvPython `
        -Argument        "-m $($svc.Module)" `
        -WorkingDirectory $RepoRoot

    Register-ScheduledTask `
        -TaskName    $svc.Name `
        -Action      $action `
        -Trigger     $trigger `
        -Settings    $taskSettings `
        -Principal   $principal `
        -Description $svc.Desc | Out-Null

    Write-Log "Registered: $($svc.Name)"
}

# ── Firewall ──────────────────────────────────────────────────────────────────
if ($SkipFirewall) {
    Write-Log "Skipping firewall setup"
}
else {
    foreach ($rule in @(
        @{ Name = "SharedOllama Proxy 11434 Inbound"; Port = 11434 },
        @{ Name = "SharedOllama Admin 11444 Inbound"; Port = 11444 }
    )) {
        if (-not (Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Action Allow `
                -Protocol TCP -LocalPort $rule.Port -Profile Any | Out-Null
            Write-Log "Firewall rule created: $($rule.Name)"
        }
    }
}

# ── Start tasks now (without waiting for reboot) ──────────────────────────────
Write-Log "Starting services now"
Start-ScheduledTask -TaskName $ProxyTaskName
Start-ScheduledTask -TaskName $AdminTaskName
Start-Sleep -Seconds 4

# ── Verify ────────────────────────────────────────────────────────────────────
Write-Log "Verifying endpoints"
try {
    $r = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11434/health" -ErrorAction Stop
    Write-Log "Proxy health OK: $($r.status)"
}
catch {
    try {
        $r = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11434/api/version" -ErrorAction Stop
        Write-Log "Proxy fallback OK: version $($r.version)"
    }
    catch {
        Write-Log "Proxy not yet responding (Ollama may still be starting): $($_.Exception.Message)"
    }
}

try {
    $r = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11444/monitor/api/admin/state" -ErrorAction Stop
    Write-Log "Admin state OK. Client endpoint: $($r.client_endpoint.url)"
}
catch {
    Write-Log "Admin not yet responding: $($_.Exception.Message)"
}

Write-Log ""
Write-Log "Setup complete."
Write-Log "  Proxy : http://localhost:11434  (clients connect here)"
Write-Log "  Admin : http://localhost:11444/monitor"
Write-Log "  Ollama: starts at user logon on port $OllamaPort (OLLAMA_HOST set in user environment)"
Write-Log ""
Write-Log "Both services will start automatically at next boot (Windows Scheduled Tasks)."
Write-Log "Input: Start the service"
Write-Log "Output: Service started successfully."
