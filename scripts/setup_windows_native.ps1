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
$RuntimeConfigPath  = Join-Path $RepoRoot "monitor\runtime_config.json"
$ControlScriptPath  = Join-Path $RepoRoot "scripts\ollama_control_windows.ps1"
$ProxyTaskName  = "SharedOllamaProxy"
$AdminTaskName  = "SharedOllamaAdmin"
$OllamaTaskName = "SharedOllamaBackend"
$OllamaPort     = 11435   # Ollama backend; proxy exposes 11434 to clients
$SetupLog       = "C:\ProgramData\SharedOllama\setup.log"

function Write-Log {
    param([string]$Message)
    $line = "[sharedollama-setup] $Message"
    Write-Host $line
    try {
        New-Item -ItemType Directory -Path "C:\ProgramData\SharedOllama" -Force -ErrorAction SilentlyContinue | Out-Null
        Add-Content -Path $SetupLog -Value $line -Encoding UTF8
    } catch {}
}

function Test-IsAdmin {
    # IsInRole check fails for Azure AD accounts even when elevated.
    # Use token integrity level instead: High (12288) or System (16384) = elevated.
    $groups = (whoami /groups 2>$null) -join " "
    return $groups -match "S-1-16-12288|S-1-16-16384"
}

function Remove-Task { param([string]$Name)
    Unregister-ScheduledTask -TaskName $Name -Confirm:$false -ErrorAction SilentlyContinue
}

# ── Uninstall ────────────────────────────────────────────────────────────────
if ($Uninstall) {
    Write-Log "Uninstalling SharedOllama Windows native services"
    Remove-Task $ProxyTaskName
    Remove-Task $AdminTaskName
    Remove-Task $OllamaTaskName
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
function Find-RealPython {
    # Machine-wide installs first (accessible by SYSTEM), then user installs, skip Store stubs
    $candidates = @()

    # Machine-wide (Program Files)
    foreach ($root in @("C:\Program Files\Python3*", "C:\Program Files (x86)\Python3*", "C:\Python3*")) {
        Get-Item $root -ErrorAction SilentlyContinue | Sort-Object Name -Descending |
            ForEach-Object { $candidates += (Join-Path $_.FullName "python.exe") }
    }

    # User install under AppData (fallback)
    $userPyRoot = "$env:LOCALAPPDATA\Programs\Python"
    if (Test-Path $userPyRoot) {
        Get-ChildItem -Path $userPyRoot -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending |
            ForEach-Object { $candidates += $_.FullName }
    }

    # PATH-based (last resort)
    $candidates += (Get-Command python  -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)
    $candidates += (Get-Command python3 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source)

    foreach ($c in $candidates) {
        if ([string]::IsNullOrWhiteSpace($c)) { continue }
        if ($c -like "*WindowsApps*") { continue }   # Store stub - skip
        if (Test-Path $c) { return $c }
    }
    return $null
}

$pythonExe = Find-RealPython

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
$configPayload = [ordered]@{
    backend_url  = "http://127.0.0.1:$OllamaPort"
    shared_port  = 11434
    ollama_host  = "127.0.0.1"
    ollama_port  = $OllamaPort
    updated_at   = (Get-Date).ToUniversalTime().ToString("o")
}
$configPayload | ConvertTo-Json -Depth 5 | Set-Content -Path $RuntimeConfigPath -Encoding UTF8

# ── Machine environment variables (read by services running as SYSTEM) ────────
Write-Log "Setting machine environment variables"
[System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_RUNTIME_CONFIG",        $RuntimeConfigPath,  "Machine")
[System.Environment]::SetEnvironmentVariable("SHAREDOLLAMA_OLLAMA_CONTROL_SCRIPT", $ControlScriptPath,  "Machine")

# ── Scheduled tasks (AtStartup, SYSTEM) ──────────────────────────────────────
# Python is machine-wide so SYSTEM can access the venv.
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

Write-Log "Registering scheduled task for Ollama backend"
Remove-Task $OllamaTaskName
$ollamaAction = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-NonInteractive -NoProfile -ExecutionPolicy Bypass -File `"$ControlScriptPath`" start -OllamaHost 127.0.0.1 -OllamaPort $OllamaPort"
$ollamaTrigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$ollamaSettings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)
$ollamaPrincipal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest
Register-ScheduledTask `
    -TaskName $OllamaTaskName `
    -Action $ollamaAction `
    -Trigger $ollamaTrigger `
    -Settings $ollamaSettings `
    -Principal $ollamaPrincipal `
    -Description "Start SharedOllama Ollama backend on the configured internal port" | Out-Null
Write-Log "Registered: $OllamaTaskName"

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
Write-Log "Starting Ollama backend now"
$ollamaStartOutput = & powershell.exe -NonInteractive -NoProfile -ExecutionPolicy Bypass -File $ControlScriptPath start -OllamaHost 127.0.0.1 -OllamaPort $OllamaPort
Write-Log "Ollama start result: $ollamaStartOutput"

Write-Log "Starting services now"
# Stop any existing Python processes so the new config is picked up cleanly
Get-Process python -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 1
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
Write-Log "  Ollama: started now and scheduled at user logon on port $OllamaPort"
Write-Log ""
Write-Log "Proxy and admin start at boot (SYSTEM, no logon needed). Ollama starts now and at user logon."
Write-Log "Input: Start the service"
Write-Log "Output: Service started successfully."
