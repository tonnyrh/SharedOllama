param(
    [string]$Distro = ""
)

$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    $timestamp = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $line = "[$timestamp] [sharedollama-startup] $Message"
    Write-Host $line
    try {
        $logDir = Join-Path $env:LOCALAPPDATA "SharedOllama"
        if (-not (Test-Path $logDir)) {
            New-Item -ItemType Directory -Path $logDir -Force | Out-Null
        }
        Add-Content -Path (Join-Path $logDir "startup.log") -Value $line -Encoding UTF8
    }
    catch { }
}

function Invoke-Wsl {
    param([string]$Command)
    if ([string]::IsNullOrWhiteSpace($Distro)) {
        & wsl -e bash -lc $Command
    }
    else {
        & wsl -d $Distro -e bash -lc $Command
    }
    return $LASTEXITCODE
}

function Get-WslPrimaryIp {
    $ip = ""
    if ([string]::IsNullOrWhiteSpace($Distro)) {
        $ip = & wsl -e bash -lc "hostname -I | awk '{print `$1}'"
    }
    else {
        $ip = & wsl -d $Distro -e bash -lc "hostname -I | awk '{print `$1}'"
    }
    return "$ip".Trim()
}

Write-Log "SharedOllama startup triggered"

# --- Wait for WSL to become ready (WSL may need time after logon) ---
$maxWait = 60
$waited = 0
Write-Log "Waiting for WSL to become available..."
while ($waited -lt $maxWait) {
    $exit = Invoke-Wsl -Command "echo ready"
    if ($exit -eq 0) { break }
    Start-Sleep -Seconds 3
    $waited += 3
}

if ($waited -ge $maxWait) {
    Write-Log "ERROR: WSL did not become available within $maxWait seconds"
    exit 1
}

Write-Log "WSL is ready"

# --- Start systemd user services inside WSL ---
Write-Log "Reloading and starting systemd user services"
Invoke-Wsl -Command "systemctl --user daemon-reload 2>/dev/null; systemctl --user start sharedollama-proxy.service; systemctl --user start sharedollama-admin.service"
Write-Log "WSL services started"

# --- Update netsh portproxy with the current WSL IP ---
$wslIp = Get-WslPrimaryIp
if ([string]::IsNullOrWhiteSpace($wslIp)) {
    Write-Log "ERROR: Could not get WSL IP address"
    exit 1
}

Write-Log "WSL IP: $wslIp - updating portproxy rules"

# Requires elevated rights; the scheduled task must run with highest privileges
try {
    & netsh interface portproxy delete v4tov4 listenport=11434 listenaddress=0.0.0.0 2>$null | Out-Null
    & netsh interface portproxy add    v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11434 connectaddress=$wslIp | Out-Null

    & netsh interface portproxy delete v4tov4 listenport=11444 listenaddress=0.0.0.0 2>$null | Out-Null
    & netsh interface portproxy add    v4tov4 listenport=11444 listenaddress=0.0.0.0 connectport=11444 connectaddress=$wslIp | Out-Null

    Write-Log "Portproxy rules updated: 11434 and 11444 -> $wslIp"
}
catch {
    Write-Log "WARNING: Failed to update portproxy rules (admin rights required): $($_.Exception.Message)"
}

Write-Log "SharedOllama startup complete"
