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

Write-Log "SharedOllama startup triggered"

# --- Run a single WSL session that waits for systemd, starts services if needed, and returns the WSL IP.
# Combining everything into one bash call avoids multiple WSL restarts between invocations. ---
Write-Log "Initializing WSL and services"
$wslScript = @'
# Wait for systemd user session to be available (max 60s)
for i in $(seq 1 20); do
    systemctl --user is-system-running >/dev/null 2>&1 && break
    sleep 3
done

# Start services only if not already active (avoid interrupting running services)
for svc in sharedollama-proxy.service sharedollama-admin.service; do
    if ! systemctl --user is-active --quiet "$svc"; then
        systemctl --user start "$svc" 2>/dev/null || true
    fi
done

# Return WSL IP
hostname -I | awk '{print $1}'
'@

$wslOutput = ""
if ([string]::IsNullOrWhiteSpace($Distro)) {
    $wslOutput = & wsl -e bash -lc $wslScript
}
else {
    $wslOutput = & wsl -d $Distro -e bash -lc $wslScript
}

if ($LASTEXITCODE -ne 0) {
    Write-Log "ERROR: WSL session failed (exit=$LASTEXITCODE)"
    exit 1
}

Write-Log "WSL services ready"

# --- Update netsh portproxy with the current WSL IP ---
$wslIp = "$wslOutput".Trim().Split("`n")[-1].Trim()
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

# --- Keep WSL alive so it does not auto-shutdown and kill the services ---
# Without an active Windows-side holder, WSL exits ~8s after the last session ends,
# which stops all systemd services. A hidden wsl.exe sleep process prevents that.
Write-Log "Starting WSL keep-alive process"

# Kill any leftover keep-alive from a previous run
Get-Process -Name wsl -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -like "*sleep*2147483647*" } |
    Stop-Process -Force -ErrorAction SilentlyContinue

if ([string]::IsNullOrWhiteSpace($Distro)) {
    Start-Process wsl.exe -WindowStyle Hidden -ArgumentList "-e", "bash", "-lc", "sleep 2147483647"
}
else {
    Start-Process wsl.exe -WindowStyle Hidden -ArgumentList "-d", $Distro, "-e", "bash", "-lc", "sleep 2147483647"
}

Write-Log "WSL keep-alive started"
Write-Log "SharedOllama startup complete"
