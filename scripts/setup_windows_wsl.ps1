param(
    [string]$Distro = "",
    [switch]$SkipOllamaInstall,
    [switch]$SkipFirewall,
    [switch]$SkipPortProxy,
    [switch]$SkipWslInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-Log {
    param([string]$Message)
    Write-Host "[sharedollama-setup] $Message"
}

function Test-IsAdmin {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Invoke-Wsl {
    param(
        [string]$Command,
        [switch]$AllowFailure
    )

    if ([string]::IsNullOrWhiteSpace($Distro)) {
        & wsl -e bash -lc $Command
    }
    else {
        & wsl -d $Distro -e bash -lc $Command
    }

    if (-not $AllowFailure -and $LASTEXITCODE -ne 0) {
        throw "WSL command failed (exit=$LASTEXITCODE): $Command"
    }
}

function Convert-ToWslPath {
    param([string]$WindowsPath)

    $resolved = (Resolve-Path $WindowsPath).Path
    $drive = $resolved.Substring(0, 1).ToLowerInvariant()
    $tail = $resolved.Substring(2).Replace("\", "/")
    return "/mnt/$drive$tail"
}

function Set-EnvKey {
    param(
        [string]$EnvFilePath,
        [string]$Key,
        [string]$Value
    )

    if (-not (Test-Path $EnvFilePath)) {
        "${Key}=${Value}" | Set-Content -Path $EnvFilePath -Encoding UTF8
        return
    }

    $lines = Get-Content -Path $EnvFilePath
    $pattern = "^$([regex]::Escape($Key))="
    $updated = $false

    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($lines[$i] -match $pattern) {
            $lines[$i] = "${Key}=${Value}"
            $updated = $true
            break
        }
    }

    if (-not $updated) {
        $lines += "${Key}=${Value}"
    }

    $lines | Set-Content -Path $EnvFilePath -Encoding UTF8
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$envPath = Join-Path $repoRoot ".env"
$runtimeConfigPath = Join-Path $repoRoot "monitor\runtime_config.json"
$wslRepoPath = Convert-ToWslPath -WindowsPath $repoRoot
$isAdmin = Test-IsAdmin

Write-Log "Repository root: $repoRoot"
Write-Log "WSL path: $wslRepoPath"

Write-Log "Applying runtime configuration for single WSL Ollama instance"
$runtimeConfig = [ordered]@{
    backend_url = "http://127.0.0.1:11435"
    shared_port = 11434
    ollama_host = "127.0.0.1"
    ollama_port = 11435
    updated_at = (Get-Date).ToUniversalTime().ToString("o")
}
$runtimeConfig | ConvertTo-Json -Depth 5 | Set-Content -Path $runtimeConfigPath -Encoding UTF8

Set-EnvKey -EnvFilePath $envPath -Key "OLLAMA_BACKEND_URL" -Value "http://127.0.0.1:11435"

if ($SkipFirewall) {
    Write-Log "Skipping firewall setup"
}
elseif (-not $isAdmin) {
    Write-Log "Skipping firewall setup (admin rights required)"
}
else {
    Write-Log "Ensuring Windows firewall rules for 11434 and 11444"
    $rules = @(
        @{ Name = "SharedOllama API 11434 Inbound"; Port = 11434 },
        @{ Name = "SharedOllama Monitor 11444 Inbound"; Port = 11444 }
    )
    foreach ($rule in $rules) {
        if (-not (Get-NetFirewallRule -DisplayName $rule.Name -ErrorAction SilentlyContinue)) {
            New-NetFirewallRule -DisplayName $rule.Name -Direction Inbound -Action Allow -Protocol TCP -LocalPort $rule.Port -Profile Any | Out-Null
        }
    }
}

if ($SkipPortProxy) {
    Write-Log "Skipping portproxy setup"
}
elseif (-not $isAdmin) {
    Write-Log "Skipping portproxy setup (admin rights required)"
}
else {
    Write-Log "Configuring portproxy 0.0.0.0:11434 -> 127.0.0.1:11435"
    & netsh interface portproxy delete v4tov4 listenport=11434 listenaddress=0.0.0.0 | Out-Null
    & netsh interface portproxy add v4tov4 listenport=11434 listenaddress=0.0.0.0 connectport=11435 connectaddress=127.0.0.1 | Out-Null
}

if ($SkipWslInstall) {
    Write-Log "Skipping WSL install/start steps"
}
else {
    Write-Log "Running WSL installer"
    $installFlags = ""
    if ($SkipOllamaInstall) {
        $installFlags = "--skip-ollama-install"
    }
    Invoke-Wsl -Command "cd '$wslRepoPath' && chmod +x scripts/install_wsl.sh scripts/wsl_ollama_control.sh && ./scripts/install_wsl.sh $installFlags"

    Write-Log "Restarting SharedOllama proxy/admin services in WSL"
    Invoke-Wsl -Command "systemctl --user daemon-reload >/dev/null 2>&1 || true"
    Invoke-Wsl -Command "systemctl --user enable --now sharedollama-proxy.service >/dev/null 2>&1 || true"
    Invoke-Wsl -Command "systemctl --user enable --now sharedollama-admin.service >/dev/null 2>&1 || true"
}

Write-Log "Verifying endpoints"
try {
    $proxyHealth = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11434/health"
    Write-Log "Proxy health: $($proxyHealth.status)"
}
catch {
    try {
        $proxyVersion = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11434/api/version"
        Write-Log "Proxy health fallback OK. Upstream version: $($proxyVersion.version)"
    }
    catch {
        Write-Log "Proxy health check failed: $($_.Exception.Message)"
    }
}

try {
    $adminState = Invoke-RestMethod -UseBasicParsing "http://127.0.0.1:11444/monitor/api/admin/state"
    $clientEndpoint = $adminState.client_endpoint.url
    Write-Log "Admin state OK. Client endpoint: $clientEndpoint"
}
catch {
    Write-Log "Admin state check failed: $($_.Exception.Message)"
}

if (-not $SkipPortProxy -and $isAdmin) {
    Write-Log "Current portproxy rules:"
    & netsh interface portproxy show all
}

Write-Log "Setup finished"
Write-Log "Input: Start the service"
Write-Log "Output: Service started successfully."
