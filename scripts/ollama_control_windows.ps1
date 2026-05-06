param(
    [string]$Action = "status",
    [string]$OllamaHost = "127.0.0.1",
    [string]$OllamaPort = "11434"
)

$ErrorActionPreference = "SilentlyContinue"

$LogFile = Join-Path $env:LOCALAPPDATA "Ollama\logs\server.log"

function Write-Json {
    param(
        [bool]$Ok,
        [string]$ActionName,
        [bool]$Running,
        [string]$Pid = "",
        [string]$Note = ""
    )
    $okStr      = if ($Ok)      { "true" } else { "false" }
    $runningStr = if ($Running) { "true" } else { "false" }
    $logEscaped  = $LogFile.Replace("\", "\\")
    $noteEscaped = $Note.Replace('"', '\"')
    Write-Output "{`"ok`":$okStr,`"action`":`"$ActionName`",`"running`":$runningStr,`"pid`":`"$Pid`",`"host`":`"$OllamaHost`",`"port`":`"$OllamaPort`",`"log_file`":`"$logEscaped`",`"message`":`"$noteEscaped`"}"
}

function Find-OllamaExe {
    $fromPath = Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if ($fromPath) { return $fromPath }
    $localInstall = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    if (Test-Path $localInstall) { return $localInstall }
    return $null
}

function Get-OllamaPid {
    # Detect Ollama by finding a listener on the configured port
    $conn = Get-NetTCPConnection -LocalPort ([int]$OllamaPort) -State Listen -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($conn) { return [string]$conn.OwningProcess }
    return $null
}

function Start-OllamaServe {
    $existingPid = Get-OllamaPid
    if ($existingPid) {
        Write-Json -Ok $true -ActionName "start" -Running $true -Pid $existingPid -Note "ollama already running"
        return
    }

    $ollamaExe = Find-OllamaExe
    if (-not $ollamaExe) {
        Write-Json -Ok $false -ActionName "start" -Running $false -Note "ollama not found - install from https://ollama.com/download"
        return
    }

    $env:OLLAMA_HOST = "${OllamaHost}:${OllamaPort}"
    Start-Process -FilePath $ollamaExe -ArgumentList "serve" -WindowStyle Hidden

    Start-Sleep -Seconds 2
    $newPid = Get-OllamaPid
    if ($newPid) {
        Write-Json -Ok $true -ActionName "start" -Running $true -Pid $newPid -Note "ollama started"
    }
    else {
        Write-Json -Ok $false -ActionName "start" -Running $false -Note "failed to start ollama"
    }
}

function Stop-OllamaServe {
    $runningPid = Get-OllamaPid
    if (-not $runningPid) {
        Write-Json -Ok $true -ActionName "stop" -Running $false -Note "ollama was not running"
        return
    }

    Stop-Process -Id ([int]$runningPid) -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1

    $stillRunning = Get-OllamaPid
    if ($stillRunning) {
        Write-Json -Ok $false -ActionName "stop" -Running $true -Pid $stillRunning -Note "ollama is still running"
    }
    else {
        Write-Json -Ok $true -ActionName "stop" -Running $false -Note "ollama stopped"
    }
}

if (-not (Get-Command ollama -ErrorAction SilentlyContinue) -and
    -not (Test-Path (Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"))) {
    Write-Json -Ok $false -ActionName $Action -Running $false -Note "ollama command not found"
    exit 1
}

switch ($Action) {
    "start" {
        Start-OllamaServe
    }
    "stop" {
        Stop-OllamaServe
    }
    "restart" {
        $runningPid = Get-OllamaPid
        if ($runningPid) {
            Stop-Process -Id ([int]$runningPid) -Force -ErrorAction SilentlyContinue
            Start-Sleep -Seconds 1
        }
        Start-OllamaServe
    }
    "status" {
        $runningPid = Get-OllamaPid
        if ($runningPid) {
            Write-Json -Ok $true -ActionName "status" -Running $true -Pid $runningPid -Note "ollama running"
        }
        else {
            Write-Json -Ok $true -ActionName "status" -Running $false -Note "ollama not running"
        }
    }
    default {
        Write-Json -Ok $false -ActionName $Action -Running $false -Note "invalid action"
        exit 2
    }
}
