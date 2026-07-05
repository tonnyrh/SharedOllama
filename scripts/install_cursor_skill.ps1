<#
.SYNOPSIS
    Install or remove the SharedOllama ollama-worker Cursor personal skill.

.DESCRIPTION
    Copies the canonical skill from skills/ollama-worker/ into the user's global
    Cursor skills directory (~/.cursor/skills/ollama-worker/), making the worker
    discoverable in any Cursor workspace.

.PARAMETER Uninstall
    Remove the skill from the global Cursor skills directory.

.PARAMETER Force
    Overwrite an existing installation without prompting.

.EXAMPLE
    .\scripts\install_cursor_skill.ps1
    # Install the skill for all Cursor workspaces

.EXAMPLE
    .\scripts\install_cursor_skill.ps1 -Uninstall
    # Remove the skill
#>

param(
    [switch]$Uninstall,
    [switch]$Force
)

$SkillName = "ollama-worker"
$SourceDir = Join-Path $PSScriptRoot "..\skills\$SkillName"
$CursorDir = Join-Path $env:USERPROFILE ".cursor\skills\$SkillName"
$DisableVar = "DISABLE_OLLAMA_WORKER"

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  ERR  $msg" -ForegroundColor Red }

if ($Uninstall) {
    if (Test-Path $CursorDir) {
        Remove-Item -Recurse -Force $CursorDir
        Write-Ok "Removed $CursorDir"
    } else {
        Write-Warn ('Skill not found at ' + $CursorDir + ' — nothing to remove.')
    }
    exit 0
}

if (-not (Test-Path $SourceDir)) {
    Write-Err "Source not found: $SourceDir"
    exit 1
}

if ((Test-Path $CursorDir) -and -not $Force) {
    Write-Warn ('Skill already installed at ' + $CursorDir)
    $answer = Read-Host "  Overwrite? [y/N]"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "  Skipped. Use -Force to overwrite without prompting." -ForegroundColor Gray
        exit 0
    }
}

Write-Step ('Installing ' + $SkillName + ' to ' + $CursorDir + ' ...')
New-Item -ItemType Directory -Force $CursorDir | Out-Null
Get-ChildItem -Path $SourceDir | ForEach-Object {
    Copy-Item -Recurse -Force $_.FullName $CursorDir
}
Write-Ok "Cursor skill installed."

Write-Host ""
Write-Host "  Open this repo in Cursor for project-local wrappers and rules." -ForegroundColor Gray
Write-Host "  Or use the installed personal skill from any workspace." -ForegroundColor Gray
Write-Host "  Disable without uninstalling: set $DisableVar=1" -ForegroundColor Gray
Write-Host ('  Health check: python ' + (Join-Path $CursorDir 'scripts\check_ollama.py')) -ForegroundColor Gray
