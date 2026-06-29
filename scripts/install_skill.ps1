<#
.SYNOPSIS
    Install or remove the SharedOllama ollama-worker Claude Code skill.

.DESCRIPTION
    Copies the skill from skills/ollama-worker/ into the user's global Claude Code
    skills directory (~/.claude/skills/ollama-worker/), making /ollama-worker available
    in all Claude Code sessions.

    The skill delegates bounded coding tasks to a local Ollama model via SharedOllama,
    saving cloud tokens for architecture and reasoning work.

.PARAMETER Uninstall
    Remove the skill from the global Claude Code skills directory.

.PARAMETER Force
    Overwrite an existing installation without prompting.

.EXAMPLE
    .\scripts\install_skill.ps1
    # Install the skill

.EXAMPLE
    .\scripts\install_skill.ps1 -Uninstall
    # Remove the skill

.EXAMPLE
    .\scripts\install_skill.ps1 -Force
    # Reinstall, overwriting existing files
#>

param(
    [switch]$Uninstall,
    [switch]$Force
)

$SkillName    = "ollama-worker"
$SourceDir    = Join-Path $PSScriptRoot "..\skills\$SkillName"
$ClaudeDir    = Join-Path $env:USERPROFILE ".claude\skills\$SkillName"
$DisableVar   = "DISABLE_OLLAMA_WORKER"

function Write-Step($msg) { Write-Host "  $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "  OK  $msg" -ForegroundColor Green }
function Write-Warn($msg) { Write-Host "  WARN $msg" -ForegroundColor Yellow }
function Write-Err($msg)  { Write-Host "  ERR  $msg" -ForegroundColor Red }

# --- Uninstall ---
if ($Uninstall) {
    if (Test-Path $ClaudeDir) {
        Remove-Item -Recurse -Force $ClaudeDir
        Write-Ok "Removed $ClaudeDir"
    } else {
        Write-Warn "Skill not found at $ClaudeDir — nothing to remove."
    }
    exit 0
}

# --- Source check ---
if (-not (Test-Path $SourceDir)) {
    Write-Err "Source not found: $SourceDir"
    Write-Host "  Run this script from the project root or with the correct path." -ForegroundColor Red
    exit 1
}

# --- Existing install check ---
if ((Test-Path $ClaudeDir) -and -not $Force) {
    Write-Warn "Skill already installed at $ClaudeDir"
    $answer = Read-Host "  Overwrite? [y/N]"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "  Skipped. Use -Force to overwrite without prompting." -ForegroundColor Gray
        exit 0
    }
}

# --- Install ---
Write-Step "Installing $SkillName skill to $ClaudeDir ..."
Copy-Item -Recurse -Force $SourceDir $ClaudeDir
Write-Ok "Skill installed."

# --- Disable hint ---
Write-Host ""
Write-Host "  To disable the skill without uninstalling:" -ForegroundColor Gray
Write-Host "    Set-Item Env:\$DisableVar 1   # current session" -ForegroundColor Gray
Write-Host "    [System.Environment]::SetEnvironmentVariable('$DisableVar','1','User')  # permanent" -ForegroundColor Gray
Write-Host ""
Write-Host "  To re-enable, remove that environment variable." -ForegroundColor Gray
Write-Host ""
Write-Host "  Invoke in Claude Code with: /ollama-worker <task>" -ForegroundColor White
Write-Host "  Requires Ollama running via SharedOllama on port 11434." -ForegroundColor Gray
