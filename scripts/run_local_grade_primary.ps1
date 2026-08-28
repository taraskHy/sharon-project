<#
.SYNOPSIS
Strong-PC entry point for the LOCAL grade_primary experiment.

DEFAULTS TO ZERO INFERENCE. Without -Execute this script only verifies the
frozen experiment and prints the exact plan; nothing is graded, downloaded,
or sent anywhere. Cloud grading is impossible on this path: the benchmark
gateway runs in production mode, where the cloud boundary
(autograder/cloudboundary.py) refuses every remote non-OCR request, and the
runner refuses remote URLs for grading roles without the explicit research
flag (which this script never passes).

.MODES
  (none) / -Preflight   verify freeze + boundary + list installed models. ZERO inference.
  -Smoke       -Candidate X [-Execute]  frozen 2-case DEV smoke for one candidate.
  -FullDev     -Candidate X [-Execute]  frozen 26-case derivable DEV population.
                                        Requires a completed, failure-free smoke
                                        run of the SAME candidate first.
  -Calibration -Candidate X [-Execute]  frozen 12-case CALIBRATION quality
                                        population. Requires a completed,
                                        failure-free smoke run of the SAME
                                        candidate (and, when -PromptVersion is
                                        given, the SAME prompt version) first.
  -PromptVersion V   pin a registered grading prompt version (e.g.
                     grade-v4-charitable-local); recorded in the run's config
                     hash. Default: the adapter default.

.EXAMPLES
  .\scripts\run_local_grade_primary.ps1
  .\scripts\run_local_grade_primary.ps1 -Smoke -Candidate qwen3-vl:8b-instruct -PromptVersion grade-v4-charitable-local -Execute
  .\scripts\run_local_grade_primary.ps1 -Calibration -Candidate qwen3-vl:8b-instruct -PromptVersion grade-v4-charitable-local -Execute

HELD_OUT is not reachable from this script (split is restricted to dev and
calibration; the final evaluation is a separate, explicitly confirmed
owner-run command). No model is ever downloaded here; missing candidates are
reported by preflight and must be pulled deliberately by the operator.
#>
[CmdletBinding()]
param(
    [switch]$Preflight,
    [switch]$Smoke,
    [switch]$FullDev,
    [switch]$Calibration,
    [string]$Candidate = "",
    [string]$PromptVersion = "",
    [switch]$Execute,
    [string]$BaseUrl = "http://localhost:11434/v1"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $repo
$py = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$runsRoot = "evaluation\model_selection\runs\local_grade_primary"

function Invoke-Preflight {
    Write-Host "=== PREFLIGHT (zero inference) ===" -ForegroundColor Cyan
    & $py "scripts\local_grade_preflight.py"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "PREFLIGHT REFUSED (exit $LASTEXITCODE): fix the mismatch before any run." -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

function Write-MachineProfile {
    New-Item -ItemType Directory -Force $runsRoot | Out-Null
    $stamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
    $profilePath = Join-Path $runsRoot "machine_profile_$stamp.json"
    $os  = Get-CimInstance Win32_OperatingSystem
    $cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
    $gpu = @(Get-CimInstance Win32_VideoController | ForEach-Object {
        @{ name = $_.Name; vram_bytes = $_.AdapterRAM } })
    $prof = @{
        recorded_at   = (Get-Date -Format "yyyy-MM-dd HH:mm:ss")
        hostname_hash = ([BitConverter]::ToString(
            [Security.Cryptography.SHA256]::Create().ComputeHash(
                [Text.Encoding]::UTF8.GetBytes($env:COMPUTERNAME))) -replace '-','').Substring(0,16)
        os            = $os.Caption
        ram_total_gb  = [math]::Round($os.TotalVisibleMemorySize / 1MB, 1)
        ram_free_gb   = [math]::Round($os.FreePhysicalMemory / 1MB, 1)
        cpu           = $cpu.Name
        gpus          = $gpu
        git_commit    = (git rev-parse HEAD)
    }
    $prof | ConvertTo-Json -Depth 4 | Out-File -Encoding utf8 $profilePath
    Write-Host "machine profile -> $profilePath"
    return $profilePath
}

$mode = if ($Smoke) { "smoke" } elseif ($FullDev) { "fulldev" } elseif ($Calibration) { "calibration" } else { "preflight" }

if ($mode -eq "preflight") {
    Invoke-Preflight
    Write-Host ""
    Write-Host "Next steps (each requires -Execute; nothing runs by default):"
    Write-Host "  .\scripts\run_local_grade_primary.ps1 -Smoke       -Candidate <model:tag> -Execute"
    Write-Host "  .\scripts\run_local_grade_primary.ps1 -FullDev     -Candidate <model:tag> -Execute"
    Write-Host "  .\scripts\run_local_grade_primary.ps1 -Calibration -Candidate <model:tag> -Execute"
    exit 0
}

if (-not $Candidate) {
    Write-Host "REFUSED: -Smoke/-FullDev/-Calibration require an explicit -Candidate <model:tag> (see candidates.toml [roles.grade_primary_local])." -ForegroundColor Red
    exit 2
}

# The active freeze pins the experiment's prompt version: default to it so a
# bare -Smoke/-Calibration cannot silently run the adapter default instead.
if (-not $PromptVersion) {
    $freezePath = Join-Path $repo "evaluation\model_selection\experiments\LOCAL_GRADE_CONTRACT_FREEZE_2026-08-28.json"
    if (Test-Path $freezePath) {
        try { $PromptVersion = (Get-Content $freezePath -Raw | ConvertFrom-Json).prompt_version } catch {}
    }
}

# split is RESTRICTED to dev | calibration; HELD_OUT stays unreachable here
$split = if ($mode -eq "calibration") { "calibration" } else { "dev" }
$subset = switch ($mode) {
    "smoke"       { "smoke" }
    "fulldev"     { "dev_verdict" }
    "calibration" { "calibration_verdict_v4" }
}
$benchArgs = @("-m", "autograder", "bench", "run",
    "--role", "grade_primary", "--split", $split, "--subset", $subset,
    "--candidate", $Candidate, "--backend", "ollama", "--base-url", $BaseUrl,
    "--runs-root", $runsRoot,
    "--note", "local_grade_primary $mode (strong-PC)",
    "--i-understand-this-spends-money")   # live-run gate; a local run spends $0 in provider fees
if ($PromptVersion) { $benchArgs += @("--prompt-version", $PromptVersion) }

Write-Host "=== PLAN ($mode) ===" -ForegroundColor Cyan
Write-Host "candidate : $Candidate"
Write-Host "backend   : ollama @ $BaseUrl (local only; remote URLs refuse)"
Write-Host "split     : $split | subset: $subset (frozen; HELD_OUT unreachable)"
if ($PromptVersion) { Write-Host "prompt    : $PromptVersion" }
Write-Host "runs root : $runsRoot"
Write-Host "command   : $py $($benchArgs -join ' ')"

if (-not $Execute) {
    Write-Host ""
    Write-Host "DRY: -Execute not given - zero inference performed. Add -Execute to run." -ForegroundColor Yellow
    exit 0
}

Invoke-Preflight

if ($mode -eq "fulldev" -or $mode -eq "calibration") {
    # FullDev/Calibration require a completed, failure-free smoke run of the
    # SAME candidate — and, when a prompt version is pinned, of the SAME
    # prompt version (a smoke under another prompt proves nothing about this
    # configuration).
    $smokeOk = $false
    Get-ChildItem -Path $runsRoot -Recurse -Filter "run.json" -ErrorAction SilentlyContinue | ForEach-Object {
        try {
            $r = Get-Content $_.FullName -Raw | ConvertFrom-Json
            # the candidate lives in run.json's config block (config_hash identity),
            # not at the top level
            $cand = $null; $pv = $null
            if ($null -ne $r.config) { $cand = $r.config.candidate; $pv = $r.config.prompt_version }
            if ($null -eq $cand -and $null -ne $r.spec) { $cand = $r.spec.candidate }
            if ($_.FullName -match "smoke" -and $cand -eq $Candidate) {
                $pvOk = $true
                if ($PromptVersion -and $pv -ne $PromptVersion) { $pvOk = $false }
                # a dry-run plans cases without executing them — only a LIVE
                # smoke counts (cases_done counts planned rows on dry runs)
                $live = ($r.last_mode -eq "live")
                $done = 0; $failed = 1
                if ($null -ne $r.cases_done) { $done = [int]$r.cases_done }
                if ($null -ne $r.cases_failed) { $failed = [int]$r.cases_failed }
                if ($pvOk -and $live -and $done -gt 0 -and $failed -eq 0) { $script:smokeOk = $true }
            }
        } catch {}
    }
    if (-not $smokeOk) {
        $pvNote = ""
        if ($PromptVersion) { $pvNote = " under prompt '$PromptVersion'" }
        Write-Host "REFUSED: no completed failure-free SMOKE run found for '$Candidate'$pvNote in $runsRoot. Run -Smoke -Execute first." -ForegroundColor Red
        exit 3
    }
}

$null = Write-MachineProfile
Write-Host "=== EXECUTING ($mode) - local inference only, cloud grading cost `$0 ===" -ForegroundColor Green
& $py @benchArgs
$rc = $LASTEXITCODE
Write-Host ""
Write-Host "exit $rc | results under $runsRoot (per-run directory; never overwritten - reruns resume or get a new config-hash id)"
exit $rc
