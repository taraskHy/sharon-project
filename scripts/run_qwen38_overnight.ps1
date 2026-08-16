# qwen38_27b_q4km overnight launcher - canonical runner only. ASCII-only file
# (PowerShell 5.1 reads unmarked files as ANSI).
# Phase 1: canonical smoke-5 as a TECHNICAL gate (accepts images, returns
#          transcriptions, no repeated crash). Phase 2: identical config over
#          the full 129-item frozen benchmark (resumable; completed items are
#          skipped by the canonical runner). No reference scoring here - run
#          scripts/m2_bench_eval.py afterwards.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ethan\PycharmProjects\sharon-project"
$cfg = "qwen38_27b_q4km"
$outdir = Join-Path $repo "evaluation\hebrew_bench_v2\outputs\$cfg"
$log = Join-Path $outdir "overnight.log"

# --- log FIRST, before anything that can fail ---
New-Item -ItemType Directory -Force $outdir | Out-Null
"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') launcher start (pid $PID)" | Out-File -FilePath $log -Append -Encoding utf8

function Log($m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host $line
}

try {
    Set-Location $repo
    $py = Join-Path $repo ".venv\Scripts\python.exe"
    $ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
    $model = "qwen3.8:27b-q4_K_M"
    # frozen provider options (see arm_freeze.md) - written to a file and
    # passed BY PATH: PowerShell strips double quotes from inline JSON args
    # to native executables (this exact failure hit the first launch)
    $extra = Join-Path $outdir "extra_body.json"
    '{"think": false, "options": {"num_ctx": 8192, "repeat_penalty": 1.0}}' | Out-File -FilePath $extra -Encoding ascii

    # frozen smoke-5 straight from the canonical gate file
    $idsPy = Join-Path $outdir "_smoke_ids.py"
    @'
import json
d = json.load(open(r"evaluation\unlimited_ocr\gate_items.json", encoding="utf-8"))
print(",".join(d["smoke_5_frozen"]))
'@ | Out-File -FilePath $idsPy -Encoding ascii
    $smoke = (& $py $idsPy).Trim()
    Log "smoke-5 (canonical): $smoke"
    Log "ollama: $(& $ollama --version)"
    Log "model: $((& $ollama list | Select-String $model) -join ' ')"

    # ---- Phase 1: smoke-5 (technical gate) ----
    # backend ollama_native: Ollama's OpenAI-compat endpoint ignores think:false
    # (verified: 500 reasoning tokens, empty content); /api/chat honors it.
    & $py "scripts\m2_bench_run.py" --config-id $cfg --backend ollama_native --model $model `
        --preproc contrast --max-edge 1100 --extra-body $extra --items $smoke 2>&1 |
        ForEach-Object { Log "$_" }
    Log "smoke exit code: $LASTEXITCODE"

    $chkPy = Join-Path $outdir "_smoke_check.py"
    @'
import json, pathlib, sys
run = pathlib.Path(sys.argv[1])
ids = sys.argv[2].split(",")
ok = err = empty = 0
for i in ids:
    p = run / (i + ".json")
    if not p.exists():
        err += 1
        continue
    r = json.loads(p.read_text(encoding="utf-8"))
    if r.get("error"):
        err += 1
    elif not (r.get("transcription") or "").strip():
        empty += 1
    else:
        ok += 1
print(ok, err, empty)
'@ | Out-File -FilePath $chkPy -Encoding ascii
    $check = (& $py $chkPy (Join-Path $outdir "run1") $smoke).Trim()
    $parts = $check.Split(" ")
    $ok = [int]$parts[0]; $err = [int]$parts[1]; $empty = [int]$parts[2]
    Log "smoke technical gate: ok=$ok err=$err empty=$empty"
    Log "ollama ps: $((& $ollama ps | Select-Object -Skip 1) -join ' | ')"

    if ($ok -lt 3 -or $err -ge 3) {
        Log "SMOKE GATE FAILED - stopping before the full run (genuine technical failure)."
        exit 2
    }
    Log "SMOKE GATE PASSED - continuing through the full 129-item frozen benchmark."

    # ---- Phase 2: full 129 (identical config; completed items skipped) ----
    & $py "scripts\m2_bench_run.py" --config-id $cfg --backend ollama_native --model $model `
        --preproc contrast --max-edge 1100 --extra-body $extra 2>&1 |
        ForEach-Object { Log "$_" }
    Log "full run exit code: $LASTEXITCODE"
    Log "final ollama ps: $((& $ollama ps | Select-Object -Skip 1) -join ' | ')"
    $n = (Get-ChildItem (Join-Path $outdir "run1") -File | Measure-Object).Count
    Log "persisted records: $n/129"
    Log "DONE - score with: $py scripts\m2_bench_eval.py $cfg"
}
catch {
    Log "LAUNCHER EXCEPTION: $($_.Exception.Message)"
    Log ($_.ScriptStackTrace)
    exit 1
}
