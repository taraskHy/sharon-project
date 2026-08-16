# qwen38_27b_q4km overnight launcher — canonical runner only.
# Phase 1: canonical smoke-5 as a TECHNICAL gate (accepts images, returns
#          transcriptions, no repeated crash). Phase 2: identical config over
#          the full 129-item frozen benchmark (resumable; smoke items skipped
#          because their run1 files already exist). Reference scoring is NOT
#          done here — run scripts/m2_bench_eval.py afterwards.
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ethan\PycharmProjects\sharon-project"
Set-Location $repo
$py = ".\.venv\Scripts\python.exe"
$cfg = "qwen38_27b_q4km"
$model = "qwen3.8:27b-q4_K_M"
$extra = '{"think": false, "options": {"num_ctx": 8192, "repeat_penalty": 1.0}}'
$log = "$repo\evaluation\hebrew_bench_v2\outputs\$cfg\overnight.log"
New-Item -ItemType Directory -Force (Split-Path $log) | Out-Null

function Log($m) { $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"; $line | Tee-Object -FilePath $log -Append }

# frozen smoke-5 straight from the canonical gate file
$smoke = ($py -c "import json; print(','.join(json.load(open(r'evaluation\unlimited_ocr\gate_items.json', encoding='utf-8'))['smoke_5_frozen']))")
Log "smoke-5 (canonical): $smoke"
Log "ollama: $(& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" --version)"
Log "model list: $((& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" list | Select-String $model) -join ' ')"

# ---- Phase 1: smoke-5 (technical gate) ----
& $py scripts\m2_bench_run.py --config-id $cfg --backend qwen_local --model $model `
    --preproc contrast --max-edge 1100 --extra-body $extra --items $smoke 2>&1 |
    Tee-Object -FilePath $log -Append
$smokeExit = $LASTEXITCODE
Log "smoke exit code: $smokeExit"

$check = & $py -c @"
import json, pathlib
run = pathlib.Path(r'evaluation\hebrew_bench_v2\outputs\$cfg\run1')
ids = '$smoke'.split(',')
n_ok = n_err = n_empty = 0
for i in ids:
    p = run / f'{i}.json'
    if not p.exists():
        n_err += 1; continue
    r = json.loads(p.read_text(encoding='utf-8'))
    if r.get('error'):
        n_err += 1
    elif not (r.get('transcription') or '').strip():
        n_empty += 1
    else:
        n_ok += 1
print(f'{n_ok} {n_err} {n_empty}')
"@
$ok, $err, $empty = $check.Trim().Split(' ')
Log "smoke technical gate: ok=$ok err=$err empty=$empty"
Log "ollama ps: $((& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps | Select-Object -Skip 1) -join ' | ')"

# gate: real transcriptions on a majority, no repeated failure
if ([int]$ok -lt 3 -or [int]$err -ge 3) {
    Log "SMOKE GATE FAILED — stopping before the full run (genuine technical failure)."
    exit 2
}
Log "SMOKE GATE PASSED — continuing through the full 129-item frozen benchmark."

# ---- Phase 2: full 129 (identical config; completed items skipped) ----
& $py scripts\m2_bench_run.py --config-id $cfg --backend qwen_local --model $model `
    --preproc contrast --max-edge 1100 --extra-body $extra 2>&1 |
    Tee-Object -FilePath $log -Append
Log "full run exit code: $LASTEXITCODE"
Log "final ollama ps: $((& "$env:LOCALAPPDATA\Programs\Ollama\ollama.exe" ps | Select-Object -Skip 1) -join ' | ')"
Log "persisted records: $((Get-ChildItem "evaluation\hebrew_bench_v2\outputs\$cfg\run1" -File | Measure-Object).Count)/129"
Log "DONE — score with: $py scripts\m2_bench_eval.py $cfg"
