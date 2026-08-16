# qwen38 post-benchmark chain - persistent detached process (ASCII-only file).
# Waits for the raw qwen38_27b_q4km benchmark DONE marker + 129 records, then:
#   1. canonical raw evaluation (m2_bench_eval) + same-item comparison vs
#      gemini_protocol_clean_v1 and mlkit_ink_rtl_a1 (persisted per-item records)
#   2. RAG arm over persisted raws only (frozen m2_rag_ocr, course CV, top-k 4)
#   3. frozen paired RAG evaluation + grading-decision preservation
# Everything logs to postchain.log; a CHAIN_DONE marker signals the session
# watcher, which then performs the image-first fidelity audit (agent work).
# No frozen configuration is touched; no reference is opened before the RAG
# records exist (m2_rag_ocr never reads references; eval scripts run after).
$ErrorActionPreference = "Continue"
$repo = "C:\Users\ethan\PycharmProjects\sharon-project"
$py = Join-Path $repo ".venv\Scripts\python.exe"
$outRaw = Join-Path $repo "evaluation\hebrew_bench_v2\outputs\qwen38_27b_q4km"
$outRag = Join-Path $repo "evaluation\hebrew_bench_v2\outputs\qwen38_rag_ocr_v1"
$log = Join-Path $outRaw "postchain.log"
$benchLog = Join-Path $outRaw "overnight.log"

"$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') postchain start (pid $PID)" | Out-File -FilePath $log -Append -Encoding utf8
function Log($m) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') $m"
    $line | Out-File -FilePath $log -Append -Encoding utf8
    Write-Host $line
}

try {
    Set-Location $repo

    # ---- Stage 0: wait for the raw benchmark ----
    Log "waiting for raw benchmark DONE + 129 records"
    while ($true) {
        $n = 0
        if (Test-Path (Join-Path $outRaw "run1")) {
            $n = (Get-ChildItem (Join-Path $outRaw "run1") -File | Measure-Object).Count
        }
        $done = $false
        if (Test-Path $benchLog) {
            $done = (Select-String -Path $benchLog -Pattern "DONE - score with" -Quiet)
        }
        if ($done -and $n -ge 129) { break }
        Start-Sleep -Seconds 60
    }
    Log "raw benchmark DONE detected; records=$n"

    # ---- Stage 1: canonical raw evaluation + comparison ----
    Log "STAGE 1: canonical evaluation of qwen38_27b_q4km"
    & $py "scripts\m2_bench_eval.py" qwen38_27b_q4km 2>&1 | ForEach-Object { Log "$_" }
    Log "stage 1 eval exit: $LASTEXITCODE"
    & $py "scripts\m2_qwen38_compare.py" 2>&1 | ForEach-Object { Log "$_" }
    Log "stage 1 comparison exit: $LASTEXITCODE"

    # ---- Stage 2: RAG arm over persisted raws (no image inference) ----
    Log "STAGE 2: qwen38_rag_ocr_v1 (frozen m2_rag_ocr, course CV, top-k 4)"
    & $py "scripts\m2_rag_ocr.py" --course CV --source-config qwen38_27b_q4km `
        --config-id qwen38_rag_ocr_v1 --top-k 4 2>&1 | ForEach-Object { Log "$_" }
    Log "stage 2 rag exit: $LASTEXITCODE"
    $nr = 0
    if (Test-Path (Join-Path $outRag "run1")) {
        $nr = (Get-ChildItem (Join-Path $outRag "run1") -File | Measure-Object).Count
    }
    Log "rag records persisted: $nr"

    # ---- Stage 3: frozen paired evaluation + grading preservation ----
    Log "STAGE 3: reference join (only now) - paired eval + grading preservation"
    & $py "scripts\m2_bench_eval.py" qwen38_rag_ocr_v1 2>&1 | ForEach-Object { Log "$_" }
    & $py "scripts\m2_rag_paired.py" qwen38_rag_ocr_v1 2>&1 | ForEach-Object { Log "$_" }
    & $py "scripts\m2_grading_eval.py" --config-id qwen38_27b_q4km 2>&1 | ForEach-Object { Log "$_" }
    & $py "scripts\m2_grading_eval.py" --config-id qwen38_rag_ocr_v1 2>&1 | ForEach-Object { Log "$_" }
    Log "stage 3 exit: $LASTEXITCODE"
    Log "CHAIN_DONE - fidelity audit of RAG-changed items is the remaining (agent) step"
}
catch {
    Log "CHAIN EXCEPTION: $($_.Exception.Message)"
    Log ($_.ScriptStackTrace)
    Log "CHAIN_DONE - with exception (see above)"
    exit 1
}
