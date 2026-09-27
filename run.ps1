param(
    [switch]$Eval,
    [switch]$Offline,
    [switch]$Demo,
    [string]$CorpusDir,
    [string]$Events
)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path $python)) {
    py -3.10 -m venv (Join-Path $PSScriptRoot '.venv')
    $python = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    & $python -m pip install -r (Join-Path $PSScriptRoot 'requirements-lock.txt')
}
if ($Eval) {
    & $python (Join-Path $PSScriptRoot 'component5/run_eval.py')
} else {
    $demoArgs = @()
    if ($CorpusDir) { $demoArgs += @('--corpus-dir', $CorpusDir) }
    elseif ($Offline -and $Demo) { $demoArgs += @('--corpus-dir', (Join-Path $PSScriptRoot 'component5/demo_corpus')) }
    if ($Events) { $demoArgs += @('--events', $Events) }
    if ($Offline) { $demoArgs += @('--backend', 'offline') }
    if ($Demo) { $demoArgs += '--demo' }
    & $python (Join-Path $PSScriptRoot 'demo_full_pipeline.py') @demoArgs
}
