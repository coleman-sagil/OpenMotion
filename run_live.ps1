# run_live.ps1 — Windows equivalent of run_live.sh
#
# Builds the openleap driver (if needed), then runs it in stream mode,
# piping frames to the Python live viewer.
#
# Prerequisites:
#   - Rust toolchain (rustup.rs)
#   - WinUSB driver installed for the Leap Motion Controller (see docs/WINUSB_SETUP.md)
#   - Python venv with onnxruntime, numpy, opencv-python-headless
#   - hand-tracking-models/ cloned as sibling (see README.md)
#
# Usage:
#   .\run_live.ps1
#   .\run_live.ps1 -SeqPath .\rust-driver\bringup_seq.txt

param(
    [string]$SeqPath = ".\rust-driver\bringup_seq.txt"
)

$ErrorActionPreference = "Stop"

# 1. Stop the Ultraleap tracking service if running (it holds the device)
$svc = Get-Service -Name "UltraleapTracking" -ErrorAction SilentlyContinue
if ($svc -and $svc.Status -eq "Running") {
    Write-Host "Stopping Ultraleap tracking service..."
    try { Stop-Service "UltraleapTracking" -Force -ErrorAction Stop } catch {
        Write-Warning "Could not stop Ultraleap service: $_"
        Write-Warning "If the device is held by another process, streaming will fail."
    }
}

# 2. Build the driver if the binary doesn't exist or source is newer
$driverDir = Join-Path $PSScriptRoot "rust-driver"
$binary = Join-Path $driverDir "target\release\openleap.exe"
$srcLib = Join-Path $driverDir "src\lib.rs"

if (-not (Test-Path $binary) -or (Get-Item $srcLib).LastWriteTime -gt (Get-Item $binary).LastWriteTime) {
    Write-Host "Building openleap driver..."
    Push-Location $driverDir
    cargo build --release
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        Write-Error "cargo build failed"
    }
    Pop-Location
}

# 3. Stream: pipe the driver's stdout (LFRM frames) to the Python viewer
#    The driver writes diagnostics to stderr; the viewer reads binary frames
#    from stdin.
Write-Host "Starting Leap Motion stream..."
Write-Host "  (Ctrl+C to stop)"

$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    $venvPython = "python"
}

& $binary stream $SeqPath 2>&1 | & $venvPython tracking\live_viewer.py --stdin
