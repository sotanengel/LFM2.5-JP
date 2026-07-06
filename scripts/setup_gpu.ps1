# Install CUDA-enabled PyTorch into the project .venv only.
# Does NOT install system-wide CUDA Toolkit or modify global Python.
# Usage: powershell -ExecutionPolicy Bypass -File scripts/setup_gpu.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

Write-Host "==> GPU setup (venv only, no system CUDA Toolkit)"
Write-Host "    Driver check:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

if (-not (Test-Path ".venv")) {
    Write-Host "==> Creating venv..."
    uv venv --python 3.11
}

Write-Host "==> Syncing deps with gpu extra (torch 2.12.1+cu126)..."
uv sync --extra dev --extra gpu

Write-Host "==> Verifying CUDA..."
uv run --no-sync python -c @"
import torch
print('torch:', torch.__version__)
print('cuda available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('gpu:', torch.cuda.get_device_name(0))
    x = torch.randn(2, 3, device='cuda')
    print('tensor device:', x.device)
else:
    raise SystemExit('CUDA not available after gpu setup')
"@

Write-Host "==> Done. Use: make test-gpu / make smoke-test"
