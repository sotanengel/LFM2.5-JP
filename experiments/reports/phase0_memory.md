# Phase 0 Memory Probe Report

> **実モデル実測**: 実モデル・GPU 上で HF Trainer による 2 step 学習を実行して計測 / モデル: LiquidAI/LFM2.5-1.2B-Instruct / GPU: NVIDIA GeForce RTX 3060 Ti / grad ckpt 有効。

> **注記**: Windows WDDM は物理 VRAM 超過分をシステム RAM にスピルするため、peak が物理容量を超えていても OOM しない場合がある(実効上限は物理 VRAM 内)。

Generated: 2026-07-09T03:54:43.071914+00:00

## Summary

- Total trials: 24
- Successful: 20
- Failed: 4

- Max successful config: seq_len=6144, batch=2, n_layers=2

## Grid Results

| seq_len | batch | n_layers | success | peak_vram | error |
|---:|---:|---:|:---:|---|---|
| 1024 | 1 | 1 | OK | 3.0 GiB |  |
| 1024 | 1 | 2 | OK | 3.0 GiB |  |
| 1024 | 2 | 1 | OK | 3.8 GiB |  |
| 1024 | 2 | 2 | OK | 3.8 GiB |  |
| 1024 | 4 | 1 | OK | 5.5 GiB |  |
| 1024 | 4 | 2 | OK | 5.5 GiB |  |
| 2048 | 1 | 1 | OK | 3.8 GiB |  |
| 2048 | 1 | 2 | OK | 3.8 GiB |  |
| 2048 | 2 | 1 | OK | 5.5 GiB |  |
| 2048 | 2 | 2 | OK | 5.5 GiB |  |
| 2048 | 4 | 1 | OK | 8.8 GiB |  |
| 2048 | 4 | 2 | OK | 8.8 GiB |  |
| 4096 | 1 | 1 | OK | 5.5 GiB |  |
| 4096 | 1 | 2 | OK | 5.5 GiB |  |
| 4096 | 2 | 1 | OK | 8.8 GiB |  |
| 4096 | 2 | 2 | OK | 8.8 GiB |  |
| 4096 | 4 | 1 | OOM | 13.4 GiB | CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 11.32 GiB is allocated by PyTorch, and 216.03 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
| 4096 | 4 | 2 | OOM | 13.4 GiB | CUDA out of memory. Tried to allocate 4.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 11.32 GiB is allocated by PyTorch, and 216.03 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
| 6144 | 1 | 1 | OK | 7.1 GiB |  |
| 6144 | 1 | 2 | OK | 7.1 GiB |  |
| 6144 | 2 | 1 | OK | 12.0 GiB |  |
| 6144 | 2 | 2 | OK | 12.0 GiB |  |
| 6144 | 4 | 1 | OOM | 13.0 GiB | CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 12.98 GiB is allocated by PyTorch, and 214.92 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
| 6144 | 4 | 2 | OOM | 13.0 GiB | CUDA out of memory. Tried to allocate 6.00 GiB. GPU 0 has a total capacity of 8.00 GiB of which 0 bytes is free. Including non-PyTorch memory, this process has 17179869184.00 GiB memory in use. Of the allocated memory 12.98 GiB is allocated by PyTorch, and 214.92 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf) |
