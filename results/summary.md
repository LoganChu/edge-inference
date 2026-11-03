# π₀.5 edge deployment paths — measured on one RTX 5090

## Latency (median ms, batch 1, 3 cameras × 224², 200 text tokens, 10 flow-matching steps)

| path | vision | prefix | denoise_step | denoise10 | end_to_end_modular | end_to_end_fused |
|---|---|---|---|---|---|---|
| PyTorch eager fp32 (reference) | 29.60 | 89.25 | 11.64 | 110.10 | 236.35 | 235.28 |
| PyTorch eager bf16 | 29.05 | 29.90 | 13.41 | 115.27 | 179.92 | 178.94 |
| PyTorch eager bf16, TF32 for fp32 layers | 16.00 | 31.27 | 12.14 | 116.65 | 168.38 | 167.72 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | 33.04 | 27.48 | 2.90 | 27.92 | 100.73 | 108.78 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | 33.08 | 27.31 | 5.40 | 31.60 | 119.84 | 90.99 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | 32.56 | 38.18 | 5.47 | 31.19 | 129.23 | 104.39 |
| Direct TensorRT 11.1 runtime | 9.84 | 27.15 | 2.92 | — | 62.08 | — |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | 12.20 | 89.31 | 8.13 | — | 165.81 | — |
| ExecuTorch + CUDA, vision compiled with TF32 | 13.04 | — | — | — | — | — |
| Direct TensorRT, vision with TF32 disabled | 14.34 | — | — | — | — | — |

LeRobot's own `sample_actions` (fp32): 246.0 ms; bf16: 188.7 ms.

## Fidelity of the predicted action chunk (MAE on the 7 real action dims, 8 LIBERO observations, same noise)

| path | variant | MAE vs eager fp32 | max abs vs fp32 | MAE vs eager bf16 | gripper sign agreement vs fp32 |
|---|---|---|---|---|---|
| PyTorch eager bf16 | — | 0.0021 | 0.022 | — | 1.000 |
| PyTorch eager bf16, TF32 for fp32 layers | — | 0.0021 | 0.022 | 0.0007 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | modular | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | fused | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | modular | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | fused | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | modular | 0.0020 | 0.024 | 0.0008 | 1.000 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | fused | 0.0020 | 0.024 | 0.0008 | 1.000 |
| Direct TensorRT 11.1 runtime | modular | 0.0019 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | modular | 0.0019 | 0.023 | 0.0009 | 1.000 |
| PyTorch eager bf16 | export units vs LeRobot sample_actions (same process) | MAE 1.22e-03 | | | |
| PyTorch eager fp32 | export units vs LeRobot sample_actions (same process) | MAE 0.00e+00 | | | |

## Export / build cost and artifacts

| path | method | torch.export s | backend build s | artifact MB | peak host RSS GB |
|---|---|---|---|---|---|
| executorch_cuda 1.4.1 | denoise10 | 19.8 | 254.1 | 1106.5 (aoti_cuda_blob.ptd 1097.5, denoise10.pte 9.0) | 8.81 |
| executorch_cuda 1.4.1 | denoise_step | 1.7 | 35.0 | 1102.2 (aoti_cuda_blob.ptd 1097.5, denoise_step.pte 4.7) | 10.47 |
| executorch_cuda 1.4.1 | prefix | 1.9 | 53.3 | 4802.6 (aoti_cuda_blob.ptd 4799.1, prefix.pte 3.5) | 10.88 |
| executorch_cuda 1.4.1 | prefix_embed | 0.3 | 9.1 | 1055.0 (aoti_cuda_blob.ptd 1053.3, prefix_embed.pte 1.7) | 17.18 |
| executorch_cuda 1.4.1 | prefix_l0 | 0.4 | 16.6 | 1324.5 (aoti_cuda_blob.ptd 1321.3, prefix_l0.pte 3.2) | 5.88 |
| executorch_cuda 1.4.1 | prefix_l1 | 0.4 | 13.6 | 1324.5 (aoti_cuda_blob.ptd 1321.3, prefix_l1.pte 3.2) | 8.83 |
| executorch_cuda 1.4.1 | prefix_l2 | 0.4 | 13.6 | 1324.5 (aoti_cuda_blob.ptd 1321.3, prefix_l2.pte 3.2) | 9.50 |
| executorch_cuda 1.5.1+cu132 | denoise10 | 23.8 | 242.5 | 1106.3 (aoti_cuda_blob.ptd 1097.6, denoise10.pte 8.7) | 9.41 |
| executorch_cuda 1.5.1+cu132 | denoise_step | 2.7 | 38.3 | 1102.9 (aoti_cuda_blob.ptd 1097.6, denoise_step.pte 5.3) | 11.71 |
| executorch_cuda 1.5.1+cu132 | prefix | 1.7 | 29.4 | 4803.1 (aoti_cuda_blob.ptd 4799.1, prefix.pte 4.0) | 11.91 |
| executorch_cuda 1.5.1+cu132 | vision | 1.1 | 36.4 | 1664.8 (aoti_cuda_blob.ptd 1659.3, vision.pte 5.5) | 11.66 |
| executorch_cuda 1.5.1+cu132 (TF32) | vision | 0.6 | 35.3 | 1663.7 (aoti_cuda_blob.ptd 1659.3, vision.pte 4.4) | 10.63 |
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise_step | 2.1 | 20.9 | 2204.4 (denoise_step.plan 1102.2, denoise_step.pte 1102.2, io.json 0.0) | 12.86 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_embed | 0.5 | 6.0 | 2108.8 (io.json 0.0, prefix_embed.plan 1054.4, prefix_embed.pte 1054.4) | 11.12 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l0 | 0.6 | 15.7 | 2646.0 (io.json 0.0, prefix_l0.plan 1323.0, prefix_l0.pte 1323.0) | 11.91 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l1 | 0.8 | 10.0 | 2646.0 (io.json 0.0, prefix_l1.plan 1323.0, prefix_l1.pte 1323.0) | 11.96 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l2 | 0.7 | 9.9 | 2646.0 (io.json 0.0, prefix_l2.plan 1323.0, prefix_l2.pte 1323.0) | 11.40 |
| tensorrt torch-tensorrt 2.14.0+cu132 | vision | 0.6 | 11.3 | 1661.2 (io.json 0.0, vision.plan 1661.2) | 17.18 |
| tensorrt (.pte re-packaged in a fresh process) | vision | — | 4.6 | 1662.6 (.pte) | 11.15 |

## Runtime memory

| path | host RSS GB at end | peak host GB | GPU used by process GB (torch allocator) | device used GB |
|---|---|---|---|---|
| PyTorch eager fp32 (reference) | 1.59 | 11.49 | 17.63 | 20.92 |
| PyTorch eager bf16 | 2.15 | 9.14 | 17.63 | 13.43 |
| PyTorch eager bf16, TF32 for fp32 layers | 1.75 | 11.61 | 17.63 | 13.43 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | 1.09 | 8.13 | 0.06 | 12.84 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | 1.31 | 7.76 | 0.06 | 12.64 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | 1.33 | 5.74 | 0.09 | 12.96 |
| Direct TensorRT 11.1 runtime | 1.07 | 4.12 | 0.15 | 12.09 |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | 1.37 | 6.94 | 0.09 | 12.14 |
| ExecuTorch + CUDA, vision compiled with TF32 | 0.99 | 0.65 | 0.02 | 5.77 |
| Direct TensorRT, vision with TF32 disabled | 0.96 | 2.31 | 0.02 | 5.51 |

I/O placement at the program boundary (cuda = device-resident, cpu = copied through host each call):

- ExecuTorch + CUDA (AOTInductor), ET 1.5.1: {'vision': 'in:cuda/out:cpu', 'prefix': 'in:cuda/out:cpu', 'denoise_step': 'in:cuda/out:cpu', 'denoise10': 'in:cuda/out:cpu'}
- ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O: {'vision': 'in:cpu/out:cpu', 'prefix': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu'}
- ExecuTorch + CUDA, vision compiled with TF32: {'vision': 'in:cuda/out:cpu'}
