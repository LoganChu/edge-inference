# π₀.5 edge deployment paths — measured on one RTX 5090

## Latency (median ms, batch 1, 3 cameras × 224², 200 text tokens, 10 flow-matching steps)

| path | vision | prefix | denoise_step | denoise10 | end_to_end_modular | end_to_end_fused |
|---|---|---|---|---|---|---|
| PyTorch eager fp32 (reference) | 29.60 | 89.25 | 11.64 | 110.10 | 236.35 | 235.28 |
| PyTorch eager bf16 | 29.05 | 29.90 | 13.41 | 115.27 | 179.92 | 178.94 |
| PyTorch eager bf16, TF32 for fp32 layers | 16.00 | 31.27 | 12.14 | 116.65 | 168.38 | 167.72 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | 29.29 | 25.94 | 2.76 | 26.45 | 91.81 | 81.60 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | 33.55 | 28.33 | 5.48 | 33.96 | 123.54 | 98.56 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | 32.56 | 38.18 | 5.47 | 31.19 | 129.23 | 104.39 |
| Direct TensorRT 11.1 runtime | 9.34 | 26.20 | 2.95 | — | 62.43 | — |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | 11.65 | 39.20 | 5.45 | — | 110.40 | — |
| PyTorch eager, FP8 W8A8 (torchao dynamic) on Gemma linears | 29.19 | 24.95 | 24.44 | 247.63 | 307.37 | 310.80 |
| PyTorch eager, ModelOpt FP8 fake-quant (numerics only) | 29.95 | 44.05 | 30.84 | 316.86 | 408.44 | 492.70 |
| Direct TensorRT, FP8 W8A8 (ModelOpt static) on Gemma linears | 9.54 | 15.88 | 2.92 | 26.24 | 53.23 | 49.91 |
| ExecuTorch + TensorRT delegate, FP8 engines | 11.09 | 25.95 | 5.74 | 31.06 | 96.37 | 69.69 |
| PyTorch eager, NVFP4 W4A4 (torchao dynamic) on Gemma linears | 30.66 | 50.91 | 45.99 | 454.56 | 539.05 | 531.10 |
| PyTorch eager, ModelOpt NVFP4 fake-quant (numerics only) | 31.32 | 38.58 | 23.75 | 252.18 | 308.64 | 310.05 |
| Direct TensorRT, NVFP4 W4A4 (ModelOpt) on Gemma linears | 9.72 | 8.80 | 3.42 | 30.86 | 51.07 | 47.47 |
| ExecuTorch + TensorRT delegate, NVFP4 engines | 11.14 | 19.11 | 6.34 | 35.68 | 94.33 | 68.00 |
| ExecuTorch + CUDA, vision compiled with TF32 | 13.04 | — | — | — | — | — |
| Direct TensorRT, vision with TF32 disabled | 14.34 | — | — | — | — | — |

LeRobot's own `sample_actions` (fp32): 246.0 ms; bf16: 188.7 ms.

## Fidelity of the predicted action chunk (MAE on the 7 real action dims, 8 LIBERO observations, same noise)

| path | variant | MAE vs eager fp32 | max abs vs fp32 | MAE vs eager bf16 | gripper sign agreement vs fp32 |
|---|---|---|---|---|---|
| PyTorch eager bf16 | — | 0.0021 | 0.022 | — | 1.000 |
| PyTorch eager bf16, TF32 for fp32 layers | — | 0.0021 | 0.022 | 0.0007 | 1.000 |
| PyTorch eager, FP8 W8A8 (torchao dynamic) on Gemma linears | — | 0.0042 | 0.055 | 0.0046 | 1.000 |
| PyTorch eager, ModelOpt FP8 fake-quant (numerics only) | — | 0.0058 | 0.063 | 0.0048 | 1.000 |
| PyTorch eager, NVFP4 W4A4 (torchao dynamic) on Gemma linears | — | 0.0209 | 0.176 | 0.0207 | 1.000 |
| PyTorch eager, ModelOpt NVFP4 fake-quant (numerics only) | — | 0.0180 | 0.184 | 0.0173 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | modular | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | fused | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | modular | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | fused | 0.0022 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | modular | 0.0020 | 0.024 | 0.0008 | 1.000 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | fused | 0.0020 | 0.024 | 0.0008 | 1.000 |
| Direct TensorRT 11.1 runtime | modular | 0.0019 | 0.023 | 0.0009 | 1.000 |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | modular | 0.0019 | 0.023 | 0.0009 | 1.000 |
| Direct TensorRT, FP8 W8A8 (ModelOpt static) on Gemma linears | modular | 0.0059 | 0.061 | 0.0049 | 1.000 |
| Direct TensorRT, FP8 W8A8 (ModelOpt static) on Gemma linears | fused | 0.0059 | 0.062 | 0.0050 | 1.000 |
| ExecuTorch + TensorRT delegate, FP8 engines | modular | 0.0059 | 0.061 | 0.0049 | 1.000 |
| ExecuTorch + TensorRT delegate, FP8 engines | fused | 0.0059 | 0.062 | 0.0050 | 1.000 |
| Direct TensorRT, NVFP4 W4A4 (ModelOpt) on Gemma linears | modular | 0.0196 | 0.108 | 0.0192 | 1.000 |
| Direct TensorRT, NVFP4 W4A4 (ModelOpt) on Gemma linears | fused | 0.0196 | 0.110 | 0.0192 | 1.000 |
| ExecuTorch + TensorRT delegate, NVFP4 engines | modular | 0.0196 | 0.108 | 0.0192 | 1.000 |
| ExecuTorch + TensorRT delegate, NVFP4 engines | fused | 0.0196 | 0.110 | 0.0192 | 1.000 |
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
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise10 | 29.5 | 186.4 | 486.0 (denoise10.plan 243.0, denoise10.pte 243.0, io.json 0.0) | 15.68 |
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise10 | 43.8 | 293.6 | 715.2 (denoise10.plan 357.6, denoise10.pte 357.6, io.json 0.0) | 17.18 |
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise_step | 2.1 | 20.9 | 2204.4 (denoise_step.plan 1102.2, denoise_step.pte 1102.2, io.json 0.0) | 12.86 |
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise_step | 2.4 | 23.6 | 1313.2 (denoise_step.plan 656.6, denoise_step.pte 656.6, io.json 0.0) | 14.05 |
| tensorrt torch-tensorrt 2.14.0+cu132 | denoise_step | 2.4 | 37.5 | 1582.4 (denoise_step.plan 791.2, denoise_step.pte 791.2, io.json 0.0) | 12.13 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_embed | 0.5 | 6.0 | 2108.8 (io.json 0.0, prefix_embed.plan 1054.4, prefix_embed.pte 1054.4) | 11.12 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l0 | 0.6 | 15.7 | 2646.0 (io.json 0.0, prefix_l0.plan 1323.0, prefix_l0.pte 1323.0) | 11.91 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l0 | 0.6 | 14.6 | 748.8 (io.json 0.0, prefix_l0.plan 374.4, prefix_l0.pte 374.4) | 15.11 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l0 | 0.6 | 15.2 | 1325.0 (io.json 0.0, prefix_l0.plan 662.5, prefix_l0.pte 662.5) | 15.00 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l1 | 0.8 | 10.0 | 2646.0 (io.json 0.0, prefix_l1.plan 1323.0, prefix_l1.pte 1323.0) | 11.96 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l1 | 0.6 | 12.8 | 748.8 (io.json 0.0, prefix_l1.plan 374.4, prefix_l1.pte 374.4) | 14.92 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l1 | 0.9 | 21.9 | 1325.0 (io.json 0.0, prefix_l1.plan 662.5, prefix_l1.pte 662.5) | 14.72 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l2 | 0.7 | 9.9 | 2646.0 (io.json 0.0, prefix_l2.plan 1323.0, prefix_l2.pte 1323.0) | 11.40 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l2 | 0.6 | 13.0 | 748.8 (io.json 0.0, prefix_l2.plan 374.4, prefix_l2.pte 374.4) | 13.08 |
| tensorrt torch-tensorrt 2.14.0+cu132 | prefix_l2 | 0.8 | 19.8 | 1325.0 (io.json 0.0, prefix_l2.plan 662.5, prefix_l2.pte 662.5) | 16.09 |
| tensorrt torch-tensorrt 2.14.0+cu132 | vision | 0.6 | 11.3 | 1661.2 (io.json 0.0, vision.plan 1661.2) | 17.18 |
| tensorrt (.pte re-packaged in a fresh process) | vision | — | 4.6 | 1662.6 (.pte) | 11.15 |

## Runtime memory

| path | host RSS GB at end | peak host GB | GPU used by process GB (torch allocator) | device used GB |
|---|---|---|---|---|
| PyTorch eager fp32 (reference) | 1.59 | 11.49 | 17.63 | 20.92 |
| PyTorch eager bf16 | 2.15 | 9.14 | 17.63 | 13.43 |
| PyTorch eager bf16, TF32 for fp32 layers | 1.75 | 11.61 | 17.63 | 13.43 |
| ExecuTorch + CUDA (AOTInductor), ET 1.5.1 | 1.09 | 7.59 | 0.06 | 13.15 |
| ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O | 1.32 | 7.34 | 0.06 | 12.76 |
| ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT | 1.33 | 5.74 | 0.09 | 12.96 |
| Direct TensorRT 11.1 runtime | 1.05 | 5.43 | 0.15 | 12.44 |
| ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks | 1.36 | 7.57 | 0.09 | 12.38 |
| PyTorch eager, FP8 W8A8 (torchao dynamic) on Gemma linears | 2.21 | 9.46 | 17.63 | 13.77 |
| PyTorch eager, ModelOpt FP8 fake-quant (numerics only) | 2.33 | 17.06 | 17.63 | 13.82 |
| Direct TensorRT, FP8 W8A8 (ModelOpt static) on Gemma linears | 1.28 | 6.07 | 0.16 | 10.45 |
| ExecuTorch + TensorRT delegate, FP8 engines | 1.60 | 5.22 | 0.09 | 10.40 |
| PyTorch eager, NVFP4 W4A4 (torchao dynamic) on Gemma linears | 1.87 | 16.36 | 17.63 | 14.52 |
| PyTorch eager, ModelOpt NVFP4 fake-quant (numerics only) | 2.33 | 8.40 | 17.63 | 13.89 |
| Direct TensorRT, NVFP4 W4A4 (ModelOpt) on Gemma linears | 1.44 | 3.65 | 0.16 | 9.46 |
| ExecuTorch + TensorRT delegate, NVFP4 engines | 1.79 | 5.90 | 0.09 | 9.49 |
| ExecuTorch + CUDA, vision compiled with TF32 | 0.99 | 0.65 | 0.02 | 5.77 |
| Direct TensorRT, vision with TF32 disabled | 0.96 | 2.31 | 0.02 | 5.51 |

I/O placement at the program boundary (cuda = device-resident, cpu = copied through host each call):

- ExecuTorch + CUDA (AOTInductor), ET 1.5.1: {'vision': 'in:cuda/out:cpu', 'prefix': 'in:cuda/out:cpu', 'denoise_step': 'in:cuda/out:cpu', 'denoise10': 'in:cuda/out:cpu'}
- ExecuTorch + CUDA (AOTInductor), ET 1.4.1 runtime, host I/O: {'vision': 'in:cpu/out:cpu', 'prefix': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + CUDA, ET 1.4.1, prefix in the same 4 chunks as TensorRT: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + TensorRT delegate, ET 1.4.1 runtime, host I/O, prefix in 4 chunks: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu'}
- ExecuTorch + TensorRT delegate, FP8 engines: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + TensorRT delegate, NVFP4 engines: {'vision': 'in:cpu/out:cpu', 'prefix_embed': 'in:cpu/out:cpu', 'prefix_l0': 'in:cpu/out:cpu', 'prefix_l1': 'in:cpu/out:cpu', 'prefix_l2': 'in:cpu/out:cpu', 'denoise_step': 'in:cpu/out:cpu', 'denoise10': 'in:cpu/out:cpu'}
- ExecuTorch + CUDA, vision compiled with TF32: {'vision': 'in:cuda/out:cpu'}
