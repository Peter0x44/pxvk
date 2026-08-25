# Project charter

## Goal

Run classic 32-bit NVIDIA GPU PhysX 2.8.x-era CUDA workloads through an eventual Vulkan compute backend. This is a focused compatibility project, not a general CUDA-on-Vulkan implementation.

Mirror's Edge (2009) is the first target. Alice: Madness Returns and Mafia II Classic are likely later targets. An RTX 3070 Laptop GPU will serve as the native NVIDIA reference system.

CUDA-visible warp size must ultimately remain 32.

## Development strategy

1. Research existing work.
2. Trace the real NVIDIA implementation.
3. Analyze the finite CUDA/PTX workload.
4. Only then design the minimum implementation required by that evidence.

## Deferred scope

- General CUDA support
- 64-bit CUDA or PhysX
- D3D/CUDA graphics interop
- Advanced APEX/Turbulence
- PhysX 3.x GPU rigid bodies
- FleX
- Flow
- OptiX
- Mobile or Winlator implementation
- Detailed memory architecture
- Detailed subgroup-emulation design

These areas remain deferred until research demonstrates that a target workload requires them or the project scope is explicitly revised.
