# Project charter

## Goal

Run classic 32-bit NVIDIA GPU PhysX 2.8.x-era CUDA workloads on any Vulkan
GPU, so games with GPU PhysX effects can use them on non-NVIDIA hardware.
This is a focused compatibility project, not a general CUDA-on-Vulkan
implementation.

Mirror's Edge (2009) is the first target. Alice: Madness Returns and Mafia II
Classic are likely later targets. The PhysX 2.8 SDK samples (`SampleCloth`,
`SampleParticleFluid`) are the day-to-day test workloads. An RTX 3070 Laptop
GPU is the native NVIDIA reference system.

CUDA-visible warp size remains 32.

## Principles

- Support what real PhysX workloads use, established by traces of the native
  driver; do not implement features only because CUDA has them.
- Match native NVIDIA results exactly where possible: differential testing
  against the real driver (`docs/testing.md`) is the main correctness signal.
- Keep changes small and independently testable.

## Out of scope

Until a target workload requires them or the scope is explicitly revised:

- general CUDA support, 64-bit CUDA or PhysX
- D3D/OpenGL–CUDA graphics interop
- APEX/Turbulence, PhysX 3.x GPU rigid bodies, FleX, Flow, OptiX

Mobile (Winlator on Qualcomm Adreno) is a later goal; `docs/roadmap.md`
records what it would need.
