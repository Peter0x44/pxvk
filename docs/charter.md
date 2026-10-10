# Project charter

## Goal

Run NVIDIA GPU PhysX CUDA workloads on any Vulkan GPU, so games with GPU
PhysX effects can use them on non-NVIDIA hardware and on NVIDIA cards that
dropped 32-bit CUDA (RTX 50). This is a focused compatibility project, not a
general CUDA-on-Vulkan implementation.

The long-term aim is every GPU PhysX game (`docs/compatibility.md` lists
them). The current focus is narrower: classic 32-bit PhysX 2.8.x. Later
stages are taken on one at a time, in this order:

1. **Now: 32-bit PhysX 2.8.x.** Mirror's Edge (2009) is the first target.
   The PhysX 2.8 SDK samples (`SampleCloth`, `SampleParticleFluid`) and
   FluidMark are the day-to-day test workloads.
2. **Later: the rest of 32-bit PhysX 2.x.** Every 2.7.x and 2.8.x runtime in
   the PhysX System Software, which covers most of the 32-bit game list
   (Alice: Madness Returns and Mafia II Classic first).
3. **Later: 32-bit PhysX 3.x and APEX.** The GPU runtime of the late 32-bit
   titles (likely Assassin's Creed IV, Batman: Arkham Origins), APEX
   Particles/Turbulence and GPU rigid bodies (GRB), plus whatever graphics
   interop they turn out to need. APEX GPU particles probably write into the
   game's Direct3D vertex buffers through CUDA graphics interop; native
   traces of the target games must confirm that first. Intended approach:
   under DXVK, run pxvk's work on DXVK's own Vulkan device (its interop
   interfaces, `IDXGIVkInteropDevice` and `ID3D9VkInteropDevice`, expose the
   device and a submission-queue lock), so registered D3D resources are
   directly usable Vulkan buffers; exposing a D3D buffer's `VkBuffer` may
   need a small DXVK addition. Without DXVK, fall back to a host round
   trip: on unmap, download the data and write it through the D3D API
   (particle buffers are small).
4. **Later: 64-bit.** A 64-bit pxvk for 64-bit GPU PhysX games (Batman:
   Arkham Knight, the Metro Redux games, Metro Exodus, ...), which only
   matters on non-NVIDIA hardware. Their PTX uses 64-bit pointers, which
   the current design (a device pointer is a 32-bit offset into one
   buffer) does not handle.

An RTX 3070 Laptop GPU is the native NVIDIA reference system. CUDA-visible
warp size remains 32.

## Principles

- Support what real PhysX workloads use, established by traces of the native
  driver; do not implement features only because CUDA has them.
- Match native NVIDIA results exactly where possible: differential testing
  against the real driver (`docs/testing.md`) is the main correctness signal.
- Keep changes small and independently testable.
- Finish a stage before starting the next; a later stage starts with traces
  of its target games, not with speculative support.

## Out of scope

Out of scope for the current stage (2.8.x) until a target workload requires
them or the stage changes: later-stage items above (other runtimes, PhysX
3.x, APEX, GRB, 64-bit, D3D/OpenGL–CUDA graphics interop).

Out of scope for good: general CUDA support, and NVIDIA libraries other
than PhysX (FleX, Flow, OptiX) unless a game on the list needs one for its
GPU PhysX effects.

Mobile (Winlator on Qualcomm Adreno) is a separate later goal;
`docs/roadmap.md` records what it would need.
