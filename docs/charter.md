# Project charter

## Goal

Run NVIDIA GPU PhysX CUDA workloads on any Vulkan GPU, so games with GPU
PhysX effects can use them on non-NVIDIA hardware and on NVIDIA cards that
dropped 32-bit CUDA (RTX 50). This is a focused compatibility project, not a
general CUDA-on-Vulkan implementation.

The long-term aim is every GPU PhysX game (`docs/compatibility.md` lists
them). Stages are taken on one at a time, in this order:

1. **Done (October 2026): 32-bit PhysX 2.8.0, Mirror's Edge.** Mirror's
   Edge (2009) was the first target; it and the PhysX 2.8 SDK samples
   (`SampleCloth`, `SampleParticleFluid`) use the 2.8.0 runtime, and
   FluidMark (a 2.8.4 build) is a further test workload. Kernels, replays
   and sample sweeps are validated against native, and a tester compared
   Mirror's Edge on pxvk with native GPU PhysX on an NVIDIA card: the
   effects looked the same and frame rates matched (about 72 against 70
   FPS in a GPU-bound path-traced scene with cloth).
2. **Now: the rest of 32-bit PhysX 2.x.** Every 2.7.x and 2.8.x runtime in
   the PhysX System Software, which covers most of the 32-bit game list.
   An offline scan (`docs/compatibility.md`) found all of them compiling
   except for three features:
   - float atomic add (`atom.global/shared.add.f32`): every 2.8.4 build's
     GPU cloth normal kernels, so likely any 2.8.4 game with GPU cloth;
     results depend on thread order, so these kernels cannot be bit-exact;
   - `mul24` and the `%clock` register: only 2.7.5, in kernels no other
     runtime has (a generic scan helper and fluid force fields).

   Compiler support comes first and is checked offline (every runtime
   compiles and validates; opcheck compares the new instructions with the
   native driver). Then validation against native traces of real games: a
   2.8.4 game with GPU cloth (Alice: Madness Returns or Mafia II Classic),
   and a 2.7.5 game if one is identified (likely an Ageia-era title such as
   Unreal Tournament 3 or Ghost Recon Advanced Warfighter 2; a game's
   `PhysXLoader.dll` version tells).
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

Out of scope for the current stage (the rest of 32-bit PhysX 2.x) until a
target workload requires them or the stage changes: later-stage items above
(PhysX 3.x, APEX, GRB, 64-bit, D3D/OpenGL–CUDA graphics interop).

Out of scope for good: general CUDA support, and NVIDIA libraries other
than PhysX (FleX, Flow, OptiX) unless a game on the list needs one for its
GPU PhysX effects.

Mobile (Winlator on Qualcomm Adreno) is a separate later goal;
`docs/roadmap.md` records what it would need.
