# Project charter

## Goal

Run NVIDIA GPU PhysX CUDA workloads on any Vulkan GPU, so games with GPU
PhysX effects can use them on non-NVIDIA hardware and on NVIDIA cards that
dropped 32-bit CUDA (RTX 50). This is a focused compatibility project for
games, not a general CUDA-on-Vulkan implementation: GPU PhysX first, then
the few games that run their own CUDA effects (stage 6).

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
2. **Done (November 2026): the rest of 32-bit PhysX 2.x.** Every 2.7.x and 2.8.x runtime in
   the PhysX System Software, which covers most of the 32-bit game list.
   An offline scan (`docs/compatibility.md`) found all of them compiling
   except for three features:
   - float atomic add (`atom.global/shared.add.f32`): every 2.8.4 build's
     GPU cloth normal kernels, so likely any 2.8.4 game with GPU cloth;
     results depend on thread order, so these kernels cannot be bit-exact.
     Done: every 2.8.4 build now compiles, and opcheck matches the native
     driver for the instruction and for the kernels' accumulation pattern;
   - `mul24` and the `%clock` register: only 2.7.5, in kernels no other
     runtime has (a generic scan helper and fluid force fields). Done:
     `mul24` matches the native driver; `%clock` (a random seed) reads
     `VK_KHR_shader_clock` where available.

   With these every 2.7/2.8 runtime compiles; what remains of the stage is
   validation against real games.

   Compiler support comes first and is checked offline (every runtime
   compiles and validates; opcheck compares the new instructions with the
   native driver). Then validation against native traces of real games: a
   2.8.4 game with GPU cloth (Alice: Madness Returns or Mafia II Classic),
   and a 2.7.5 game if one is identified (an Ageia-era title such as Ghost
   Recon Advanced Warfighter 2; a game's `PhysXLoader.dll` version tells).
   Unreal Tournament 3 turned out to use 2.7.3, and runs (its PhysX Mod Pack
   maps, fluid and cloth, not yet compared with native). The stage was
   closed with Borderlands 2, Batman: Arkham City and Unreal Tournament 3
   running; a 2.8.4 game with GPU cloth normals (Alice, Mafia II Classic)
   and a 2.7.5 game were not traced.
3. **Now: 32-bit PhysX 3.x and APEX.** The GPU runtime of the late 32-bit
   titles (Batman: Arkham Origins; Assassin's Creed IV is not planned, as
   its only effect, APEX smoke, uses the same modules), APEX
   Particles/Turbulence and GPU rigid bodies (GRB), plus whatever graphics
   interop they turn out to need. Batman: Arkham Origins, the first target,
   runs (November 2026): it is PhysX 2.8.4 with APEX 1.3 and its PhysX 3
   GPU helper (`docs/compatibility.md`), and a native trace showed no
   graphics interop: APEX copies its particle output to the host. It
   needed `vote.ballot`, `cvt.sat`, `.surfref` declarations, float
   comparisons against double constants and pinned memory without device
   mapping. Other games may still need interop; native traces must show
   it first. Intended approach:
   under DXVK, run pxvk's work on DXVK's own Vulkan device (its interop
   interfaces, `IDXGIVkInteropDevice` and `ID3D9VkInteropDevice`, expose the
   device and a submission-queue lock), so registered D3D resources are
   directly usable Vulkan buffers; exposing a D3D buffer's `VkBuffer` may
   need a small DXVK addition. Without DXVK, fall back to a host round
   trip: on unmap, download the data and write it through the D3D API
   (particle buffers are small).
4. **Later: 64-bit.** A 64-bit pxvk for 64-bit GPU PhysX games (Batman:
   Arkham Knight, the Metro Redux games, ...), which only matters on
   non-NVIDIA hardware. Their PTX uses 64-bit pointers, which the current
   design (a device pointer is a 32-bit offset into one buffer) does not
   handle. Later 64-bit APEX titles may need graphics interop (ZLUDA's
   GameWorks support registers D3D11 2D and 3D textures, which suggests
   volume textures for smoke), so the stage starts with a 64-bit build of
   `tools/nvcuda-trace` and a native trace of Arkham Knight. Whether Metro
   Exodus's Advanced PhysX uses the GPU is unknown (a review and forum
   posts say CPU only, unconfirmed); its install (a PhysX GPU DLL) or a
   native trace would tell.
5. **Later: NVIDIA FleX.** GPU PhysX in games largely ended around
   2015–2016 with Arkham Knight; the newer NVIDIA simulation in games is
   FleX, a separate particle-based library: Fallout 4's weapon debris
   (patch 1.3) and Killing Floor 2's gore and fluids. Its kernels,
   runtime and rendering path differ from the PhysX SDK's, so it is a
   stage of its own after the PhysX ones. Native traces come first:
   whether these builds use FleX's CUDA backend (rather than its Direct3D
   compute one) and which interop they need.
6. **Later: games' own CUDA effects.** Games that call CUDA directly
   rather than through PhysX, such as Just Cause 2's GPU water and bokeh
   (`docs/compatibility.md`). They need older CUDA runtimes, CUDA libraries
   the game ships (cuFFT) and D3D interop, so they come after the PhysX
   stages and start with native traces of the game.

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

Out of scope for the current stage (32-bit PhysX 3.x and APEX) until a
target workload requires them or the stage changes: later-stage items above
(64-bit, games' own CUDA effects), and within the stage anything no traced
game uses (D3D/OpenGL–CUDA graphics interop, GRB, f64 arithmetic).

Out of scope for good: CUDA applications that are not games (Blender,
compute and machine-learning software), and general CUDA support beyond
what game workloads use. NVIDIA libraries other than PhysX (FleX, Flow,
cuFFT, ...) only as far as a game needs them for its effects.

Mobile (Winlator on Qualcomm Adreno) is a separate later goal;
`docs/roadmap.md` records what it would need.
