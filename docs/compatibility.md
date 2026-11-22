# Compatibility

Which games use 32-bit GPU PhysX, which PhysX runtime they load, and what
pxvk can run today. Status as of November 2026.

## How a game reaches the GPU

PhysX 2.x games do not contain GPU PhysX themselves. They ship a small
`PhysXLoader.dll`, which loads the matching `PhysXCore.dll` from the
**PhysX System Software** (`Program Files (x86)\NVIDIA Corporation\PhysX\Engine`).
The System Software keeps every runtime version side by side: versioned
folders (`v2.8.0`, ...) and hashed folders (newer 2.8.4 builds, PhysX 3
and GRB runtimes). The GPU kernels live in those runtimes as CUDA fatbins.
Compatibility is therefore mostly a question of runtime versions, of which
there are few, not of individual games.

To find a game's runtime, read the version of the `PhysXLoader.dll` it
ships, or run it under pxvk and compare the logged `module load hash`
values with a known runtime's.

## Runtimes

Scanned from the System Software installed on the development machine. All
GPU runtimes carry PTX (as well as precompiled GPU code), which is what
pxvk compiles. "Compiles" means every kernel compiles with `pxvk-ptxc` and
passes `spirv-val`; it does not mean the results were compared with native.

| Runtime | GPU code | pxvk | Missing |
|---|---|---|---|
| 2.3.x – 2.6.x, 2.7.0, 2.7.2 | none (CPU or Ageia PPU only) | not needed | – |
| 2.7.1, 2.7.4, 2.7.6 | 17 modules, PTX 4.1 | compiles | – |
| 2.7.3 | 17 modules, PTX 4.1 | **runs in Unreal Tournament 3** (fluid and cloth kernels; not compared with native) | – |
| 2.7.5 | 19 modules | compiles (`mul24` for a generic scan helper, `%clock` for fluid force fields' random seeds, both supported) | – |
| **2.8.0** | 17 modules | **validated**: bit-exact replays, samples pixel-identical, Mirror's Edge runs | – |
| 2.8.1, 2.8.3 (two builds) | 17–18 modules | compiles | – |
| 2.8.4 (eight builds, 2.8.4.9 – 2.8.4.12) | 17 modules | compiles (float atomic add, new in 2.8.4 for the GPU cloth normal kernels `PxgCudaDeformable(Uber)PreCalNormals`, is supported); **FluidMark validated** (bit-exact replays; it does not launch the cloth normal kernels) | – |
| PhysX 3.3 x86 (`PhysX3Gpu_x86` 3.3.0.2, beside APEX Particles and Turbulence) | 5 modules, PTX 5.0 | 50 of 51 kernels compile; never run | `simulateCloths` uses `atom.global.inc.u32` (wrapping increment) |
| APEX Particles x86 | 20 modules | 68 of 69 kernels compile; never run | `madc.hi.u32` (IOFX mesh output) |
| APEX Turbulence x86 | 9 modules | 43 of 45 kernels compile; never run | two double-to-float grid copies load `.f64` |
| `Engine\4F8D3BB436C7`: PhysX 2.8.4.9 core, APEX 1.3.0.4 Particles and TurbulenceFS (CHECKED builds), `PhysX3GpuCHECKED_x86` | 17 + 14 + 9 + 1 modules | **runs in Batman: Arkham Origins**; 190 of 193 kernels compile | `madc.hi.u32` (IOFX mesh output) and two double-to-float grid copies, none launched natively |
| GRB x86 (`GRB_1_x86`, GPU rigid bodies) | 22 modules | 155 of 222 kernels compile; never run | texture objects (65 kernels), `bfind`, `.f64` |
| 64-bit: `PhysXCore64` 2.8.3, `PhysX3Gpu_x64` 3.3.1.4 | 64-bit pointers | later (charter stage 4) | a 64-bit pxvk |

The whole 2.7/2.8 family compiles: all 17 GPU runtimes, 1,824 kernels,
every one passing `spirv-val` (October 2026). The last gaps were float
atomic add, which 2.8.4 games with GPU cloth will need, and `mul24` and
`%clock` for 2.7.5. Neither of the latter was refactored away. 2.7.5 is
an outlier with two extra modules (eleven more variants of a scan helper,
the `mul24` users, and fluid force-field kernels, the `%clock` users) that
neither 2.7.4 nor 2.7.6 and later have. 2.8.4 is the first runtime with
GPU cloth normal kernels, which accumulate vertex normals with float atomic
adds; earlier runtimes have no such kernels. To repeat the scan, extract each runtime with
`tools/extract-physx-ptx.py` into `tmp/` and compile the modules with
`pxvk-ptxc` (the runtimes are proprietary; keep them out of the repository).

## Games

The 32-bit games with GPU PhysX, as compiled by the community when RTX 50
cards dropped 32-bit CUDA (ResetEra list, reported by PC Guide). NVIDIA
driver 591.44 restored GPU PhysX on RTX 50 for the nine titles marked ★,
with Batman: Arkham Asylum announced for 2026; the others run PhysX on the
CPU on RTX 50 cards. pxvk is relevant to all of them on AMD, Intel and
other Vulkan devices.

Only the runtimes of Mirror's Edge and Unreal Tournament 3 (by hash),
Borderlands 2 and Batman: Arkham City (from the running games' loaded
modules) and Batman: Arkham Origins (its update GUID and a native trace)
are confirmed. The other runtime assignments are
unverified estimates from release dates and engines. A game's
`PhysXLoader.dll` alone does not decide it: the update loader can swap in a
System Software core for a GUID the game passes, so check what a running
game actually loads.

Borderlands 2 (Steam) uses the System Software's 2.8.4.10 runtime
(`Engine\6FE164B5308A`, confirmed from the running game's loaded modules),
not its bundled core. It ships its own PhysX build
(`PhysXSDK\2.8.4\Special\Borderlands2`, `PhysXCore.dll` 2.8.4.6, CUDA 4.1
`cudart32_41_22.dll`) whose 17 fatbins hold only precompiled sm_10/11/20/30
code and **no PTX**. But `Borderlands2.exe` passes its own GUID
(`26097F37-…`) as the last argument of `NxCreatePhysicsSDK`, the loader
forwards it to `PhysXUpdateLoader`, and every System Software release
since 9.13.1220 (December 2013) maps that GUID to the 2.8.4.10 core. The
bundled core, which pxvk cannot compile, is only the fallback without a
current System Software. The 2.8.4.10 core has PTX for all 105 of the
bundled core's kernels, with the same names, parameter sizes and
shared-memory sizes. Details in [`nvidia-32bit-physx.md`](nvidia-32bit-physx.md).

Borderlands 2 runs on pxvk (November 2026, RTX 5070 laptop and its Intel
iGPU, PhysX High). Its PhysX benchmark map is a repeatable workload: start
the game as `Borderlands2.exe benchmark` (or under `pxvk-launch`) and an
85.8 s fly-through with fluid and cloth plays, then the main menu loads. It
launches 64 of the core's kernels about 250,000 times; the 2.8.4 cloth
normal kernels (float atomics) are not among them. Results:

- **Correctness.** No pxvk errors on either GPU; the scene looked right to
  the tester. Four native captures of the cloth solver
  `KernelDeformableHybridUberSolver<1024,true>` replay EXACT on NVIDIA. On
  Intel they are BAD: up to 4 ULP on most changed words, plus 1–6
  integer-looking words. That is the known approximation drift of the
  iterative cloth solvers (`testing.md`).
- **Frame rate**, with the game's 62 fps cap off (`bSmoothFrameRate=FALSE`
  in `WillowEngine.ini`). Present times over the fly-through, recorded with
  `tools/game-bench.py` (`testing.md`), mean of two runs:

  | PhysX on | Avg fps | Median frame | 1% low fps |
  |---|---|---|---|
  | RTX 5070, native 32-bit CUDA (driver 621.01) | 137.9 | 7.3 ms | 65.6 |
  | RTX 5070, pxvk | 189.4 | 5.1 ms | 63.2 |
  | Intel iGPU, pxvk (one run) | 177.9 | 5.4 ms | 50.8 |

  pxvk on the same GPU is about 37% faster than NVIDIA's 32-bit driver. The
  iGPU result shows the frame rate is not mainly bound by PhysX GPU time, so
  the native deficit is likely per-launch or synchronisation overhead in the
  32-bit driver (not measured). The first run after new kernel variants
  appear still stalls once per variant for pipeline creation (0.5 s in
  total); later runs use the pipeline cache.

Batman: Arkham City (GOG, November 2026) runs on pxvk on the RTX 5070 laptop,
PhysX High, through `pxvk-launch`, and through the drop-in `dinput8.dll`
whether started directly or from `BmLauncher.exe` (setup in
[`install.md`](install.md)). The game has no in-game PhysX option and neither
it nor the launcher checks the GPU vendor. Its `PhysXLoader.dll` (2.8.4.4) has the
update GUID `3F527913-…` built in, which maps to the System Software's
2.8.4.12 core (`Engine\C22346819C37`), so the game needs the System Software:
the bundled `PhysXCore.dll` (2.8.4.6) has no fatbins in the current format.
The benchmark is started from the game's Graphics settings; loading
`Benchmark_01` from the command line crashes natively too. A benchmark run
makes about 540,000 launches of 79 kernels, mostly cloth, with no pxvk errors;
the cloth normal kernels (float atomics) are not among them. Over the same
stretch of the benchmark (between its stalls at about 36 s and 93 s, one run
each, measured with `tools/game-bench.py`):

| PhysX on RTX 5070 | Avg fps | Median frame | p99 frame | 1% low fps |
|---|---|---|---|---|
| native 32-bit CUDA (driver 621.01) | 184.5 | 5.17 ms | 10.54 ms | 39.6 |
| pxvk | 193.4 | 4.58 ms | 11.02 ms | 35.5 |

The stalls of 150–490 ms during the benchmark happen at the same moments and
for the same durations natively, so they are the game's own. The first pxvk
run spent 8.5 s in pipeline creation, the third 51 ms.

It also runs under Wine on Linux (November 2026: Arch Linux, Wine 11.22,
NVIDIA 620.70.06, same RTX 5070 laptop; setup in [`linux.md`](linux.md)),
through the drop-in with the game started directly and PhysX High. PhysX
used the System Software's 2.8.4.12 core and pxvk on the RTX 5070. One
session including a benchmark run made 151,896 launches of 77 kernels with
no pxvk errors, and spent 12.8 s in pipeline creation (empty cache).
Frame rates were not measured, and `BmLauncher.exe` was not tried under
Wine.

60 native captures of 10 fluid kernels and 60 of 10 cloth kernels replay
EXACT on NVIDIA, and fluid launch sizes, which follow particle counts, have the same distribution as
native (`KernelFluidDensity`: mean 6,157 against 6,162 threads, same p90 and
p99). By eye, paper and leaves move differently between runs and the sparks
look right; blue fragments might differ, which is unconfirmed and could be
ordinary variation. Effects outside the GPU kernels were not compared.

On AMD (Radeon 8060S, November 2026, build d72d826, through `pxvk-launch`)
the benchmark also ran at PhysX High: 623,708 launches of 75 kernels, fluid
and cloth, with no pxvk errors, and the tester found the effects looked
right. It was not compared with native, and later builds have not been run
there.

Unreal Tournament 3 (Steam, November 2026) runs on pxvk on the RTX 5070
laptop through the drop-in `dinput8.dll`, with NVIDIA's free PhysX Mod Pack
(August 2008), whose three maps are the game's GPU PhysX content (setup in
[`install.md`](install.md)). At the main menu the game creates the context
and loads the modules but launches no kernels; stock maps were not tried. The
game ships `PhysXLoader.dll` and
`PhysXCore.dll` 2.7.3.7; that core has no GPU code, and the loader picks the
System Software's 2.7.3.57 core (`Engine\v2.7.3`): all 17 logged module
hashes match it (2.7.4's have the same sizes but different contents). It is
the first 2.7.x runtime run in a game. The engine logs `Primary PhysX scene
will be in software`, so rigid bodies (breaking walls, thrown containers)
stay on the CPU; only fluid particles (debris, hail) and cloth run on the
GPU. One session per map with 8 bots, no pxvk errors:

| Map | Kernels | Launches | Notes |
|---|---|---|---|
| CTF-Tornado_PhysX | 10: fluid particles, fluid collision (constraints, static and dynamic response, triangles, boxes, convexes), cloth (`KernelDeformableUberSolver<512>`, `UberCollision`, `Update`) | 293,000 in 8.5 min | debris from the start; cloth probably for the deforming metal roofs |
| CTF-LightHouse_PhysX | 8: fluid only, including capsule collision | 41,000 | no launches until something breaks |
| DM-HeatRay_PhysX | 8: fluid only, including capsule collision | 50,000 | falling hail from load |

About 6–9 launches per PhysX step at about 62 steps
per second (about 117 with cloth), with 7–8% of the time spent waiting for
the GPU. The tester saw the effects as correct. Native captures were not
made, so nothing was compared bit-exact.

Batman: Arkham Origins (GOG, November 2026) runs on pxvk on the RTX 5070
laptop with PhysX High, through the drop-in `dinput8.dll` (setup in
[`install.md`](install.md)). It is not a PhysX 3 game: it ships PhysX
2.8.4.8 plus APEX 1.3.0.1 in CHECKED builds (optimized, with NVIDIA's
internal checks), whose GPU code is cubins only. `BatmanOrigins.exe`
carries the update GUID `56C8B958-C577-4BC6-AEB7-4F8D3BB436C7`, which the
System Software maps to `Engine\4F8D3BB436C7`: a 2.8.4.9 core, APEX
Particles and TurbulenceFS 1.3.0.4 and, as an implicit library,
`PhysX3GpuCHECKED_x86.dll`, all with PTX; APEX ClothingGPU has no update and
never runs on the GPU. A native trace of the benchmark (`tools/nvcuda-trace`,
which logs graphics interop calls) shows 5.4 million launches of 78
kernels, mostly APEX Turbulence (smoke and steam), field samplers, IOFX
particle output and 2.8.4 cloth, and no graphics interop: no D3D context or
resource registration, no surface binding. APEX's IOFX declares four
`.surfref`s but no kernel uses a surface instruction; its output reaches the
host through `PhysX3Gpu`'s `MemCopyBatchedAsync` kernel writing mapped
pinned memory, and the game uploads it into the textures its sprite vertex
shader reads.

What pxvk needed (all checked with opcheck, bit-exact against the native
driver on NVIDIA):

- `.surfref` declarations (parsed, `cuModuleGetSurfRef` returns a handle);
- `cvt.sat.f32.f32` (IOFX colour packing);
- `vote.ballot.b32` (IOFX bounding boxes), as a subgroup ballot;
- `setp.f64` of a float widened by `cvt.f64.f32` against a double
  constant (`x < 1e-7` without an `f`; TurbulenceFS and BasicFS),
  rewritten as an exact f32 comparison;
- `cuMemsetD32(Async)`, and stubs for the D3D context creation entry points
  `PhysX3Gpu` binds but never calls (it fails to start if any is missing);
- `CAN_MAP_HOST_MEMORY` reported as 0: with mapping `PhysX3Gpu` asks for
  device pointers of pinned memory, which pxvk cannot provide (the APEX
  smoke and steam were missing); without it, it copies with
  `cuMemcpy*Async` and every effect renders;
- the real SM count (36, from `VK_NV_shader_sm_builtins`) instead of the
  persona's 16: APEX sizes its grid-stride launches by it, and the field
  samplers, 39% of pxvk's GPU time, ran on half the workgroups (with
  `PXVK_SM_COUNT=0` the benchmark gives 198.8 and 197.7 fps, with 36
  207.4 and 207.8);
- on Intel: total memory reported as the device's memory instead of the
  heap limit, which is 4 bytes short of 1 GiB there. The game enables GPU
  APEX only from 1 GiB (768 MiB in a case that looks like a dedicated
  PhysX GPU): `BatmanOrigins.exe` 0xf46320 asks `PhysX3Gpu`'s context
  manager for its total memory and otherwise releases the context;
- an acquire barrier before volatile shared loads in warp-synchronous
  code: on Intel `BasicIOS_compactKernel` lost particles (flickering
  smoke that vanished) because lanes read shared memory other lanes had
  written under a branch. With `VK_KHR_shader_maximal_reconvergence` this
  code now gets subgroup control barriers at reconvergence points, defined
  by the Vulkan memory model (`architecture.md`); in the benchmark, back to
  back, NVIDIA gave 201.6 fps with them against 204.2 with memory-only
  barriers, Intel 116.9 against 108.7, both within noise.

The tester saw smoke, steam, paper, newspapers and snow tracks. Measured
back to back after a reboot, on AC power, from the game's own benchmark
(`BMGame\Logs\benchmark.log`, 2560×1600, PhysX High; one run each):

| | Avg fps | Min fps |
|---|---|---|
| native 32-bit CUDA (driver 621.01) | 163.6 | 131.5 |
| pxvk | 207.4 | 158.7 |
| PhysX Off (for reference, earlier) | 237.3 | 154.4 |

An earlier pair measured in a slower power state gave 84 native against
113 pxvk, and on battery the GPU's Battery Boost caps the game at 30 fps.
97 native captures of 49 APEX kernels (two launches each, taken mid-
benchmark with `PXVK_CAPTURE_KERNEL`) replay 90 EXACT and 5 ok on
NVIDIA, 89 EXACT and 6 ok on Intel. The other 2 are
`TurbulenceFS_Multigrid8x8x8Solve`, which races by design and differs
between replays (`architecture.md`). They are part of the regression check
(`testing.md`). On the Intel iGPU (PhysX there,
rendering on the RTX 5070) the tester saw the same smoke and steam as on
NVIDIA.

| Likely runtime | Games |
|---|---|
| 2.7.x / early 2.8 | Ghost Recon Advanced Warfighter 2, Monster Madness: Battle for Suburbia, Warmonger: Operation Downtown Destruction, Crazy Machines 2, Armageddon Riders |
| 2.7.3 (confirmed, by hash) | Unreal Tournament 3 (PhysX Mod Pack) |
| 2.8.0 (confirmed) | Mirror's Edge ★ |
| 2.8.4.10 (confirmed, through the update loader) | Borderlands 2 ★ |
| 2.8.4.12 (confirmed, through the update loader) | Batman: Arkham City ★ |
| 2.8.1 – 2.8.3 | Sacred 2: Fallen Angel and Ice & Blood, Cryostasis, Darkest of Days, Batman: Arkham Asylum, Shattered Horizon, Metro 2033 ★, Dark Void, Mafia II ★, Blur |
| 2.8.4, often with APEX | Alice: Madness Returns ★, The Bureau: XCOM Declassified, Metro: Last Light ★, Hydrophobia: Prophecy, The Secret World, Rise of the Triad, Mars: War Logs, Gas Guzzlers: Combat Carnage, Deep Black, Depth Hunter |
| 2.8.4.9 + APEX 1.3 (confirmed, through the update loader) | Batman: Arkham Origins ★ |
| possibly PhysX 3.3 (APEX Turbulence) | Assassin's Creed IV: Black Flag ★ |
| unknown | Star Trek D-A-C, Star Trek (2013), 7554, QQ Dance and QQ Dance 2, Hot Dance Party and Hot Dance Party II, Jianxia 3, MStar, Continent of the Ninth, Passion Leads Army |

## 64-bit games

64-bit GPU PhysX games keep working on every NVIDIA card, RTX 50
included, because NVIDIA still supports 64-bit CUDA. A 64-bit pxvk would
only help non-NVIDIA users. Known examples (unverified): Batman: Arkham
Knight, Metro 2033 Redux, Metro: Last Light Redux, Metro Exodus, Call of
Duty: Ghosts, Lords of the Fallen (2014); Fallout 4's weapon debris and
Killing Floor 2's gore use FleX, a separate CUDA library. Their runtimes are
mostly PhysX 3.x, whose PTX uses 64-bit pointers, which pxvk's design (a
device pointer is a 32-bit offset into one buffer) does not handle yet. It
is the charter's last stage, for later.

## Games with their own CUDA effects

Some games call CUDA directly instead of through PhysX. The charter puts
them in a later stage (5).

- **Just Cause 2** (32-bit, D3D10; GOG Complete Edition checked): GPU
  water simulation and bokeh, NVIDIA-only options. `JustCause2.exe` imports
  the CUDA runtime `cudart.dll` (about CUDA 2.3, PTX ISA 1.4), and the game
  ships `cufft.dll`, likely for the FFT ocean. The executable holds 3 PTX
  modules (sm_10) and cuFFT about 120 (sm_10, sm_13). It uses the legacy
  launch path (`cudaConfigureCall`, `cudaSetupArgument`, `cudaLaunch`; the
  runtime calls `cuParamSet*` and `cuLaunchGrid`), texture references
  (`cudaBindTexture2D`) and D3D9/D3D10 resource interop
  (`cudaD3D10RegisterResource`, `cudaD3D10MapResources`,
  `cudaD3D10ResourceGetMappedPointer`, ...), which pxvk only stubs. Not
  traced natively yet: a `tools/nvcuda-trace` run with GPU water on would
  show which kernels and interop calls it really uses.
