# Compatibility

Which games use 32-bit GPU PhysX, which PhysX runtime they load, and what
pxvk can run today. Status as of October 2026.

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
| 2.7.1, 2.7.3, 2.7.4, 2.7.6 | 17 modules, PTX 4.1 | compiles | – |
| 2.7.5 | 19 modules | compiles (`mul24` for a generic scan helper, `%clock` for fluid force fields' random seeds, both supported) | – |
| **2.8.0** | 17 modules | **validated**: bit-exact replays, samples pixel-identical, Mirror's Edge runs | – |
| 2.8.1, 2.8.3 (two builds) | 17–18 modules | compiles | – |
| 2.8.4 (eight builds, 2.8.4.9 – 2.8.4.12) | 17 modules | compiles (float atomic add, new in 2.8.4 for the GPU cloth normal kernels `PxgCudaDeformable(Uber)PreCalNormals`, is supported); **FluidMark validated** (bit-exact replays; it does not launch the cloth normal kernels) | – |
| PhysX 3.3 x86 (`PhysX3Gpu_x86` 3.3.0.2, beside APEX Particles and Turbulence) | 5 modules, PTX 5.0 | 2 modules fail to parse | dual-destination `setp`; otherwise untested |
| GRB x86 (`GRB_1_x86`, GPU rigid bodies) | 22 modules | most fail | texture objects and more |
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

Only Mirror's Edge's runtime (by hash) and Borderlands 2's (from the running
game's loaded modules) are confirmed. The other runtime assignments are
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
  in `WillowEngine.ini`). Present times were recorded by a small injected
  D3D9 hook over the fly-through, mean of two runs:

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

| Likely runtime | Games |
|---|---|
| 2.7.x / early 2.8 | Ghost Recon Advanced Warfighter 2, Monster Madness: Battle for Suburbia, Warmonger: Operation Downtown Destruction, Unreal Tournament 3, Crazy Machines 2, Armageddon Riders |
| 2.8.0 (confirmed) | Mirror's Edge ★ |
| 2.8.4.10 (confirmed, through the update loader) | Borderlands 2 ★ |
| 2.8.1 – 2.8.3 | Sacred 2: Fallen Angel and Ice & Blood, Cryostasis, Darkest of Days, Batman: Arkham Asylum, Shattered Horizon, Metro 2033 ★, Dark Void, Mafia II ★, Blur |
| 2.8.4, often with APEX | Alice: Madness Returns ★, Batman: Arkham City ★, The Bureau: XCOM Declassified, Metro: Last Light ★, Hydrophobia: Prophecy, The Secret World, Rise of the Triad, Mars: War Logs, Gas Guzzlers: Combat Carnage, Deep Black, Depth Hunter |
| possibly PhysX 3.3 (APEX Turbulence) | Assassin's Creed IV: Black Flag ★, Batman: Arkham Origins ★ |
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
