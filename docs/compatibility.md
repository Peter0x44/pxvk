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
| 2.7.5 | 19 modules | 12 kernels fail in two modules: a scan helper (`vectorAddUniform4`) and 2 fluid force-field kernels | `mul24.lo.u32`, the `%clock` register |
| **2.8.0** | 17 modules | **validated**: bit-exact replays, samples pixel-identical, Mirror's Edge runs | – |
| 2.8.1, 2.8.3 (two builds) | 17–18 modules | compiles | – |
| 2.8.4 (eight builds, 2.8.4.9 – 2.8.4.12) | 17 modules | 9 GPU cloth normal kernels (`PxgCudaDeformable(Uber)PreCalNormals`) fail; **FluidMark validated** (bit-exact replays; it does not launch them) | float atomic add (`atom.global/shared.add.f32`, new in 2.8.4) |
| PhysX 3.3 x86 (`PhysX3Gpu_x86` 3.3.0.2, beside APEX Particles and Turbulence) | 5 modules, PTX 5.0 | 2 modules fail to parse | dual-destination `setp`; otherwise untested |
| GRB x86 (`GRB_1_x86`, GPU rigid bodies) | 22 modules | most fail | texture objects and more |
| 64-bit: `PhysXCore64` 2.8.3, `PhysX3Gpu_x64` 3.3.1.4 | 64-bit pointers | later (charter stage 4) | a 64-bit pxvk |

The whole 2.7/2.8 family is three small additions away from compiling
completely: float atomic add (which 2.8.4 games with GPU cloth will need),
`mul24` and `%clock` (2.7.5 only). Neither was refactored away. 2.7.5 is
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

Only Mirror's Edge's runtime is confirmed (by hash). The other runtime
assignments are unverified estimates from release dates and engines; check
each game's `PhysXLoader.dll` before relying on them.

| Likely runtime | Games |
|---|---|
| 2.7.x / early 2.8 | Ghost Recon Advanced Warfighter 2, Monster Madness: Battle for Suburbia, Warmonger: Operation Downtown Destruction, Unreal Tournament 3, Crazy Machines 2, Armageddon Riders |
| 2.8.0 (confirmed) | Mirror's Edge ★ |
| 2.8.1 – 2.8.3 | Sacred 2: Fallen Angel and Ice & Blood, Cryostasis, Darkest of Days, Batman: Arkham Asylum, Shattered Horizon, Metro 2033 ★, Dark Void, Mafia II ★, Blur |
| 2.8.4, often with APEX | Alice: Madness Returns ★, Batman: Arkham City ★, Borderlands 2 ★, The Bureau: XCOM Declassified, Metro: Last Light ★, Hydrophobia: Prophecy, The Secret World, Rise of the Triad, Mars: War Logs, Gas Guzzlers: Combat Carnage, Deep Black, Depth Hunter |
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
