# Mirror's Edge

Mirror's Edge (2009, Unreal Engine 3) is the first compatibility target. It
uses PhysX 2.8 GPU effects: smoke and particles (fluid), breaking glass, and
cloth/deformables. Its 17 CUDA modules are byte-identical to those of the
PhysX 2.8 SDK samples, so every kernel it runs is covered by the sample-based
tests in `docs/testing.md`.

Status: runs on pxvk on the RTX 3070 development laptop. A session through
the glass scenes ran 13 distinct kernels and 65,735 launches with no errors
and a clean shutdown. It also runs on an AMD Radeon 8060S laptop with no
NVIDIA driver (through `pxvk-nvapi.dll`). There, an outdoor level ran the
fluid kernels for 83,000 launches with no errors, with PhysX at the game's
62 fps cap and about 0.6 ms of GPU wait per frame. Not yet compared visually
or by capture against native.

## Setup (GOG version)

Installed by default at `C:\GOG Games\Mirror's Edge`. The tester version of
these steps, with the required pxvk files and how to check the log, is in
[`install.md`](install.md).

1. Install the legacy **NVIDIA PhysX System Software** (9.x). It provides the
   PhysX 2.8 runtime under `Program Files (x86)\NVIDIA Corporation\PhysX`.
   The installer also works without an NVIDIA GPU.
2. Make the game use that runtime instead of its own obsolete copy, which
   reports hardware PhysX as unavailable. With the game closed, in
   `Binaries\`, rename:
   - `PhysXDevice.dll` → `PhysXDevice.dll.bak`
   - `PhysXCore.dll` → `PhysXCore.bak`
   - the `PhysXLocal` directory → `PhysXLocal.bak`

   Keep `PhysXLoader.dll` and `PhysXExtensions.dll`. Rename back to undo.
3. Enable PhysX in the game's video options (`PhysXEnhanced=True` and
   `bDisablePhysXHardwareSupport=False` in `TdEngine.ini`).

Optional, for a window instead of fullscreen: the development machine uses
[dxwrapper](https://github.com/elishacloud/dxwrapper) as `Binaries\d3d9.dll`
with `d3d9.ini` (`EnableD3d9Wrapper=1`, `EnableWindowMode=1`,
`WindowModeBorder=1`). Delete both files to restore the original renderer.

## Launching on pxvk

```powershell
$env:PXVK_LOG = "C:\temp\pxvk.log"
build\x86\pxvk-launch.exe "C:\GOG Games\Mirror's Edge\Binaries\MirrorsEdge.exe"
```

Keep `nvcuda.dll` and `pxvk-nvapi.dll` next to `pxvk-launch.exe`. GOG Galaxy
cannot start the game through pxvk. The log shows the context creation, 17
module loads and a 32 MiB allocation at startup, then each kernel's first
launch once a level with PhysX effects is loaded. Periodic `stats` lines
count launches, with about one `sync` per PhysX step. If the log has only
the `init` line, PhysX chose CPU PhysX (see below).

## Getting to a scene

The game ignores a map URL on the command line and always starts at the main
menu. Unreal `start` commands bound to keys work once any level is loaded
(start or continue a chapter first). The bindings live in the user's
`Documents\EA Games\Mirror's Edge\TdGame\Config\TdInput.ini` (under OneDrive
if Documents is redirected); edit it only while the game is closed, as the
game rewrites it on exit. Current bindings on the development machine:

| Key | Command | Workload |
|---|---|---|
| F3 | `start Subway_p?LoadCheckpoint=Subway_Station?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE` | deformable (cloth) solver |
| F4 | `start Scraper_p?LoadCheckpoint=Server_room?...` | glass broken by gunfire: particles and shape collisions |
| F11 | `start Escape_p?LoadCheckpoint=Office?...` | chapter 1 escape through the RPA offices |
| F1 | `start Escape_p?LoadCheckpoint=RPA_Lower?...` | same building, lower floors |

The green RPA lobby with glass display cases (a well-known PhysX glass
showcase) is in `Escape_p`; which of the two checkpoints starts closest to it
is not yet confirmed. Chapter start checkpoints are listed in
`TdGame\Config\DefaultGame.ini`; other checkpoint names can be found with
`strings` on the `.umap` files in `TdGame\CookedPC\Maps\SP0x`.

Notes:

- F5/F6 are quicksave/quickload; on the development laptop F12 is also the
  touchpad toggle.
- Cheat commands such as `GiveWeapon` are disabled in the retail game. To get
  a gun, disarm a police officer (melee when the weapon flashes red).

## Workloads seen in native traces

Traces of the native driver (taken with ZLUDA's tracer before
`tools/nvcuda-trace` existed) are kept locally in
`tmp/evidence/mirrors-edge-*-trace-001`:

| Route | Kernels |
|---|---|
| Tutorial (smoke) | fluid step: hash, radix sort, reorder, density, force, collision constraints/responses; one `cuCtxSynchronize` per step |
| Subway station (F3) | `PxgCudaDeformableUpdate`, `KernelDeformableUberCollision`, `KernelDeformableUberSolver<512>` |
| Server room (F4) | fluid step plus `KernelFluidSimpleParticle` and box/capsule/convex/triangle collision batches |

All launches use the default stream, grid/block Y and Z of 1, legacy
`tex1Dfetch` texture references on linear memory, and one 32 MiB pinned host
pool.

To capture launches for replay, build `tools/nvcuda-trace` (see
`docs/testing.md`) and run the game under its launcher with launch ordinals
(the game has no harness to publish frame numbers; not yet tried with the
game):

```powershell
$env:PXVK_CAPTURE_LAUNCHES = "20000-20040"
build\nvcuda-trace-mingw\pxvk-trace.exe --output test-output\me-trace -- "C:\GOG Games\Mirror's Edge\Binaries\MirrorsEdge.exe"
```

## Machines without an NVIDIA driver

Before any CUDA call, the PhysX runtime decides whether hardware PhysX is
available through NVAPI (`nvapi.dll`, part of the NVIDIA driver). The native
trace shows it calls only `NvAPI_Initialize`, `NvAPI_GetDisplayDriverVersion`,
`NvAPI_EnumLogicalGPUs`, `NvAPI_EnumPhysicalGPUs`, `NvAPI_Stereo_IsEnabled`,
`NvAPI_SYS_GetDriverAndBranchVersion`, `NvAPI_GPU_CudaEnumComputeCapableGpus`
(which returns an error even natively), `NvAPI_GetGPUIDfromPhysicalGPU`,
`NvAPI_GetPhysicalGPUFromGPUID` and three private interfaces
(`0x36e39e6b`, `0x84244524`, `0xcb7b09d2`). Without NVAPI, PhysX calls
`cuInit` and then falls back to CPU PhysX.

pxvk handles this with `pxvk-nvapi.dll` (`src/runtime/nvapi.cpp`), which
follows ZLUDA's
[`nvapi/src/lib.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/nvapi/src/lib.rs#L544-L719).
It implements the initialize, driver-version, compute-GPU enumeration
(including `0x36e39e6b`) and GPU id/handle interfaces, and reports one GPU.
The enumeration, stereo and other two private ids return "no implementation",
and the game still selects hardware PhysX. On machines without a system
`nvapi.dll`, the injected `nvcuda.dll` redirects
only `PhysXDevice.dll`'s and `PhysXCore.dll`'s NVAPI loads to it. The game's
own NVAPI use is unaffected.
