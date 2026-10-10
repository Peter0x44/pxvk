# Installing and testing pxvk

This guide is for running PhysX 2.8 applications on pxvk: the PhysX SDK
samples and Mirror's Edge. pxvk works on any GPU with a current Windows Vulkan
driver (AMD, Intel or NVIDIA), and an NVIDIA driver is not required.

## What you need

- **Windows 10/11** with a current GPU driver. The application is 32-bit, so
  it uses the GPU driver's 32-bit Vulkan driver. All current AMD, Intel and
  NVIDIA Windows drivers install one.
- **The legacy NVIDIA PhysX System Software** (version 9.x, for example
  9.23.1019). It installs the PhysX 2.8 runtime into
  `C:\Program Files (x86)\NVIDIA Corporation\PhysX` and registers it.
  pxvk does not replace this runtime. It provides the CUDA driver that the
  runtime uses. The installer also works on machines without an NVIDIA GPU.
- **The pxvk files**, all three in **one folder** (for example `C:\pxvk`):

  | File | Role |
  |---|---|
  | `pxvk-launch.exe` | starts the application with pxvk loaded |
  | `nvcuda.dll` | the pxvk CUDA driver (PTX compiler and Vulkan backend) |
  | `pxvk-nvapi.dll` | minimal NVAPI so PhysX selects hardware PhysX |

  These are built into `build\x86` (see the README). They depend only on
  Windows system DLLs. Do **not** copy `nvcuda.dll` into System32 or into the
  game folder, because it is not loaded from there.

## How loading works

You must start the application through `pxvk-launch.exe`. Starting it normally,
or through Steam, GOG Galaxy or EA, gives the original behavior: native CUDA
on NVIDIA machines and CPU PhysX elsewhere.

1. `pxvk-launch.exe` starts the application suspended and loads
   `nvcuda.dll` from its own folder into the process. Then it resumes the
   application. The application's working directory is its own folder.
2. When the PhysX runtime loads NVIDIA's CUDA runtime
   (`cudart32_65.dll`), pxvk redirects that runtime's request for
   `System32\nvcuda.dll` to itself.
3. Before it uses CUDA, PhysX (`PhysXDevice.dll`, `PhysXCore.dll`) asks NVAPI
   whether a PhysX-capable NVIDIA GPU exists. pxvk redirects those NVAPI
   loads to `pxvk-nvapi.dll`, which reports one GPU. Without this step, a
   machine without an NVIDIA driver silently falls back to CPU PhysX.

Only these loads are redirected, and only in the process started by the
launcher. Nothing is installed, no system files are changed and no registry
keys are written. To stop using pxvk, start the application normally.

This also applies on NVIDIA machines, so PhysX runs on pxvk's Vulkan path
there too. To use the NVIDIA driver's own NVAPI for this check, set
`PXVK_NVAPI=system`. Try that if PhysX picks CPU PhysX on an NVIDIA machine.
The application's own NVAPI use (driver version, stereo, ...) is never
redirected.

## Running an application

```powershell
C:\pxvk\pxvk-launch.exe "C:\path\to\Application.exe" [arguments]
```

Recommended for testing, with a log written to an existing folder:

```powershell
$env:PXVK_LOG = "C:\temp\pxvk.log"
C:\pxvk\pxvk-launch.exe "C:\path\to\Application.exe"
```

From `cmd.exe`, run `set PXVK_LOG=C:\temp\pxvk.log` first. A shortcut can
point at `pxvk-launch.exe` with the application path as its argument.

On a machine with several GPUs, pxvk uses the first discrete GPU unless you
set `PXVK_DEVICE` to part of the device name (`radeon`, `nvidia`, `intel`) or
to a device index. The device in use is printed in the first log line. A
name that matches no device is logged with the list of available devices.
Each run replaces the log, and a missing log folder is created.
[`environment.md`](environment.md) lists every variable.

Kernels are compiled the first time each one is used, and the pipelines are
cached in `%LOCALAPPDATA%\pxvk`. The first run of a scene can stutter briefly.
Later runs start without compilation. You can delete the cache folder at any
time.

## Checking that hardware PhysX is active

In the PhysX SDK samples the window title or overlay says `(hardware)` or
`(software)`. In general, the log shows it:

| Log contents | Meaning |
|---|---|
| `init device="..."`, `context create`, 17 × `module load`, `first launch ...`, periodic `stats launches=...` | hardware PhysX is running on pxvk |
| only the `init device=` line | PhysX initialized CUDA but chose CPU PhysX. The line ends with `nvapi=pxvk` when pxvk's NVAPI is in use. `nvapi=missing` means `pxvk-nvapi.dll` is not next to `nvcuda.dll`, and `nvapi=system` means `PXVK_NVAPI=system` is set |
| no log file | pxvk was not loaded or never used: the application was not started through `pxvk-launch.exe`, PhysX is disabled in the application, the PhysX System Software is missing, or the application uses its own PhysX copy (see Mirror's Edge below) |
| `init failed: no Vulkan device matches ...` or `... out of range` | `PXVK_DEVICE` names no device. The line lists the available devices |
| `error ...`, `unsupported call ...` or other `init failed` lines | a pxvk problem. Please report it with the log |

`stats` lines appear about every 5 seconds while PhysX runs. `launches`
counts kernel launches, `syncs` counts roughly one per PhysX simulation step,
and `wait` is the total time the CPU waited for the GPU.

## PhysX SDK samples

The PhysX 2.8 SDK samples (`SampleCloth.exe`, `SampleParticleFluid.exe`, ...)
run directly from the SDK's `Bin\win32` folder:

```powershell
C:\pxvk\pxvk-launch.exe "C:\...\Bin\win32\SampleCloth.exe"
```

`SampleCloth` (keys 1–9 switch scenes) and `SampleParticleFluid` use GPU
PhysX. The rigid-body samples and `SampleSoftBody` do not use CUDA in this
SDK version, so they run on the CPU and pxvk only logs its `init` line.

## Mirror's Edge (GOG)

The game ships an obsolete PhysX runtime that reports hardware PhysX as
unavailable. Make it use the installed PhysX System Software instead. With
the game closed, rename these entries in
`C:\GOG Games\Mirror's Edge\Binaries`:

| Original | Rename to |
|---|---|
| `PhysXCore.dll` | `PhysXCore.bak` |
| `PhysXDevice.dll` | `PhysXDevice.dll.bak` |
| `PhysXLocal` (folder) | `PhysXLocal.bak` |

Leave `PhysXLoader.dll` and `PhysXExtensions.dll` in place. To undo, rename
them back. Enable PhysX in the game's video options. Then:

```powershell
$env:PXVK_LOG = "C:\temp\pxvk.log"
C:\pxvk\pxvk-launch.exe "C:\GOG Games\Mirror's Edge\Binaries\MirrorsEdge.exe"
```

Start it this way, not through GOG Galaxy. The log should show the context,
17 module loads and a 32 MiB allocation at startup. Kernel launches start once
a level with PhysX effects is loaded. Smoke and steam appear in most outdoor
levels. [`mirrors-edge.md`](mirrors-edge.md) describes how to jump to the
cloth and glass scenes.

## Reporting results

Please include:

- GPU, driver version and Windows version (the first log line names the
  Vulkan device);
- the application and scene, and what you saw: effects present or missing,
  visual glitches, stutter, frame rate compared with PhysX disabled;
- the complete `PXVK_LOG` file;
- for a crash, the time and any dump in `%LOCALAPPDATA%\CrashDumps`.

`PXVK_SYNC=1` (very slow) makes the GPU finish each operation before the
next. It helps tell a synchronization bug from a computation bug. The other
debugging variables are listed in [`environment.md`](environment.md).

## Known issues

- Some antivirus software may block or flag `pxvk-launch.exe`, because it
  loads a DLL into another process. This has not been observed so far.
- Store launchers cannot start the game through pxvk yet
  ([`roadmap.md`](roadmap.md), "Easier installation").
- On non-NVIDIA GPUs the hardware square-root and reciprocal approximations
  differ from NVIDIA's by 1–2 ULP. The results are therefore not bit-identical
  to native, and long cloth simulations can drift visibly from a native run
  while still looking physically correct.
