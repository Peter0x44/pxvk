# Installing and testing pxvk

This guide is for running PhysX 2.8 applications on pxvk: the PhysX SDK
samples and Mirror's Edge. pxvk works on any GPU with a current Windows Vulkan
driver (AMD, Intel or NVIDIA), and an NVIDIA driver is not required.

## What you need

- **Windows 10/11** with a current GPU driver. The application is 32-bit, so
  it uses the GPU driver's 32-bit Vulkan driver. All current AMD, Intel and
  NVIDIA Windows drivers install one.
- **[NVIDIA PhysX System Software](https://www.nvidia.com/en-us/drivers/physx/physx-9-26-0703-driver/)**
  (version 9.x, for example 9.26.0703). It installs the PhysX 2.8 runtime
  into
  `C:\Program Files (x86)\NVIDIA Corporation\PhysX` and registers it.
  pxvk does not replace this runtime. It provides the CUDA driver that the
  runtime uses. The installer also works on machines without an NVIDIA GPU.
- **The pxvk files**, all three in **one folder** (for example `C:\pxvk`):

  | File | Role |
  |---|---|
  | `pxvk-launch.exe` | starts the application with pxvk loaded |
  | `nvcuda.dll` | the pxvk CUDA driver (PTX compiler and Vulkan backend) |
  | `pxvk-nvapi.dll` | minimal NVAPI so PhysX selects hardware PhysX |

  These are built into `build\x86` (see [`development.md`](development.md)). They depend only on
  Windows system DLLs. Do **not** copy `nvcuda.dll` into System32: it is not
  loaded from there. (The game folder works only together with the proxy
  `dinput8.dll`, below.)
- Alternatively, the **`game-folder`** files (`dinput8.dll`, `nvcuda.dll`,
  `pxvk-nvapi.dll`) for the drop-in method below.

## Two ways to start a game on pxvk

1. **Drop-in (any store or shortcut).** Copy the three files from
   `game-folder` (built into `build\x86\game-folder\dinput8.dll` plus
   `build\x86\nvcuda.dll` and `build\x86\pxvk-nvapi.dll`) into the folder
   that contains the game's executable, for example
   `...\Borderlands 2\Binaries\Win32`. Then start the game however you
   normally do: Steam, GOG Galaxy, the EA app or a shortcut. Games that
   import DirectInput 8 (`DINPUT8.dll`), which includes most Unreal Engine 3
   games, load the proxy `dinput8.dll` from their folder at startup. It loads
   pxvk from the same folder and passes DirectInput calls on to Windows' own
   `dinput8.dll`. It writes `pxvk.log` next to the game unless `PXVK_LOG` is
   set. To undo, delete the three files. Check first that the game folder
   has no `dinput8.dll` of its own (some mods use the same trick); do not
   overwrite one.
2. **`pxvk-launch.exe`** for games that do not import DirectInput 8 (test it:
   no `pxvk.log` appears), directly or as a Steam launch option (below).

Tested with Borderlands 2 started from Steam through its 2K launcher, both
ways, and with the drop-in on GOG's Batman: Arkham City, started directly and
through its own launcher.

## How loading works

pxvk has to be in the game process before PhysX starts, either loaded by the
proxy `dinput8.dll` or started through `pxvk-launch.exe`. Starting a game
normally without either gives the original behavior: native CUDA on NVIDIA
machines and CPU PhysX elsewhere.

1. `pxvk-launch.exe` starts the application suspended and loads
   `nvcuda.dll` from its own folder into the process. Then it resumes the
   application. The application's working directory is its own folder.
   pxvk also loads itself into every 32-bit program that application starts,
   so a game launcher (such as Borderlands 2's 2K launcher, which Steam
   starts) passes it on to the game. `PXVK_CHILDREN=0` turns this off.
2. When the PhysX runtime loads NVIDIA's CUDA runtime
   (`cudart32_65.dll`), pxvk redirects that runtime's request for
   `System32\nvcuda.dll` to itself.
3. Before it uses CUDA, PhysX (`PhysXDevice.dll`, `PhysXCore.dll`) asks NVAPI
   whether a PhysX-capable NVIDIA GPU exists. pxvk redirects those NVAPI
   loads to `pxvk-nvapi.dll`, which reports one GPU. Without this step, a
   machine without an NVIDIA driver silently falls back to CPU PhysX.

Only these loads are redirected, and only in the process started by the
launcher and the 32-bit programs it starts. Nothing is installed, no system
files are changed and no registry keys are written. To stop using pxvk,
start the application normally.

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

### Steam

In the Steam Library, right-click the game, choose **Properties**, and on
the **General** tab set **Launch Options** to:

```
"C:\pxvk\pxvk-launch.exe" %command%
```

Steam replaces `%command%` with the game's own command line, so it starts the
game (or the game's launcher) through `pxvk-launch.exe`. The form
`"C:\pxvk\pxvk-launch.exe" -- %command%` also works. Clear the box to go back
to normal. Tested with Borderlands 2, whose Steam entry starts the 2K
`Launcher.exe`, which starts `Borderlands2.exe` when you press Play.

Launch options cannot set environment variables on Windows. For a log, add
`PXVK_LOG` (and any other variable from [`environment.md`](environment.md))
as a user environment variable: Start menu, "Edit environment variables for
your account". Then restart Steam completely (Steam, Exit), since programs
only see variables that existed when they started.

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

## FluidMark

[Geeks3D FluidMark 1.5.4](https://www.geeks3d.com/20130308/fluidmark-1-5-1-physx-benchmark-fluid-sph-simulation-opengl-download/)
is a quick, repeatable test: a 60-second PhysX fluid benchmark with a score.
It is 32-bit and uses PhysX 2.8.4 from the PhysX System Software (its own
bundled `PhysXCore.dll` only serves its startup GPU check).

```powershell
$env:PXVK_LOG = "C:\temp\pxvk.log"
C:\pxvk\pxvk-launch.exe "C:\Program Files (x86)\Geeks3D\Benchmarks\FluidMark\FluidMark.exe" /preset_720 /log_score /noscore /nogui
```

`/preset_720` runs fullscreen at 1280×720 with 30,000 particles. The results
are in `FluidMark_0001.txt` next to the program: `found 1 PhysX GPU(s)` and
`PhysX hardware detected` confirm GPU PhysX, and `PhysX Avg SPS` is the
simulation rate. For a comparison on the same machine, run it once without
pxvk-launch (native CUDA on NVIDIA, CPU PhysX elsewhere), or set
`physx_cpu="1"` in `startup_options.xml` for CPU PhysX (set it back to `0`
afterwards). Scores vary by several percent between runs; compare averages.

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

The release package also contains `run-mirrors-edge.cmd`, which does the same
for the default GOG path and writes `pxvk.log` next to it. Edit the path in
it if the game is installed elsewhere.

Start it this way, not through GOG Galaxy. The log should show the context,
17 module loads and a 32 MiB allocation at startup. Kernel launches start once
a level with PhysX effects is loaded. Smoke and steam appear in most outdoor
levels. [`mirrors-edge.md`](mirrors-edge.md) describes how to jump to the
cloth and glass scenes.

## Batman: Arkham City (GOG)

Install the PhysX System Software first: the game's bundled PhysX cannot run
on pxvk. Then copy the `game-folder` files into `Binaries\Win32`.

The game has no PhysX option of its own, and a fresh install can start with
PhysX off on any GPU. It is set in the launcher, `Binaries\Win32\BmLauncher.exe`, under
**Display**: Off, Normal or High. The GPU hints shown there are only advice,
and the option is not hidden on other GPUs. A warning may appear on them, but
it only warns. GOG's Play button starts the game directly, and the launcher is
the Settings entry.

If the launcher crashes on start with a .NET error about
`Microsoft.DirectX.Direct3D`, the game's DirectX components were not installed.
Run `Setup\DxRedist\DXSETUP.exe` from the game folder.

The launcher saves the setting as `PhysXLevel` (0, 1 or 2) in
`Documents\WB Games\Batman Arkham City GOTY\BmGame\Config\UserEngine.ini`.
To change it by hand, edit that line (the file is UTF-16; Notepad keeps that).
The game copies it over `BmEngine.ini` in the same folder at every start, so
edit `BmEngine.ini` only if the launcher has never run and `UserEngine.ini`
has no such line.
The drop-in works whether the game is started from the launcher or directly.
Look for `pxvk.log` in `Binaries\Win32` to check that
PhysX runs on pxvk ("Checking that hardware PhysX is active" above). Kernel
launches start in-game or in the benchmark (Graphics settings), not at the
menu. Tested on an RTX 5070 laptop and its Intel GPU; AMD is untested.

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
- The drop-in `dinput8.dll` only works for games that import DirectInput 8.
  Other games need `pxvk-launch.exe` (directly, or as a Steam launch
  option), which GOG Galaxy and the EA app cannot use
  ([`roadmap.md`](roadmap.md), "Starting games from store launchers").
- On non-NVIDIA GPUs the hardware square-root and reciprocal approximations
  differ from NVIDIA's by 1–2 ULP. The results are therefore not bit-identical
  to native, and long cloth simulations can drift visibly from a native run
  while still looking physically correct.
