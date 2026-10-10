# pxvk — GPU PhysX for any graphics card

Some older games, such as **Mirror's Edge**, have extra physics effects
(breaking glass, smoke, cloth) that only run on NVIDIA graphics cards, through
"GPU PhysX". On AMD and Intel graphics those effects either fall back to the
CPU, where they are very slow, or can't be turned on at all.

pxvk makes those effects run on the GPU you have: **AMD, Intel or NVIDIA**.
It works with games using PhysX 2.8 (roughly 2008–2012) on Windows, and
needs no NVIDIA driver.

## What works

- **Mirror's Edge** (tested with the GOG version): smoke and breaking glass.
- **Geeks3D FluidMark** benchmark.
- The **PhysX 2.8 SDK samples** (cloth and fluid demos).

Physics runs as fast as with NVIDIA's own driver or faster, and in heavy
scenes several times faster than CPU PhysX. Other PhysX 2.8 games may work too but haven't been
tried.

**[Download the latest release](https://github.com/Peter0x44/pxvk/releases/latest)**
(`pxvk-<version>.zip`).

## What you need

- Windows 10 or 11 with an up-to-date graphics driver (AMD, Intel or NVIDIA).
- **[NVIDIA PhysX System Software](https://www.nvidia.com/en-us/drivers/physx/physx-9-26-0703-driver/)**
  (9.26.0703 or another 9.x version). It installs on any PC, including ones
  without NVIDIA graphics.
- The pxvk release, unzipped into one folder, for example `C:\pxvk`. It
  contains `pxvk-launch.exe`, `nvcuda.dll` and `pxvk-nvapi.dll` (keep them
  together) and a setup guide.

## How to use it

Start the game **through `pxvk-launch.exe`**, not through Steam, GOG Galaxy or
a desktop shortcut. For example, in PowerShell:

```powershell
C:\pxvk\pxvk-launch.exe "C:\GOG Games\Mirror's Edge\Binaries\MirrorsEdge.exe"
```

Then turn on PhysX in the game's options. Nothing is installed or changed on
your PC: start the game normally and it behaves as before.

Some games need a small one-time change first. Mirror's Edge needs three of
its old PhysX files renamed so it uses the system PhysX instead; see
[`docs/install.md`](docs/install.md).

The first time a scene uses PhysX it may stutter for a moment while pxvk
compiles shaders. They are cached, so later runs are smooth.

## Checking it works and reporting problems

[`docs/install.md`](docs/install.md) explains step by step how to set up each
game, how to confirm from a log file that pxvk is running the physics, and
what to include when reporting a problem.

## Known limitations

- Games must be started with `pxvk-launch.exe`; store launchers (Steam, GOG
  Galaxy, EA) don't use pxvk yet.
- Only 32-bit PhysX 2.x games. Newer PhysX 3/4/5 games are not supported.
- Not yet available for phones or handhelds through Winlator.

## For developers

How pxvk works, building it, and the testing tools are in
[`docs/development.md`](docs/development.md).

## License and credits

pxvk is licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.

pxvk is heavily inspired by [ZLUDA](https://github.com/vosen/ZLUDA), which
runs CUDA applications on non-NVIDIA GPUs. Its handling of the CUDA runtime's
private interfaces and its NVAPI implementation informed pxvk's. PhysX and
CUDA are trademarks of NVIDIA; pxvk is not affiliated with NVIDIA.
