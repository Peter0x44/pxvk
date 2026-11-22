# Linux setup (Wine)

pxvk is a Windows DLL, so on Linux it runs inside Wine with the game.
Wine's Vulkan (winevulkan) passes pxvk's compute work to the host's Vulkan
driver. This page covers building pxvk on Linux and preparing a Wine prefix
for PhysX games. Running a game afterwards works as in
[`install.md`](install.md), with the Wine-specific notes below.

First set up on Arch Linux (November 2026: Wine 11.22, NVIDIA 620.70.06,
Mesa 26.2.7) on a laptop with an RTX 5070 and an Arrow Lake Intel iGPU.

## Packages (Arch Linux)

Enable the `[multilib]` repository in `/etc/pacman.conf`, then:

```bash
sudo pacman -S wine winetricks wine-mono wine-gecko mingw-w64-gcc cmake ninja python vulkan-tools
```

plus the Vulkan driver for each GPU:

| GPU | Packages |
|---|---|
| NVIDIA | `nvidia-utils` (with `nvidia-open` or `nvidia`) |
| Intel | `vulkan-intel` |
| AMD | `vulkan-radeon` |

Arch's `wine` is a pure WoW64 build (`/usr/lib/wine` has `i386-windows` but
no `i386-unix`): 32-bit Windows programs run in a 64-bit Wine process, so
the 32-bit `lib32-*` Vulkan drivers are not needed for pxvk. Other
distributions that ship a classic 32-bit Wine need their 32-bit Vulkan
drivers installed too.

`vulkaninfo --summary` should list every GPU.

## Building pxvk

Same as [`development.md`](development.md), but CMake must be told that the
target is Windows; otherwise it treats the cross compiler as a Linux one
and the tools fail to link (`cannot find -ldl`):

```bash
cmake -S . -B build/x86 -G Ninja -DCMAKE_SYSTEM_NAME=Windows -DCMAKE_SYSTEM_PROCESSOR=x86 -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ -DCMAKE_BUILD_TYPE=Release
```

```bash
cmake --build build/x86
```

## Creating the Wine prefix

A separate prefix keeps the PhysX runtime and games apart from anything else
in `~/.wine`:

```bash
export WINEPREFIX=$HOME/.local/share/wineprefixes/pxvk
```

```bash
wineboot -i
```

The prefix reports Windows 10. Every `wine` command below assumes
`WINEPREFIX` is set. `WINEDEBUG=-all` silences Wine's debug output.

For the drop-in method (`dinput8.dll` in the game folder,
[`install.md`](install.md)) Wine must prefer a `dinput8.dll` found next to
the game over its own built-in one:

```bash
wine reg add 'HKCU\Software\Wine\DllOverrides' /v dinput8 /d native,builtin /f
```

Games without their own `dinput8.dll` are unaffected: Wine falls back to the
built-in DLL. (`WINEDLLOVERRIDES=dinput8=n,b` does the same for one run.)

## Installing the PhysX runtime

Every application pxvk runs needs NVIDIA's PhysX System Software
([`install.md`](install.md)); games' bundled PhysX runtimes do not work on
pxvk. The copy inside NVIDIA's display driver package needs the driver's
own setup program, so use the standalone installer:

```bash
mkdir -p tmp/physx && curl -fL -o tmp/physx/PhysX_9.26.0703_SystemSoftware.exe https://us.download.nvidia.com/Windows/9.26.0703/PhysX_9.26.0703_SystemSoftware.exe
```

Optionally check NVIDIA's signature (`osslsigncode`, from the AUR):

```bash
osslsigncode verify -in tmp/physx/PhysX_9.26.0703_SystemSoftware.exe
```

Then run it in the prefix:

```bash
wine tmp/physx/PhysX_9.26.0703_SystemSoftware.exe -s -noreboot
```

`-s` installs silently (it is NVIDIA's own setup program, so the usual
`/quiet` fails with exit code 25); leave it out to click through the
installer. Afterwards
`wine reg query 'HKLM\Software\Wow6432Node\AGEIA Technologies' /v 'PhysX Version'`
shows `0x8d4e9f` (9.26.0703). Game installers that bundle an older System
Software (GOG's Arkham City brings 9.17) may run it too; install 9.26
afterwards.

It installs into
`C:\Program Files (x86)\NVIDIA Corporation\PhysX`, which is
`$WINEPREFIX/drive_c/Program Files (x86)/NVIDIA Corporation/PhysX`.

## Checking that pxvk sees the GPU

Start the application through `pxvk-launch.exe` with a log
(`PXVK_LOG`, [`environment.md`](environment.md)):

```bash
PXVK_LOG=/tmp/pxvk.log wine build/x86/pxvk-launch.exe 'C:\path\to\game.exe'
```

The log's `init device="..."` line names the Vulkan device pxvk uses. By
default that is the discrete GPU; `PXVK_DEVICE=intel` (or `nvidia`,
`radeon`, an index) picks another. Paths in `PXVK_LOG` are Linux paths
mapped by Wine (`/tmp/pxvk.log` is fine).

## Tested games

- **Batman: Arkham City (GOG).** Install it with GOG's offline installer
  (`wine setup_batman_arkham_city_goty_1.1_(38264).exe`), then install
  PhysX 9.26 as above, since the game's installer puts 9.17 on. Copy the
  `game-folder` files into `Binaries\Win32` and start `BatmanAC.exe` there
  with `wine`. A fresh install starts with PhysX off: start the game once,
  quit, and set `PhysXLevel=2` in
  `$WINEPREFIX/drive_c/users/$USER/Documents/WB Games/Batman Arkham City GOTY/BmGame/Config/BmEngine.ini`
  (plain text; the launcher is untested under Wine). Results in
  [`compatibility.md`](compatibility.md).

## Notes

- Mesa's EGL prints `MESA-EGL: warning: pci id for fd ...: 10de:...,
  driver (null)` lines on NVIDIA machines when Wine starts. They come from
  Mesa probing the NVIDIA GPU, which it does not drive, and are harmless.
- Wine has no NVIDIA CUDA driver (`nvcuda.dll`) or NVAPI of its own, so
  without pxvk PhysX games in Wine use CPU PhysX, even on NVIDIA GPUs.
  Native CUDA is not available to opcheck either: NVIDIA's 32-bit Linux
  `libcuda.so` (620.73.04) finds no device on an RTX 50 GPU (`cuInit`
  returns 100), and Arch's WoW64 Wine could not load a 32-bit Linux library
  anyway. Run the regression check's opcheck and replays on Windows.
- Wine translates Direct3D 9 to OpenGL or Vulkan itself. DXVK
  (`winetricks dxvk`) is an alternative for the game's graphics and is
  independent of pxvk.
