# pxvk — classic GPU PhysX on Vulkan

pxvk runs applications that use classic 32-bit NVIDIA GPU PhysX (PhysX 2.8,
e.g. Mirror's Edge) on any GPU with a Vulkan driver: AMD, Intel or NVIDIA,
with or without an NVIDIA driver installed. It takes the place of the CUDA
driver inside the application, compiles PhysX's GPU kernels from PTX to
SPIR-V at runtime and runs them with Vulkan compute. It is not a general CUDA
implementation (see [`docs/charter.md`](docs/charter.md)).

## Status

Runs without errors on an RTX 3070 laptop (with its Intel UHD iGPU) and on a
Radeon 8060S laptop with no NVIDIA driver:

- **PhysX 2.8 SDK samples** `SampleCloth` (9 scenes) and
  `SampleParticleFluid`.
- **Mirror's Edge** (GOG): smoke and breaking-glass effects on GPU PhysX.
  Not yet compared side by side with native.
- **Geeks3D FluidMark 1.5.4** (PhysX 2.8.4).

On NVIDIA, every PTX instruction PhysX uses is bit-exact against the native
driver, and nearly all real kernel launches replayed from native captures
are too (84 of 90 cloth launches and all 112 fluid launches; the other six
are within 4 ULP). Other GPUs differ by 1–2 ULP in hardware approximations, which
is invisible in play.

Performance matches or beats native CUDA. FluidMark, 720 preset, mean of 10
runs on the RTX 3070 laptop:

| PhysX on | Simulation steps/s |
|---|---|
| RTX 3070, pxvk | 283 |
| RTX 3070, native CUDA | 269 |
| Intel UHD iGPU, pxvk | 132 |
| CPU PhysX (i7-11800H, 16 threads) | 39 |

On the Radeon 8060S the heavy cloth scenes run 2–3.6× faster than CPU
PhysX, and Mirror's Edge holds its 62 fps cap.

Not done yet: starting games through Steam or GOG Galaxy (they must be started
with `pxvk-launch.exe`), and mobile (Winlator) drivers. See
[`docs/roadmap.md`](docs/roadmap.md).

## Using pxvk

You need the legacy **NVIDIA PhysX System Software** (9.x; it installs on any
machine) and three files from the build, kept in one folder:
`pxvk-launch.exe`, `nvcuda.dll` and `pxvk-nvapi.dll`. Start the application
through the launcher:

```powershell
$env:PXVK_LOG = "C:\temp\pxvk.log"   # optional, but useful
C:\pxvk\pxvk-launch.exe "C:\path\to\Application.exe"
```

Nothing is installed or changed on the system; starting the application
normally gives the original behaviour. [`docs/install.md`](docs/install.md) is
the guide for testers: setup for the samples, Mirror's Edge and FluidMark, how
to confirm GPU PhysX from the log, and what to report.

## Building

Requires CMake, Ninja, Python 3 and a 32-bit (i686) MinGW toolchain or MSVC
Win32. Dependencies are fetched during configuration.

```powershell
cmake -S . -B build/x86 -G Ninja -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc `
      -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/x86
```

`build/x86` then contains the three runtime files, which depend only on
Windows system DLLs, and the developer tools below.

## Documentation

| Document | Contents |
|---|---|
| [`docs/install.md`](docs/install.md) | installing, running and reporting (for testers) |
| [`docs/environment.md`](docs/environment.md) | every environment variable |
| [`docs/mirrors-edge.md`](docs/mirrors-edge.md) | Mirror's Edge setup, scenes and workloads |
| [`docs/architecture.md`](docs/architecture.md) | design, Vulkan requirements, known limitations |
| [`docs/testing.md`](docs/testing.md) | differential testing, profiling, regression check |
| [`docs/roadmap.md`](docs/roadmap.md) | open work |
| [`docs/charter.md`](docs/charter.md) | goal and scope |

## Development tools

| Tool | Purpose |
|---|---|
| `pxvk-ptxc` | compile PTX kernels to SPIR-V |
| `pxvk-opcheck` | compare every PTX instruction variant with the native driver |
| `pxvk-replay` | replay captured native kernel launches on Vulkan and grade the results |
| `pxvk-nativeinfo` | native kernel attributes and compiled code, for comparison |
| `pxvk-run`, `pxvk-harness.dll` | deterministic runs of the GLUT SDK samples (used by `bench.py`) |
| `tools/nvcuda-trace` | native CUDA proxy that logs calls and captures launches (separate CMake project) |
| `tools/check.py` | regression check: corpus, opcheck, replays |
| `tools/bench.py` | native/pxvk sample runs, frame rates and image comparison |
| `tools/capture-diff.py` | find the first diverging launch between native and pxvk captures |
| `tools/extract-physx-ptx.py` | extract the PTX corpus from a PhysX DLL |
| `tools/gen-stubs.py` | regenerate the stubs for unsupported CUDA exports |

PhysX PTX, captures, samples and game files are proprietary: keep them in
`tmp/` or `test-output/`, which git ignores.
