# pxvk — classic GPU PhysX on Vulkan

`pxvk` runs classic 32-bit NVIDIA GPU PhysX (2.8.x era) on any Vulkan GPU.
It is a replacement `nvcuda.dll` that JIT-compiles the PTX embedded in the
application's CUDA fatbins to SPIR-V and executes it with Vulkan compute.
Mirror's Edge (2009) is the first compatibility target; this is not a general
CUDA implementation (see [`docs/charter.md`](docs/charter.md)).

## Status

Tested on two laptops: RTX 3070 Laptop + Intel UHD, and AMD Radeon 8060S
(no NVIDIA driver), both on Windows 11.

- All PhysX 2.8 kernels compile (70 entries in 17 modules; Mirror's Edge
  ships the same 17 modules as the SDK samples) and validate with
  `spirv-val`.
- Every PTX instruction variant PhysX uses is bit-exact against native CUDA
  on NVIDIA, including ptxas-style FMA contraction. On other vendors the
  hardware `rcp`/`rsqrt`/`sqrt` approximations differ by 1–2 ULP.
- Replaying real kernel launches captured from native runs: 196 of 202
  SampleCloth launches and all 112 SampleParticleFluid launches are bit-exact
  on the 3070; the rest are within 4 ULP.
- `SampleCloth` (9 scenes) and `SampleParticleFluid` run on NVIDIA, Intel
  and AMD. On the 3070 every scene runs as fast as native CUDA or faster, and
  with a warm pipeline cache startup and scene switches are faster than
  native. On the 8060S the cloth scenes run at 310–720 fps and the fluid
  sample at about 670 fps. That is 2–3.6× CPU PhysX on the same machine for
  the heavy scenes and roughly on par for the light ones (4, 5, 8).
  Repeated runs are pixel-identical.
- Mirror's Edge runs on pxvk on both machines (fluid, glass-related particle
  and collision kernels) with no errors. On the 8060S, PhysX keeps up with
  the game's 62 fps cap, and GPU waits take about 0.6 ms per frame. Visual
  validation against native is still to do.
- Geeks3D FluidMark 1.5.4 (PhysX 2.8.4, whose 17 modules are different
  PTX from the 2.8 samples) runs on pxvk with no errors. 720 preset, mean
  of 10 interleaved runs on the RTX 3070 laptop: 283 simulation steps/s on
  pxvk vs 269 on native CUDA, 132 on the Intel UHD iGPU through pxvk, and
  39 with CPU PhysX on its i7-11800H.
- Machines without an NVIDIA driver work: `pxvk-nvapi.dll` answers the NVAPI
  queries that PhysX uses to select hardware PhysX.

Not yet tested: replaying the native captures on AMD (bit-level comparison)
and mobile (Adreno) drivers. See [`docs/roadmap.md`](docs/roadmap.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/install.md`](docs/install.md) | **for testers**: required files, how loading works, running samples and Mirror's Edge, checking and reporting results |
| [`docs/charter.md`](docs/charter.md) | goal, targets and scope |
| [`docs/environment.md`](docs/environment.md) | every environment variable of the runtime, capture tools and harness |
| [`docs/architecture.md`](docs/architecture.md) | design: compiler, Vulkan backend, runtime, Vulkan requirements, known limitations |
| [`docs/testing.md`](docs/testing.md) | differential testing tools, profiling, regression check |
| [`docs/mirrors-edge.md`](docs/mirrors-edge.md) | setting up, launching and testing Mirror's Edge |
| [`docs/roadmap.md`](docs/roadmap.md) | next steps and open problems |

## Build

Requires CMake, Ninja, Python 3 and an i686 MinGW toolchain (or MSVC Win32).
Dependencies (SPIRV-Headers, Vulkan-Headers, volk) are fetched at configure.

```powershell
cmake -S . -B build/x86 -G Ninja -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc `
      -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/x86
```

Outputs in `build/x86`: the three runtime files `pxvk-launch.exe`,
`nvcuda.dll` (the runtime) and `pxvk-nvapi.dll` (NVAPI for PhysX's hardware
check), plus the developer tools (`pxvk-ptxc`, `pxvk-replay`, `pxvk-opcheck`,
`pxvk-nativeinfo`, `pxvk-run`, `pxvk-harness.dll`, ...). The runtime files
depend only on Windows system DLLs. To hand pxvk to a tester, give them those
three files and [`docs/install.md`](docs/install.md).

## Run an application

CUDART loads `nvcuda.dll` from System32 by absolute path, so pxvk is injected:

```powershell
build\x86\pxvk-launch.exe C:\path\to\SampleCloth.exe [arguments]
```

Keep `nvcuda.dll` and `pxvk-nvapi.dll` next to `pxvk-launch.exe`. The
application must use the legacy NVIDIA PhysX System Software runtime (Mirror's
Edge needs a small change for that; see [`docs/install.md`](docs/install.md)).
Kernels compile on first use and are cached on disk, so the first run of each
scene is slower. [`docs/install.md`](docs/install.md) explains the loading
steps and how to confirm from the log that hardware PhysX is active.

The most common environment variables are `PXVK_LOG` (log file),
`PXVK_DEVICE` (Vulkan device by name fragment or index) and `PXVK_NVAPI`
(`system` keeps the NVIDIA driver's NVAPI). Every variable read by the runtime,
the capture tools and the benchmark harness, with values and defaults, is
listed in [`docs/environment.md`](docs/environment.md).

Captured PhysX PTX, capture files and the samples are proprietary; keep them
local (`tmp/` and `test-output/` are git-ignored).
