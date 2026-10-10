# pxvk — classic GPU PhysX on Vulkan

`pxvk` runs classic 32-bit NVIDIA GPU PhysX (2.8.x era) on any Vulkan GPU.
It is a replacement `nvcuda.dll` that JIT-compiles the PTX embedded in the
application's CUDA fatbins to SPIR-V and executes it with Vulkan compute.
Mirror's Edge (2009) is the first compatibility target; this is not a general
CUDA implementation (see [`docs/charter.md`](docs/charter.md)).

## Status

Tested on one laptop (RTX 3070 Laptop + Intel UHD, Windows 11):

- All PhysX 2.8 kernels compile (70 entries in 17 modules; Mirror's Edge
  ships the same 17 modules as the SDK samples) and validate with
  `spirv-val`.
- Every PTX instruction variant PhysX uses is bit-exact against native CUDA
  on NVIDIA, including ptxas-style FMA contraction. On other vendors the
  hardware `rcp`/`rsqrt`/`sqrt` approximations differ by 1–2 ULP.
- Replaying real kernel launches captured from native runs: 196 of 202
  SampleCloth launches and all 112 SampleParticleFluid launches are bit-exact
  on the 3070; the rest are within 4 ULP.
- `SampleCloth` (9 scenes) and `SampleParticleFluid` run on NVIDIA and Intel.
  On the 3070 every scene runs as fast as native CUDA or faster, and with a
  warm pipeline cache startup and scene switches are faster than native.
- Mirror's Edge runs on pxvk on this machine (fluid, glass-related particle
  and collision kernels), with no errors and a clean shutdown. Visual
  validation against native is still to do.

Not yet working or tested: machines without an NVIDIA driver (PhysX detects
hardware PhysX through NVAPI first, which pxvk does not provide yet), AMD
GPUs, and mobile (Adreno) drivers. See [`docs/roadmap.md`](docs/roadmap.md).

## Documentation

| Document | Contents |
|---|---|
| [`docs/charter.md`](docs/charter.md) | goal, targets and scope |
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

Outputs in `build/x86`: `nvcuda.dll` (the runtime), `pxvk-launch.exe`, and
the developer tools (`pxvk-ptxc`, `pxvk-replay`, `pxvk-opcheck`,
`pxvk-nativeinfo`, `pxvk-run`, `pxvk-harness.dll`, ...). `nvcuda.dll` and
`pxvk-launch.exe` depend only on Windows system DLLs.

## Run an application

CUDART loads `nvcuda.dll` from System32 by absolute path, so pxvk is injected:

```powershell
build\x86\pxvk-launch.exe C:\path\to\SampleCloth.exe [arguments]
```

Keep `nvcuda.dll` next to `pxvk-launch.exe`. The application must use the
legacy NVIDIA PhysX System Software runtime (Mirror's Edge needs a small
change for that; see [`docs/mirrors-edge.md`](docs/mirrors-edge.md)). Kernels
compile on first use and are cached on disk, so the first run of each scene
is slower.

Environment variables:

| Variable | Effect |
|---|---|
| `PXVK_DEVICE` | Vulkan device by name fragment (`intel`, `nvidia`, `radeon`) or index |
| `PXVK_LOG` | log file path (init, module loads, compiles, errors, periodic stats) |
| `PXVK_API_LOG` | also log every copy/sync/launch/texture call (with `PXVK_LOG`) |
| `PXVK_CACHE_DIR` | pipeline cache directory (default `%LOCALAPPDATA%\pxvk`; `off` disables) |
| `PXVK_PROFILE` | per-kernel GPU timing and fence-wait/pipeline-creation times in the log |
| `PXVK_PIPELINE_STATS` | print the driver's register count/binary size for each pipeline (stderr) |
| `PXVK_SYNC` | wait for the GPU after every operation (debugging) |
| `PXVK_DUMP_SPIRV` | directory to write every compiled kernel's SPIR-V |
| `PXVK_CAPTURE_DIR`, `PXVK_CAPTURE_FRAMES` | capture launches for differential testing |

Captured PhysX PTX, capture files and the samples are proprietary; keep them
local (`tmp/` and `test-output/` are git-ignored).
