# pxvk — classic GPU PhysX on Vulkan

`pxvk` runs classic 32-bit NVIDIA GPU PhysX (2.8.x era) on any Vulkan GPU.
It is a replacement `nvcuda.dll` that JIT-compiles the PTX embedded in the
application's CUDA fatbins to SPIR-V and executes it with Vulkan compute.
Mirror's Edge (2009) is the first compatibility target; this is not a general
CUDA implementation (see [`docs/charter.md`](docs/charter.md)).

Status (PhysX 2.8 SDK samples, RTX 3070 Laptop + Intel UHD):

- All PhysX kernels compile (70 entries in 17 modules) and validate.
- Every PTX instruction variant PhysX uses is bit-exact against native CUDA on
  NVIDIA hardware, including ptxas-style FMA contraction. On other vendors the
  only differences are the hardware `rcp`/`rsqrt`/`sqrt` approximations.
- Replaying 202 real kernel launches captured from native runs: 196 bit-exact
  on the 3070, the rest within a few ULP.
- `SampleCloth` (9 scenes) and `SampleParticleFluid` run on NVIDIA and Intel
  through the JIT runtime. On the 3070 steady-state frame rates are within a
  few percent of native CUDA or faster, and with a warm pipeline cache
  startup and scene switches are faster than native.

See [`docs/architecture.md`](docs/architecture.md) for the design and
[`docs/testing.md`](docs/testing.md) for the testing tools.

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
`pxvk-run`, `pxvk-harness.dll`, ...).

## Run an application

CUDART loads `nvcuda.dll` from System32 by absolute path, so pxvk is injected:

```powershell
build\x86\pxvk-launch.exe C:\path\to\SampleCloth.exe
```

Put `nvcuda.dll` next to `pxvk-launch.exe`. No offline translation step,
captured PTX or shader directory is needed; kernels compile on first use.

Environment variables:

| Variable | Effect |
|---|---|
| `PXVK_DEVICE` | Vulkan device by name fragment (`intel`, `nvidia`, `radeon`) or index |
| `PXVK_LOG` | log file path (init, module loads, compiles, errors, periodic stats) |
| `PXVK_API_LOG` | also log every copy/sync/launch/texture call (with `PXVK_LOG`) |
| `PXVK_SYNC` | wait for the GPU after every operation (debugging) |
| `PXVK_DUMP_SPIRV` | directory to write every compiled kernel's SPIR-V |
| `PXVK_CACHE_DIR` | pipeline cache directory (default `%LOCALAPPDATA%\pxvk`; `off` disables) |
| `PXVK_PROFILE` | per-kernel GPU timing and fence-wait/pipeline-creation times in the log |
| `PXVK_CAPTURE_DIR`, `PXVK_CAPTURE_FRAMES` | capture launches for differential testing |

Captured PhysX PTX and capture files are proprietary; keep them local
(`tmp/`, `test-output/` are git-ignored).
