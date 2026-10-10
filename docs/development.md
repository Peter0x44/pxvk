# Developing pxvk

pxvk takes the place of the NVIDIA CUDA driver (`nvcuda.dll`) inside a 32-bit
PhysX 2.8 application. It compiles PhysX's GPU kernels from PTX to SPIR-V at
runtime and runs them with Vulkan compute. It is not a general CUDA
implementation ([`charter.md`](charter.md)).

## Building

Requires CMake, Ninja, Python 3 and a 32-bit (i686) MinGW toolchain or MSVC
Win32. Dependencies are fetched during configuration.

```powershell
cmake -S . -B build/x86 -G Ninja -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc `
      -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build/x86
```

`build/x86` then contains the three runtime files (`pxvk-launch.exe`,
`nvcuda.dll`, `pxvk-nvapi.dll`), which depend only on Windows system DLLs,
and the developer tools below.

`python tools/package.py` turns a build into the tester package
(`dist/pxvk-<commit>/` and `.zip`): the runtime files, the tester guide
(`docs/install.md` as `README.md`), `environment.md`, `mirrors-edge.md` and
`run-mirrors-edge.cmd`. It refuses binaries that import non-system DLLs.

CI (`.github/workflows/build.yml`) builds every push and pull request with
Ubuntu's `i686-w64-mingw32` cross compiler (posix threads, as in
w64devkit) and uploads that package as the
`pxvk-<commit>` artifact on the run's page. CI has no GPU, so it does not
run the GPU tests.

## Documentation

| Document | Contents |
|---|---|
| [`architecture.md`](architecture.md) | design, Vulkan requirements, known limitations |
| [`testing.md`](testing.md) | differential testing, profiling, regression check |
| [`environment.md`](environment.md) | every environment variable |
| [`mirrors-edge.md`](mirrors-edge.md) | Mirror's Edge setup, scenes and workloads |
| [`install.md`](install.md) | installing, running and reporting (for testers) |
| [`roadmap.md`](roadmap.md) | open work |
| [`charter.md`](charter.md) | goal and scope |

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
| `tools/package.py` | build the tester package (`dist/pxvk-<commit>.zip`) |
| `tools/bench.py` | native/pxvk sample runs, frame rates and image comparison |
| `tools/capture-diff.py` | find the first diverging launch between native and pxvk captures |
| `tools/extract-physx-ptx.py` | extract the PTX corpus from a PhysX DLL |
| `tools/gen-stubs.py` | regenerate the stubs for unsupported CUDA exports |

PhysX PTX, captures, samples and game files are proprietary: keep them in
`tmp/` or `test-output/`, which git ignores.

## Technical status

Tested on an RTX 3070 laptop (with its Intel UHD iGPU) and on a Radeon 8060S
laptop with no NVIDIA driver.

- All 70 PhysX 2.8 kernels (17 modules, the same in the SDK samples and
  Mirror's Edge) compile and pass `spirv-val`. FluidMark's PhysX 2.8.4
  brings 17 different modules, which also run without errors.
- On NVIDIA, every PTX instruction variant PhysX uses is bit-exact against
  the native driver, including ptxas-style FMA contraction. Replaying native
  kernel captures: 84 of 90 SampleCloth launches and all 112
  SampleParticleFluid launches are bit-exact; the other six are within 4 ULP
  (see the known limitations in [`architecture.md`](architecture.md)).
- Other GPUs differ from NVIDIA by 1–2 ULP in the hardware `rcp`, `rsqrt`
  and `sqrt` approximations.

FluidMark 1.5.4, 720 preset, mean of 10 interleaved runs on the RTX 3070
laptop:

| PhysX on | Simulation steps/s |
|---|---|
| RTX 3070, pxvk | 283 |
| RTX 3070, native CUDA | 269 |
| Intel UHD iGPU, pxvk | 132 |
| CPU PhysX (i7-11800H, 16 threads) | 39 |

On the 3070 the SampleCloth scenes run as fast as native CUDA or faster. On
the Radeon 8060S the heavy cloth scenes run 2–3.6× faster than CPU PhysX,
and Mirror's Edge holds its 62 fps cap.
