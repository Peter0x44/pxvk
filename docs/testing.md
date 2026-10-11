# Testing pxvk

All environment variables used below (capture, tracer, harness, profiling)
are described in [`environment.md`](environment.md).

pxvk is tested differentially against the real NVIDIA driver at three
levels. Everything derived from PhysX binaries (PTX, captures, samples) is
proprietary and stays local under `tmp/` or `test-output/`.

## 1. Instruction semantics: `pxvk-opcheck`

Generates a small PTX kernel for each instruction variant PhysX uses (and the
FMA-contraction patterns ptxas applies), runs it through the 32-bit NVIDIA
driver and through pxvk with random and special-value inputs, and compares
results bit for bit.

```powershell
build\x86\pxvk-opcheck.exe --device nvidia
build\x86\pxvk-opcheck.exe --device intel --filter f32
```

## 2. Real launches: capture and replay

`tools/nvcuda-trace` is a native proxy driver with its own CMake project. It
forwards to the real `nvcuda.dll` and logs every call; with
`PXVK_CAPTURE_FRAMES` (frame numbers published by the harness) or
`PXVK_CAPTURE_LAUNCHES` (launch ordinals, e.g. `100-120,500`) it snapshots
all device memory around selected launches and records parameters, texture
bindings and module globals. Capturing needs an NVIDIA GPU; replaying works
on any Vulkan device.
`pxvk-replay` re-executes each capture on Vulkan at the original addresses
and grades the result (EXACT, ok = within 4 ULP / 1e-3 relative, BAD, FAIL).
Memory is untyped, so tolerances apply only to words that look like floats:
words with a zero exponent field (integers below 2^23, such as counts,
indices and flags) must match exactly, and the 1e-4 absolute tolerance is
not applied between two tiny values (32-bit device pointers read as such).
A capture that cannot be replayed (corrupt file, compile error) is a FAIL
and the remaining captures still run.

```powershell
cmake -S tools\nvcuda-trace -B build\nvcuda-trace-mingw -G Ninja -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc
cmake --build build\nvcuda-trace-mingw
$env:PXVK_CAPTURE_FRAMES = "150,450,750"
python tools\bench.py sweep captures --mode capture
build\x86\pxvk-replay.exe --device nvidia test-output\bench\captures\trace\SampleCloth.exe_<pid>
```

Replay options: `--verbose` (show wrong words), `--time N` (GPU timing),
`--storage-shared` (force storage-backed shared memory), `--ptx FILE`
(override the module PTX, e.g. with hand-inserted probes storing to a
`.global pxvk_debug[]` symbol, which replay maps to a scratch area and
prints), `--spirv-filter CMD` (run `CMD in.spv -o out.spv` on every kernel,
e.g. `spirv-opt -O`, for code-generation experiments) and
`--force-native-ftz` (skip `.ftz` emulation even where the driver lacks
it; results are then not exact, but it bounds the emulation's cost).

## Kernel performance

Replay with `--time N` is the quickest loop for kernel code generation:
launches such as the cloth `UberSolver` run as one or a few workgroups and
are latency-bound, so instruction count on the dependency chain matters.

- `PXVK_PIPELINE_STATS=1` prints the driver's statistics for every pipeline
  (`VK_KHR_pipeline_executable_properties`; on NVIDIA register count and
  binary size).
- `pxvk-nativeinfo module.ptx` prints the native driver's function
  attributes (registers, local memory) and writes the native JIT cubin to
  `module.ptx.cubin`; `nvdisasm -c` from the CUDA toolkit disassembles it.
- In-game, `PXVK_PROFILE=1` (pxvk) and `PXVK_TRACE_TIMING=1` (tracer, native)
  log per-kernel GPU time per 500 launches.

The pxvk runtime writes captures in the same format (`PXVK_CAPTURE_DIR` +
`PXVK_CAPTURE_FRAMES`), and `tools/capture-diff.py` compares native and
pxvk captures of the same frames with pointer translation. That finds the
first launch whose inputs or outputs diverge in a full application run.

`PXVK_API_LOG=1` makes both the tracer and the runtime log every copy,
synchronization, texture binding and launch per frame for sequence diffs.

## 3. Applications: `tools/bench.py`

`pxvk-harness.dll` makes GLUT samples deterministic (virtual 20 ms frames),
scripts key presses, captures frames and logs frame times; native runs are
bit-for-bit reproducible.

```powershell
python tools\bench.py sweep native --mode native
python tools\bench.py sweep pxvk --mode pxvk --device nvidia
python tools\bench.py compare test-output\bench\native test-output\bench\pxvk
```

Native runs force OpenGL onto the NVIDIA GPU (`SHIM_MCCOMPAT`), as loading
pxvk's Vulkan device does on Optimus laptops, so images are comparable.

## Regression check

The PTX corpus is extracted from a PhysX runtime DLL; this needs no NVIDIA
GPU. The 2.8.0 core from the PhysX System Software contains the same 17
modules as the SDK samples and Mirror's Edge:

```powershell
python tools\extract-physx-ptx.py "C:\Program Files (x86)\NVIDIA Corporation\PhysX\Engine\v2.8.0\PhysXCore.dll" tmp\extracted-ptx
python tools\check.py --ptx tmp\extracted-ptx --captures <capture dirs> --device nvidia intel
```

Builds, compiles and validates every kernel of the PTX corpus (with
`spirv-val` when it is on `PATH`), runs opcheck and replays the captures. On
NVIDIA exact semantics are required; on other vendors approximation-unit
differences are tolerated. A tool that crashes, exits with an unexpected
code, or stops before its summary line (and a `--device` that matches no
device) makes the check fail. The reference capture sets on the development
machine are `test-output/bench/capture-all/trace/SampleCloth.exe_*` (all nine
cloth scenes) and `test-output/bench/fluid-capture/trace/SampleParticleFluid.exe_*`.

Expected results on the reference laptop, and on the Radeon 8060S laptop
(driver 26.9.1, no NVIDIA driver):

| Captures | RTX 3070 | Intel UHD | Radeon 8060S |
|---|---|---|---|
| SampleCloth (90 launches) | 88 EXACT, 2 ok | 28 EXACT, 55 ok, 7 tolerated BAD | the same as Intel |
| SampleParticleFluid (112 launches) | 112 EXACT | 104 EXACT, 8 ok | the same as Intel |
| FluidMark 1.5.4 (82 launches) | 82 EXACT | 63 EXACT, 19 ok | the same as Intel |

The tolerated BAD launches are all `KernelDeformableUberSolver` (scenes 1,
3, 4 and 9). A launch is BAD when any word is "wrong": over 4 ULP and over
1e-3 relative, or a mismatch in a word that looks like an integer. Off
NVIDIA, `check.py` accepts it while wrong words stay under 0.5% of the words
the kernel changed; here they are 0.003–0.14%. The cause is the
approximation units: the cloth kernel without approximate math
(`PxgCudaDeformableUpdate`) is EXACT on every GPU, the single-pass
`UberCollision` (`rcp`, `rsqrt`, `sqrt`, `div.full`) stays within 4 ULP,
and only the iterative `UberSolver` (20 `rcp`, 6 `rsqrt`, 2 `sqrt`, 6
`div.full`) drifts further, while the same launches are EXACT on NVIDIA.
The wrong words `pxvk-replay --verbose` prints (the first six per launch) are
all floats: near-zero residuals such as 4.2e-11 against -7.0e-12, and
values off in the third or fourth digit such as -0.123394 against -0.123234.

FluidMark uses PhysX 2.8.4 from the PhysX System Software, whose 17 PTX
modules differ from the samples' (Mirror's Edge uses the samples' modules).
Capture it by launch ordinal, since it does not run under the harness:

```powershell
$env:PXVK_CAPTURE_LAUNCHES = "3000-3040,30000-30040"
build\nvcuda-trace-mingw\pxvk-trace.exe --output test-output\fm-native -- "C:\Program Files (x86)\Geeks3D\Benchmarks\FluidMark\FluidMark.exe" /preset_720 /log_score /noscore /nogui
```

These 82 captures take 1.9 GB (see the roadmap).

## lavapipe (no second GPU needed)

Mesa's software Vulkan driver exercises the fallbacks for limited devices
(warp emulation and the split heap; see `docs/architecture.md`). Point the
Vulkan loader at a 32-bit lavapipe build only for the process, e.g. from
[mesa-dist-win](https://github.com/pal1000/mesa-dist-win) unpacked into
`tmp/mesa`:

```powershell
$env:VK_DRIVER_FILES = "$PWD\tmp\mesa\x86\lvp_icd.x86.json"
build\x86\pxvk-replay.exe test-output\bench\fluid-capture\trace\SampleParticleFluid.exe_<pid>
```

Expected: the same results as on the Intel UHD (fluid 104 EXACT + 8 ok;
cloth 28 EXACT, 55 ok, 7 tolerated BAD). `pxvk-replay --emulate-warps`
forces warp emulation on any device; on NVIDIA the results must stay
bit-exact. `pxvk-ptxc --emulate-warps --split-heap` compiles the corpus
with both fallbacks for `spirv-val`.

## Testing on another machine

Clone the repository, build, extract the PTX corpus as above, and copy the
capture directories privately (they are proprietary and not in git). Without an
NVIDIA driver, `check.py` skips opcheck and still compiles the corpus and
replays the captures, which compares the new GPU against the native NVIDIA
results recorded in them. Pass the device with `--device <name fragment>`.

To compare whole runs with native, also copy native sweep frames
(`bench.py sweep <name> --mode native` on the NVIDIA machine), run the same
sweep with `--mode pxvk` and use `bench.py compare`. Pixels never match
across vendors, because OpenGL renders slightly differently and the
approximation units let cloth drift, so look at the frames side by side.

On the Radeon 8060S (2026-10-11), against RTX 3070 native frames, 3 per
scene: SampleCloth differs by 0.9–9.6% of the pixels (29.6% on the last
frame of scene 9, a large patterned rug that has shifted slightly) and
SampleParticleFluid by 0.16–6.7%, constant within some scenes (4.41%
throughout fluid scene 1), which points to rendering rather than physics.
Every scene ends the same way as native. In particular, part of the rug
stays on the bar in cloth scene 6 and the sphere ends crumpled in scene 9
on native NVIDIA too, so those differences from CPU PhysX are GPU PhysX
behaviour, not pxvk bugs.
