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

On RTX 50 cards the driver only gives 32-bit CUDA to allowed executable
names (`nvidia-32bit-physx.md`), so opcheck reports `native CUDA unavailable`
and `check.py` skips it. Copy `pxvk-opcheck.exe` to `tmp\opcheck-native\borderlands2.exe`
and run that copy instead. Two variants then report `native JIT failed`: driver
621.01 cannot compile 32-bit `red.*.add.f32` for sm_120. PhysX uses only the
`atom` forms, which do compile and have a variant of their own. On NVIDIA
`add.f32` and `mul.f32` (no `.ftz`) also differ: its Vulkan driver flushes
denormals there and cannot be told not to (`architecture.md`).

Without an NVIDIA driver, `--cpu-reference` compares pxvk with a CPU
implementation instead, for the variants that have one: `shfl`, global
float atomic add, `vote.ballot`, `cvt.sat` and the f64 comparisons of
widened floats (APEX 1.3), with the semantics established on NVIDIA. `--block N`
(a multiple of 32, default 256) sets the block size. `PXVK_SUBGROUP_SIZE=64`
runs warp-synchronous kernels in 64-wide subgroups on devices that offer
them (RDNA), two warps per subgroup as on AMD GCN:

```powershell
build\x86\pxvk-opcheck.exe --device radeon --cpu-reference
$env:PXVK_SUBGROUP_SIZE = "64"
build\x86\pxvk-opcheck.exe --device radeon --cpu-reference --block 96
```

On the Radeon 8060S (driver 26.10.1, November 2026) all 23 such variants
are exact with 32- and 64-wide subgroups and blocks of 32, 96 and 256.
Before `shfl` took the lane as the subgroup id modulo 32, 19 of its 20
variants differed with 64-wide subgroups (blocks of 96 and 256). The
Radeon's native float atomic flushes a subnormal already in memory, as
NVIDIA does (Intel keeps it).

## 2. Real launches: capture and replay

`tools/nvcuda-trace` is a native proxy driver with its own CMake project. It
forwards to the real `nvcuda.dll` and logs every call; with
`PXVK_CAPTURE_FRAMES` (frame numbers published by the harness),
`PXVK_CAPTURE_LAUNCHES` (launch ordinals, e.g. `100-120,500`) or
`PXVK_CAPTURE_KERNEL` (the first `PXVK_CAPTURE_COUNT` launches of a kernel
whose name contains this text, for games whose ordinals vary) it snapshots
all device memory around selected launches and records parameters, texture
bindings and module globals. Every module image is saved, whether the CUDA
runtime registers it or the program loads it with `cuModuleLoadData(Ex)` or
`cuModuleLoadFatBinary`, as `fatbin_NNNN.bin` (or `.ptx` for PTX text), and
each capture names its module's number. Capturing needs an NVIDIA GPU;
replaying works on any Vulkan device.
Lines starting with `interop` record graphics interop and surface calls
(D3D/GL context creation, resource registration, map/unmap, 2D copies,
`cuModuleGetSurfRef`); per-frame ones are logged the first 32 times, then
every 1,000th. Runtimes that load `nvcuda.dll` by name and bind every entry
point from it (PhysX 3 GPU and APEX 1.3 in Batman: Arkham Origins) get the
proxy itself, so it must export everything they bind; a missing export makes
them fail with "Failed to bind CUDA API." Pass `--output` as an absolute
path: a relative one is resolved against the game's working directory.
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

Replay options: `--verbose` (show the first wrong words), `--mismatches`
(every differing word with its grade: ulp4, near, wrong, wrong-int),
`--time N` (GPU timing),
`--storage-shared` (force storage-backed shared memory), `--ptx-dir DIR`
(for captures whose module the tracer did not see, `fatbin 0`: older
tracers missed modules loaded with `cuModuleLoadData(Ex)` or
`cuModuleLoadFatBinary`, as APEX's are; the kernel is looked up by name
among the `.ptx` files under DIR, extracted with
`tools/extract-physx-ptx.py` from the same runtime folder), `--ptx FILE`
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
- `tools/synth-cloth-normals.py` builds captures of the 2.8.4 cloth normal
  kernel on a generated cloth grid, with expected normals computed on the
  CPU. Measured with them (October 2026, 64–1024-vertex cloths, minimum of
  300 runs): on the RTX 3070 a launch takes 4–7 µs whatever the atomics;
  making its shared adds free (non-atomic, wrong results) would save at
  most 1 µs, so native shared float atomics
  (`VK_KHR_workgroup_memory_explicit_layout`) were not added. On the Intel
  UHD the shared compare-and-swap loop dominates: 43–180 µs against a
  21–43 µs floor.
- In-game, `PXVK_PROFILE=1` (pxvk) and `PXVK_TRACE_TIMING=1` (tracer, native)
  log per-kernel GPU time per 500 launches.

The pxvk runtime writes captures in the same format (`PXVK_CAPTURE_DIR` +
`PXVK_CAPTURE_FRAMES`), and `tools/capture-diff.py` compares native and
pxvk captures of the same frames with pointer translation. That finds the
first launch whose inputs or outputs diverge in a full application run.

`PXVK_API_LOG=1` makes both the tracer and the runtime log every copy,
synchronization, texture binding and launch per frame for sequence diffs.

## Warp scheduling: `pxvk-warptest`

Lockstep is what warp-synchronous code assumes and Vulkan does not
promise (`architecture.md`). `pxvk-warptest` checks whether a GPU breaks
it in practice. It compiles two self-checking PTX kernels:

- divergent: lanes store under a branch, then every lane reads another
  lane's slot;
- scan: a padded warp inclusive scan through volatile shared memory.

Each runs with pxvk's two strategies, control barriers under maximal
reconvergence (`maximal`, skipped where the device lacks the extension)
and memory-only barriers (`fence`), and for comparison with no barriers
(`none`) and control barriers without the extension (`control`). Lanes are
skewed by 0, 3 or 31 iterations of busy work, and wrong results are
counted:

```
build\x86\pxvk-warptest.exe --device nvidia
build\x86\pxvk-warptest.exe --device intel --blocks 64 --runs 2
```

The defaults are 512 blocks of 256 threads, 256 rounds and 20 launches,
about 671 million checks per case. The exit code is 1 if `maximal` or
`fence` mismatched. In November 2026 every case was 0 on the RTX 5070 and
the Intel iGPU; an earlier version without `maximal` showed no problem on
an AMD GPU (tester's report, no counts recorded). The skewed scan first
failed with control barriers, identically on both GPUs, until the
structurizer stopped nesting post-loop code in loop bodies
(`architecture.md`). Per-launch times are printed too; on Intel the scan
takes 16.8 ms with `maximal` against 12.0 with `fence`.

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

### Games: `tools/game-bench.py`

For Direct3D 9 games, `pxvk-framelog.dll` (`tools/framelog`) records the time
of every `IDirect3DDevice9::Present`, without administrator rights (which
PresentMon needs). `game-bench.py run` starts the game through
`pxvk-launch --inject-dll` with it, natively or with pxvk (the DLL then
loads pxvk's `nvcuda.dll` itself), closes the game after `--seconds`, and
`report` gives frame statistics for the longest stretch between loading
stalls. Borderlands 2's PhysX benchmark is the reference workload (results
in `docs/compatibility.md`); turn its 62 fps cap off first
(`bSmoothFrameRate=FALSE` in `WillowEngine.ini`):

```powershell
$bl2 = "C:\Program Files (x86)\Steam\steamapps\common\Borderlands 2\Binaries\Win32\Borderlands2.exe"
python tools\game-bench.py run native-1 --mode native --seconds 110 --exe $bl2 -- benchmark
python tools\game-bench.py run pxvk-1 --mode pxvk --seconds 110 --exe $bl2 -- benchmark
python tools\game-bench.py report native-1 pxvk-1
```

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

Kernels listed in `check.py`'s `RACY_KERNELS` (APEX 1.3's
`TurbulenceFS_Multigrid8x8x8Solve`, whose result depends on thread timing)
are reported but not graded. On the RTX 5070 laptop the check runs with
Batman: Arkham City's cloth and fluid captures, Borderlands 2's and the 97
Arkham Origins APEX captures (`compatibility.md`):

```powershell
python tools\check.py --ptx tmp\extracted-ptx --device nvidia intel `
    --captures test-output\bac-capture-cloth\BatmanAC.exe_18288 test-output\bac-capture-fluid\BatmanAC.exe_19300 `
               test-output\bl2-capture\Borderlands2.exe_4796 test-output\bao-capture-apex2\BatmanOrigins.exe_3772
```

| Captures | RTX 5070 | Intel iGPU |
|---|---|---|
| Arkham City cloth (60 launches) | 60 EXACT | 45 EXACT, 14 ok, 1 tolerated BAD |
| Arkham City fluid (60 launches) | 60 EXACT | 32 EXACT, 28 ok |
| Borderlands 2 (4 launches) | 4 EXACT | 4 tolerated BAD |
| Arkham Origins APEX (97 launches) | 90 EXACT, 5 ok, 2 racy | 89 EXACT, 6 ok, 2 racy |

Expected results on the reference laptop, and on the Radeon 8060S laptop
(drivers 26.9.1 and 26.10.1, no NVIDIA driver):

| Captures | RTX 3070 | Intel UHD | Radeon 8060S |
|---|---|---|---|
| SampleCloth (90 launches) | 88 EXACT, 2 ok | 28 EXACT, 55 ok, 7 tolerated BAD | the same as Intel |
| SampleParticleFluid (112 launches) | 112 EXACT | 104 EXACT, 8 ok | the same as Intel |
| FluidMark 1.5.4 (82 launches) | 82 EXACT | 63 EXACT, 19 ok | the same as Intel |

With `PXVK_SUBGROUP_SIZE=64` the Radeon gives the same results, and all
67 warp-synchronous launches (`radixSortBlocks` and `scan4`) stay EXACT.

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
Of the 222 wrong words in those launches (on Intel), 220 are floats:
near-zero residuals such as 4.2e-11 against -7.0e-12, and values off in the
third or fourth digit such as -0.123394 against -0.123234. The summary
line's `wrong_int=` counts words that look like integers; there are two,
both in launch 4489, and probes inserted with `--ptx` (2026-10-11) traced
both to one float comparison:

- The solver sets bit 5 of a particle's flags when its squared movement is
  below 0.0225 (`setp.geu.ftz.f32` before `or.b32 ..., 32`). For particle
  flags word `0x29a4ce84`, native computes 0.0224901 and sets it; Intel
  computes 0.0225006 and does not.
- The second word is a later copy of that flags word into the packed
  output (loaded from `0x29a4ce84`), so it differs with it.

The float difference comes from the approximation units. Making `rcp`,
`rsqrt`, `sqrt` and `div.full` more precise on NVIDIA (one Newton step
each, an experiment, not a mode) turns the same cloth captures from 88
EXACT + 2 ok into 28 EXACT + 53 ok + 9 BAD, close to Intel's 28/55/7 with
the same 28 exact launches, and launch 4489 gets 84 wrong words (Intel:
92). The same change on Intel moves this particle to 0.0224941, below the
threshold, and both integer words then match native. Integer flags that
follow float tests can therefore differ off NVIDIA, which is why
`check.py` does not reject integer-looking mismatches there; it prints
their count. The scene-5 launches 6643 and 6646 are especially sensitive:
with the refined math on NVIDIA they grow to over 5,000 wrong words each,
318 of them integer-looking (not traced individually).

FluidMark uses PhysX 2.8.4 from the PhysX System Software, whose 17 PTX
modules differ from the samples' (Mirror's Edge uses the samples' modules).
Capture it by launch ordinal, since it does not run under the harness:

```powershell
$env:PXVK_CAPTURE_LAUNCHES = "3000-3040,30000-30040"
build\nvcuda-trace-mingw\pxvk-trace.exe --output test-output\fm-native -- "C:\Program Files (x86)\Geeks3D\Benchmarks\FluidMark\FluidMark.exe" /preset_720 /log_score /noscore /nogui
```

These 82 captures take 1.9 GB (see the roadmap).

## Vulkan validation layers

`PXVK_VALIDATION=1` (`environment.md`) runs pxvk under the Khronos
validation layer with synchronization validation, in the tools and in
games. The layer has to be 32-bit, and Vulkan SDK 1.4.363 ships only a
64-bit one, so build it from the release matching the SDK with the same
MinGW toolchain as pxvk:

```powershell
git clone --depth 1 --branch vulkan-sdk-1.4.363.1 https://github.com/KhronosGroup/Vulkan-ValidationLayers.git tmp\vvl\src
cmake -S tmp\vvl\src -B tmp\vvl\build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ -DUPDATE_DEPS=ON -DBUILD_TESTS=OFF -DBUILD_WERROR=OFF
cmake --build tmp\vvl\build
```

Then copy `VkLayer_khronos_validation.dll` and `.json` from
`tmp\vvl\build\layers` next to `nvcuda.dll`, or set `VK_ADD_LAYER_PATH` to
that folder.

The layers check pxvk's use of the Vulkan API (resources, descriptors,
command recording, barriers between commands), not what happens between
invocations inside a shader: warp-synchronous races such as the one in
APEX's compaction kernel (`architecture.md`) do not show up. In November
2026 every replay set and opcheck on the RTX 5070 and the Intel iGPU, and a
full Batman: Arkham Origins benchmark (4.6 million launches) on NVIDIA, gave
no validation error and no synchronization hazard. The best-practices
checks (`VK_KHRONOS_VALIDATION_VALIDATE_BEST_PRACTICES=true`) only note
that pxvk declares the workgroup size with the `WorkgroupSize` built-in,
which SPIR-V 1.6 deprecated and the SPIR-V version pxvk targets uses.

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

lavapipe's subgroup size is its LLVM vector width divided by 32 bits, and it
offers exactly that one size: 8 by default on an AVX2 CPU (256 bits), 4 with
`LP_NATIVE_VECTOR_WIDTH=128` and 16 with `512`. Do not go above 512: with
`1024` or `2048` it reports 32 or 64 lanes, but opcheck's `shfl` variants
show subgroups really span only 16 lanes (lane 16 and up read 0), so
warp-level results there are wrong. The Batman: Arkham City captures give
the same results as on Intel at the default width (Mesa 26.2.6,
mesa-dist-win, November 2026).

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
