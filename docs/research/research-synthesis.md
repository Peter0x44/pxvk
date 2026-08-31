# Mirror's Edge research synthesis and architecture gate

## Decision

Research is complete enough to begin one narrow, disposable Vulkan execution
slice. The evidence supports an in-process PE32 CUDA compatibility DLL with a
finite PTX 4.1 frontend and a raw Vulkan 1.2 compute backend. It does not support
building a general CUDA implementation.

The first implementation artifact is not the game DLL. It is a vendor-neutral
x86 Vulkan oracle runner that executes the committed standalone semantic cases
and emits the same JSON schema as the native NVIDIA oracle. This isolates
SPIR-V generation, memory, synchronization, floating point, and device
selection before the CUDA facade, proprietary fatbins, or NVAPI are involved.

A cross-vendor hardware preflight is runnable now at
`tools/vulkan-probe/preflight.ps1`. It enumerates every Vulkan physical device,
applies the captured Mirror's Edge hardware gate, and emits hashed JSON. It does
not claim shader or PhysX correctness.

## Evidence synthesized

| Research result | Load-bearing conclusion |
|---|---|
| `mirrors-edge-native-trace.md` | Three routes exercise fluid, deformable, and visibly broken-glass workloads through one x86 context and default stream |
| `mirrors-edge-ptx-corpus.md` | Required frontend is finite: PTX 4.1, 17 modules, 16 launched entries, 120 launched variants, 32-bit addresses |
| `cuda-driver-contract.md` | Host contract is 35 active exports, six Dark tables, stable handles, two 32 MiB pools, packed launches, and ordered default-stream work |
| `zluda-runtime.md` | A separate 64-bit server is unnecessary; local u32 virtual pointers and typed handles work for PhysX 2.8 |
| `zluda-ptx.md` | Typed parsing, entry selection, normalization, and narrow semantic helpers cover the corpus; HIP/LLVM machinery is not reusable architecture |
| `native-oracle-probes.md` | FTZ, fusion, conversion, NaN, barrier, and integer-texture behavior have exact NVIDIA outputs |
| `vulkan-requirements.md` | Vulkan 1.2 plus a small feature/limit gate can execute the corpus; subgroup size 32, images, BDA, atomics, and Vulkan memory model are unnecessary |
| `validation-plan.md` | Standalone, real-kernel, sequence, and game claims have separate observable acceptance gates |

The raw traces and PTX remain proprietary local evidence. Project-owned tools,
normalized metadata, and expected outputs are the redistributable contract.

## Minimum product scope

### Required host surface

The eventual product consists of an x86 `nvcuda.dll` and the minimum x86 NVAPI
shim needed to pass the PhysX hardware gate on a non-NVIDIA renderer.

The CUDA DLL must provide:

- all 196 queried export names until the deferred NULL-versus-stub probe narrows
  the symbol surface;
- full semantics only for the 35 active public exports;
- six observed `cuGetExportTable` layouts and active slots;
- one device, one current context, CLS state, and stable typed handles;
- raw fatbin registration, the 70 function names, 7 globals, and 33 texture
  references;
- one 32 MiB host allocation and one 32 MiB device allocation;
- HtoD, DtoD, DtoH, memset, packed launches, and context synchronization on the
  default stream;
- linear legacy texture bindings and the two observed fetch forms;
- clean detach and allocation release.

Queried-only exports return `CUDA_ERROR_NOT_SUPPORTED` if called. They do not
silently pretend to implement unused CUDA subsystems.

### Required PTX surface

The frontend parses metadata for all 17 modules and lowers selected entries. It
must represent:

- module globals/constants/texrefs and exact parameter `(size, align)` layouts;
- predicates, explicit control flow, scalar/vector values, and weak helpers;
- global, constant, local, static/dynamic shared, and parameter memory;
- the 31 launched instruction families and exact 120 launched variants;
- `.maxntid`, shared-memory requirements, and X-only CUDA built-ins;
- backend-sensitive FTZ, FMA, approximate math, `bar.sync 0`, and linear
  texture operations as explicit semantic operations.

The 27 variants found only in unlaunched code need parser coverage. They do not
need executable lowering until selected control flow reaches them.

## Explicit exclusions

The first product does not implement:

- 64-bit CUDA, a helper server, IPC, shared mappings, or RPC;
- general CUDA, CUDART, NVRTC, linking, profiling, occupancy, or graphs;
- streams other than the default stream or CUDA events;
- graphics interop, arrays, surfaces, texture objects, or sampled images;
- managed memory, UVA, peer access, or buffer device address;
- atomics, warp intrinsics, tensor operations, f16, or f64;
- PhysX 3.x GPU rigid bodies, FleX, Flow, OptiX, or mobile targets;
- performance claims before semantic and real-workload gates pass.

A new game or trace may revise this list. Vulkan or CUDA feature availability
alone may not.

## Chosen implementation architecture

### Language and build

Use C++17 and CMake for the permanent x86 runtime and development runners.
Export CUDA entry points with `extern "C"`, `__declspec(dllexport)`, and
`__stdcall`. Build both MinGW i686 and MSVC Win32 configurations.

Reasons:

- the project already builds and executes PE32 C/Volk tools with the installed
  MinGW toolchain;
- MSVC and MinGW provide direct Windows DLL/export control without an additional
  runtime process;
- C++17 supplies tagged unions, RAII, deterministic ownership, and typed AST
  structures without requiring LLVM or a second implementation language;
- no Rust toolchain is installed on the reference workstation;
- a single-language DLL avoids a cross-language parser/backend ABI.

ZLUDA's Rust parser remains the pinned behavior/reference implementation. Its
MIT/Apache-2.0 licensing permits reuse, but importing its Rust parser plus macro
crates would introduce a second toolchain and bridge. The first pxvk frontend
will instead implement only the evidenced grammar in a table-driven lexer and
typed C++ AST, checked against all 17 corpus hashes. This choice is reversible
if parser coverage becomes the dominant risk.

No C++ exception, RTTI, or standard-library type crosses the exported C ABI.
Translation-time allocations are acceptable; launch, copy, and synchronization
paths reuse storage rather than allocating per call.

### Vulkan loading

Use Vulkan headers plus Volk and compile `volk.c` into each executable/DLL. Do
not link `vulkan-1.lib` and do not use vendor SDK APIs. The deployed binary loads
the system x86 `vulkan-1.dll`, which dispatches to NVIDIA, Intel, or AMD ICDs.

CMake FetchContent pins and retrieves Vulkan-Headers, SPIRV-Headers, and Volk
revisions. A Vulkan SDK is optional development tooling for `spirv-val`, not a
build dependency of the preflight. End users do not need an SDK when running a
built binary.

This is intentionally raw Vulkan rather than a higher-level wrapper. The
required API surface is one instance, one device/queue, buffers, descriptors,
compute pipelines, command buffers, barriers, fences, and shader modules.

### Process and component shape

```text
Mirror's Edge / PhysX 2.8 / CUDART
  -> PE32 nvcuda.dll exports + Dark API tables
  -> context, typed handles, modules, allocations, texture state
  -> finite PTX parser and normalized entry IR
  -> SPIR-V 1.5 emitter and pipeline cache
  -> Volk/raw Vulkan 1.2 compute backend
  -> selected vendor ICD
```

Everything is in process. Internal components have ordinary typed interfaces;
there is no generic plugin layer between them.

### PTX frontend

Use these stages:

```text
bytes
  -> fatbin member reader
  -> token stream with source spans
  -> typed module AST
  -> selected-entry reachability and weak-helper retention
  -> normalized predicated CFG and typed operations
  -> backend semantic lowering
```

Parsing is strict. An unsupported or malformed construct reports the PTX hash,
entry, source span, and form; it never compiles a partial module as success.
Module objects retain immutable global initializers, texref declarations,
parameter layouts, and shared-memory metadata.

Compile and cache one selected entry at a time while retaining every reachable
module declaration. Module `0017` parses but its five unlaunched entries are not
compiled in the first slice.

### SPIR-V emission

Emit SPIR-V binary directly from the normalized IR using Vulkan/SPIR-V header
enums. Keep the emitter small and structural:

- deterministic ID allocation;
- explicit types/constants/interfaces;
- one compute entry point per module;
- `Logical` addressing and `GLSL450` memory model;
- explicit float execution modes and `NoContraction` decorations;
- `OpControlBarrier` for the one workgroup barrier form;
- source operation/span retained in diagnostics.

Every generated module passes SDK `spirv-val --target-env vulkan1.2` in tests
before pipeline creation. The runtime does not ship or invoke `spirv-val`.
Direct emission avoids a GLSL/HLSL text round trip and a runtime shader compiler,
which would obscure PTX operation boundaries and contraction controls.

Modern upstream LLVM does include a SPIR-V backend, so LLVM is not excluded on
capability grounds. Its documented Vulkan path supports `spirv-unknown-vulkan`
triples, Vulkan buffer target-extension types, SPIR-V 1.5/Vulkan 1.2
environments, and KHR float-control extensions. It still would not remove this
PTX semantic frontend, and adopting LLVM would add LLVM IR construction,
legalization, a large build/runtime dependency, and another place to audit FTZ,
fusion, barrier scope, texture bounds, and u32 offsets. No SPIR-V-capable `llc`
is installed on the reference workstation. Direct emission remains the first
implementation path; LLVM is a credible fallback if SSA/legalization grows
beyond the finite emitter. See the official
[LLVM SPIR-V backend guide](https://llvm.org/docs/SPIRVUsage.html).

The first vertical slice now emits a 2,967-word SPIR-V 1.5 module directly from
the schema-5 structured IR for `tools/cuda-oracle/probe.ptx`. The module uses
one 64-thread Vulkan entry point to invoke the original one-thread math,
64-thread barrier, 64-thread invocation-private local-memory, and 12-thread
integer and float texture routines against the oracle binding profile. SDK
`spirv-val --target-env vulkan1.2` accepts the binary, and Intel UHD plus RTX
3070 executions match every native CUDA math, barrier, local-memory, and texture
field.

The second slice lowers the captured 89-instruction
`KernelFluidCalcHashPacked` entry from `module_0013_01_05.ptx`. A conservative
structured state machine stores PTX registers in function storage and dispatches
14 PTX basic blocks through one SPIR-V loop/switch. Binding 0 is the 32-bit
CUDA heap and binding 1 is the packed 52-byte launch parameter buffer. The
3,139-word module is deterministic and passes Vulkan 1.2 validation.

The paired third slice lowers the captured 116-instruction
`KernelReorderPosVelAndFindCellStartPacked` entry. It adds shared `u32`
storage, predicate conjunction, integer equality/inequality, a 260-byte
dynamic-shared contract, workgroup barriers, and vector `f32` stores. Its
pre-barrier diamond is emitted as structured selections that reconverge before
one dynamically uniform `OpControlBarrier`; the remaining 12 blocks use the
state machine. The deterministic 3,863-word module passes Vulkan 1.2
validation.

Schema 5 retains every routine `.local`/`.shared` symbol with space, type,
aligned byte offset, size, alignment, and external status. Kernel emission now
uses those symbols rather than the former `sharedHash` special case, allocates
`.local` storage per invocation, and takes the captured dynamic-shared byte
count explicitly when sizing each Workgroup arena. The standalone local probe
stores and reloads the aligned symbol at byte offset 16; native RTX and
translated Intel/RTX return `0x5a000000 + tid` for all 64 invocations.

This closes the local-storage prerequisite for trace-ordered
`KernelFluidDensity`. Its remaining structural work is module constant storage
and the finite instruction/helper subset used by that captured entry.

`tools/cuda-physx-fixture` runs both kernels as a one-particle pipeline against
the local proprietary PTX. Native RTX and PTX-derived Vulkan on Intel/RTX all
produce:

- hash `0x00003081`, original index `0`;
- reordered position bits `[0x3f800000, 0x40000000, 0x40400000, 0x00000000]`;
- reordered velocity bits `[0x40800000, 0x40a00000, 0x40c00000, 0x40e00000]`;
- cell interval `[0, 1]` at hash `0x00003081`.

The committed evidence hashes are:

- paired native CUDA: `ff29f41035678f0fa4c829676baa5c7f43eaa2b5008f44b2ae100be24a0a1543`;
- translated calc-hash Vulkan: `ffec75c749d458b61c71712375300ab21aee723755ffbcb403273c038b953b1c`;
- translated reorder Vulkan: `a048d9dc1467761ae2c39b5d53e3b3c38351cfc3428e844b898b94965a5e847d`.

Reproduce locally without committing proprietary PTX:

```powershell
cmake -S tools/cuda-physx-fixture -B build/cuda-physx-fixture -A Win32
cmake --build build/cuda-physx-fixture --config Release
build/cuda-physx-fixture/Release/cuda-physx-fixture.exe `
  path/to/module_0013_01_05.ptx native-fluid-pair.json

build/ptx-frontend/Release/ptx-frontend.exe `
  path/to/module_0013_01_05.ptx module-0013.json
build/ptx-frontend/Release/ptx-spirv-emitter.exe --entry `
  _Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj `
  64 0 module-0013.json fluid-calc-hash.spv
build/ptx-frontend/Release/ptx-spirv-emitter.exe --entry `
  _Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej `
  64 260 module-0013.json fluid-reorder.spv
spirv-val.exe --target-env vulkan1.2 fluid-calc-hash.spv
spirv-val.exe --target-env vulkan1.2 fluid-reorder.spv
build/vulkan-probe/Release/vulkan-memory-smoke.exe `
  translated-fluid-hash.json --kernel-fixture fluid-calc-hash.spv
build/vulkan-probe/Release/vulkan-memory-smoke.exe `
  translated-fluid-reorder.json --reorder-fixture fluid-reorder.spv
```

This proves two real captured PhysX kernels on a synthetic deterministic state.
It does not yet reproduce a captured in-game memory snapshot.

The emitted modules declare float32 `RoundingModeRTE` and
`SignedZeroInfNanPreserve`. All three probed target GPUs advertise those
properties. A conversion-only software fallback would not cover `fma.rn` and
the other correctly-rounded float operations; a complete fallback would
require software floating-point lowering. That added backend is not justified
while the properties remain evidence-backed target requirements. Preferred
8/16-bit storage paths and their packed-`u32` fallbacks are documented in
`vulkan-requirements.md`.

Cache keys include PTX hash, entry name, translator version, device
vendor/device/driver key, enabled float path, block size, and dynamic shared
size. Start with an in-memory cache; a disk cache is not required for the oracle
runner.

### Device selection and AMD portability

Enumerate every physical device. Apply the documented feature/limit gate without
vendor-name branches. Development runners accept an explicit device index and
print vendor/device IDs; the DLL defaults to the first passing non-CPU device
and later exposes a diagnostic override such as `PXVK_VULKAN_DEVICE`.
An AMD Radeon 8060S has passed the PE32 hardware gate. Its execution replay is
deferred until that machine is available again, so it does not block continued
oracle-runner work on Intel and RTX. AMD remains a required target for a later
cross-vendor execution claim, not a source of AMD-only lowering or semantic
shortcuts.

The AMD machine needs:

- 64-bit Windows with an x86 Vulkan loader/AMD ICD installed by the display
  driver;
- Vulkan 1.2 and the documented feature/limit gate;
- for building: CMake, Git/network access on first configure, and a Win32 C/C++
  compiler;
- for running a future prebuilt binary: only the x86 Vulkan loader/driver.

AMD game integration additionally needs the deferred minimal NVAPI shim. The
hardware preflight and standalone Vulkan oracle runner do not.

### CUDA virtual memory

Keep CUDA-visible addresses as nonzero `u32` virtual addresses into allocation
records. Reserve zero for null. For the captured public allocation, translate
an address to a byte offset in one 32 MiB Vulkan storage buffer after validating
that it belongs to the live range.

Use:

- one device-local 32 MiB buffer with storage and transfer usage;
- one persistent host-visible staging buffer;
- a small module-state storage buffer for globals and texture metadata;
- push constants for the exact packed parameter bytes, up to 100 bytes.

Implement `cuMemHostAlloc` with checked, page-aligned Windows virtual memory.
It need not be Vulkan-mapped or physically pinned for this workload; ZLUDA's
successful Mirror's Edge path already proves host mapping capability is not
load-bearing. HtoD copies stage from it. DtoH copies complete into it at
`cuCtxSynchronize`.

Represent heap storage as `u32` words. Lower byte and halfword accesses with
shift/mask operations; native 8/16-bit storage features remain optional.
`shaderInt64` handles the observed 64-bit integer operations.

### Textures and module state

A texture reference is typed metadata: bound heap offset, byte length, element
format/component count, and observed invariant sampler state. Generated shaders
perform signed bounds checks and four storage-buffer component reads; OOB
returns zero. No Vulkan image or sampler is created.

The focused native float-texture probe is complete. `tex.1d.v4.f32.s32`
preserves all tested in-range bits, including negative zero, subnormals,
infinities, and NaN payloads; negative and high signed coordinates return four
positive-zero components. The storage-buffer lowering reproduces those bits on
Intel and RTX without a Vulkan image or sampler.

### Commands and synchronization

Use one compute-capable queue and one logical default stream. Record copies and
dispatches in issue order. Insert conservative transfer/compute and
compute/compute barriers over the heap; optimize hazards only after sequence
fixtures pass.

`cuCtxSynchronize` closes/submits pending work, waits on a fence, copies pending
DtoH staging results into caller memory, and returns completion. Context teardown
waits before destroying Vulkan objects.

`bar.sync 0` becomes a workgroup execution and workgroup-memory acquire/release
`OpControlBarrier`. It never becomes a subgroup barrier.

### State and concurrency

Use typed opaque ID tables for contexts, modules, functions, globals, textures,
and allocations. Never expose Vulkan or host pointers as CUDA handles. Destroyed
IDs are not silently reused while stale handles can remain observable.

Protect host state with one coarse mutex initially. Release it while waiting on
Vulkan fences when state lifetime is already pinned. Thread IDs are still a
hardening probe; unknown call-thread behavior is not permission to build
thread-unsafe state.

## Runnable cross-vendor preflight

The script derives its CMake source from its own location, not the current
directory. It can therefore be invoked from any directory:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File D:\path\to\pxvk\tools\vulkan-probe\preflight.ps1 `
  -Toolchain MinGW `
  -RunOracle
```

CMake fetches pinned Volk, Vulkan-Headers, and SPIRV-Headers commits on first
configure. No Vulkan SDK path is required. Use `-SourceDir` only if the script
is deliberately moved away from `CMakeLists.txt`, `main.c`, and
`memory_smoke.cpp`. For Visual Studio 2022 with x86 tools:

```powershell
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass `
  -File D:\path\to\pxvk\tools\vulkan-probe\preflight.ps1 `
  -Toolchain MSVC `
  -RunOracle
```

By default, outputs are relative to the invocation directory:

```text
vulkan-preflight/capabilities.json
vulkan-preflight/preflight.json
vulkan-preflight/oracle.json
```

Override paths with `-OutputDir` and `-BuildDir`. `capabilities.json` is the raw
deterministic PE32 query. `preflight.json` contains the compiled Vulkan header
version, gate values, per-device pass/fail reasons, FTZ path, dependency source,
and capability hash. Exit status is nonzero when no enumerated device passes
the hardware gate or when a requested execution mode fails.

`-RunOracle` runs the full 64-thread semantic shader on every device. It
allocates the real 32 MiB device and staging buffers, performs
transfer/compute barriers, and checks exact f32, FMA, conversion, workgroup
barrier, integer texture, and input-guard output. `-RunMemorySmoke` remains as
the smaller allocation/integer-only diagnostic. Neither mode validates PTX,
NVAPI, or real PhysX kernels.

Observed preflight devices:

| Device | Vendor | Result | f32 path | Default subgroup |
|---|---|---|---|---:|
| Intel UHD Graphics | `0x8086` | hardware gate passed | native FTZ | 32 |
| RTX 3070 Laptop | `0x10de` | hardware gate passed | software FTZ required | 32 |
| AMD Radeon 8060S | `0x1002` | hardware gate passed | native FTZ | 64 |

The AMD submission is archived under
`tools/vulkan-probe/expected/amd-radeon-8060s-device1586-driver8389003/`.
Its raw capability SHA-256 is
`f35d24ad1404a2267e2dd5853466ef63960fb0de330b891d2de6c949c503fa7d`.
It proves the cross-vendor PE32 loader and hardware gate on AMD. It does not yet
prove SPIR-V execution, PTX semantics, NVAPI game startup, or PhysX output.

AMD's default subgroup size is 64 while this corpus has no subgroup operation.
This independently supports the decision not to make Vulkan subgroup size 32 a
hardware gate; CUDA-visible warp size remains an emulated API value.

## First independently testable implementation slice

Build one PE32 console target, provisionally `pxvk-vulkan-oracle`, with no CUDA
exports and no proprietary inputs.

### Slice contents

1. Reuse the preflight device enumeration/gate and add explicit device index.
2. Create one instance, device, compute queue, command pool, fence, device
   buffer, and mapped staging buffer.
3. Generate and validate one SPIR-V integer/global-memory shader.
4. Dispatch it, read exact results, and emit deterministic JSON.
5. Add the existing f32 cases, 64-thread barrier, and unsigned integer texture
   cases without changing expected field names.
6. Compare against
   `tools/cuda-oracle/expected/rtx3070-driver610.88.json`.
7. Run each passing physical device independently.

### Current status

All seven items now run through `preflight.ps1 -RunOracle`. The PE32 runner
allocates device-local 32 MiB buffers, dispatches one 64-thread SPIR-V module,
and emits the CUDA oracle's math, barrier, and texture field names per device.
Intel and RTX match every committed NVIDIA field bit-for-bit. Three runs
produced byte-identical JSON.

The readable GLSL source is `oracle.comp`. `rebuild-oracle.ps1` compiles it,
adds the required `NoContraction` decoration to the GLSL.std.450 `Fma`, assembles
`oracle.spv`, and validates it for Vulkan 1.2. Without that explicit decoration,
Intel split the FMA and returned zero while RTX happened to fuse it; the patched
module returns native `0xa8800000` on both.

The committed result is
`tools/vulkan-probe/expected/reference-workstation/oracle.json`, SHA-256
`42a8778edb77a7a74429e30fc2ec68c87c677ac0b9ca51a604539b07a8953968`.
The SPIR-V SHA-256 is
`2f79e7a87e40ac60ae316f51772b9eb40160ba1838c4a64ed3425badacbf1fab`.
The AMD replay remains a deferred external cross-vendor gate.

### Slice acceptance

- PE32 process confirmed in output;
- no vendor-name conditionals or non-core vendor extensions;
- integer/global-memory case exact on RTX and Intel;
- every committed CUDA oracle field exact on RTX and Intel;
- generated SPIR-V passes `spirv-val --target-env vulkan1.2`;
- three runs per selected local device produce byte-identical normalized JSON;
- command and report format is preserved for the deferred AMD run.

Do not begin the CUDA DLL, PTX corpus parser, NVAPI shim, or real-kernel capture
inside this slice. Those changes would obscure whether the Vulkan semantic core
works.

## Subsequent slices

1. **Finite PTX frontend:** parse synthetic cases, then all 17 local modules;
   emit the standalone oracle through normalized PTX operations.

The strict frontend now covers both project-owned PTX and all 17 local captured
modules. `tools/ptx-frontend` provides source-located tokenization; weak/visible
functions and entries; return/launch parameter layout; constant, shared,
external-shared, local, and texref metadata; entry launch directives; labels,
predication, structured operands; and finite opcode base/modifier
normalization.

Whole-corpus output matches the independent static analysis exactly:

| Metric | Parsed |
|---|---:|
| Modules | 17 |
| Entry declarations | 70 |
| Function declarations | 184 |
| Texture references | 33 |
| Constant globals | 7 |
| Base instruction families | 32 |
| Exact whole-corpus variants | 147 |

All 16 launched entries were cross-checked by module/name. Every packed
parameter offset/size/alignment, effective static shared size, and static local
size matches `mirrors-edge-ptx-corpus-analysis.json`. This includes the two
module-scope shared allocations that contribute to launched entry totals.

The project-owned schema-5 structured, value-typed oracle IR manifest SHA-256
is `611d93f54c373bfbf9f7915e0d1a75165d1e7af59ea6e1dcfdf834a3578c8c6b`.
The schema-3 redistributable structural corpus manifest is
`tools/ptx-frontend/expected/corpus-manifest.json`, SHA-256
`d49317aaeaf1f39f080d0cfdaa5c56b96dba8fa71ea06d76c672792b831e02ee`.
Raw proprietary PTX remains local.

The frontend now preserves labels, builds 4,174 deterministic basic blocks and
5,935 successor edges, and records register definitions/uses per instruction.
Every branch target and every register reference resolves across all 17
modules. The oracle path additionally emits direct SPIR-V. Whole-corpus
SSA/phi construction, full value propagation, and SPIR-V lowering remain open.

Build and run:

```powershell
cmake -S tools/ptx-frontend -B build/ptx-frontend -A Win32
cmake --build build/ptx-frontend --config Release
build/ptx-frontend/Release/ptx-frontend.exe --self-test
build/ptx-frontend/Release/ptx-frontend.exe `
  tools/cuda-oracle/probe.ptx oracle-manifest.json
build/ptx-frontend/Release/ptx-spirv-emitter.exe `
  oracle-manifest.json generated-oracle.spv
build/ptx-frontend/Release/ptx-frontend.exe --corpus `
  path/to/extracted/ptx corpus-manifest.json
```

The build fetches pinned nlohmann/json commit
`9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`; JSON serialization is external,
while PTX grammar and normalization remain project-owned.

2. **CUDA facade without game:** x86 export/ABI client, Dark lifecycle, handles,
   memory, module state, packed launch, and clean teardown.
3. **Native real-kernel fixtures:** capture and replay one fluid and one
   deformable entry only after standalone Vulkan semantics pass.
4. **Sequence execution:** one nine-kernel fluid step and one three-kernel
   deformable step.
5. **NVAPI and game integration:** only after the facade and real workload
   fixtures pass.

Each slice produces its own runnable artifact and cannot claim the next slice's
behavior.

## Blockers by acceptance point

| Blocker | Blocks | Does not block |
|---|---|---|
| Approximate math beyond fixed oracle inputs | accepting affected real kernels | exact committed reciprocal/sqrt/rsqrt cases |
| Deterministic deformable fixture | accepting corresponding real kernels | standalone runner, fluid pair, and frontend parsing |
| Queried-only NULL/stub probe | shrinking export table | conservative stubs and standalone work |
| Minimal non-NVIDIA NVAPI set | game startup on AMD/non-NVIDIA renderer | AMD Vulkan preflight and oracle runner |
| AMD memory/oracle execution replay | claiming generated SPIR-V execution on AMD | continuing and accepting the reference-system runner on Intel/RTX |
| Host call thread IDs | relaxing coarse locking | thread-safe implementation |
| PhysX-off server-room control | attributing particle launches specifically to glass | broken-glass integration smoke |

## Ranked risks and mitigation

| Risk | Evidence | Mitigation/gate |
|---|---|---|
| f32 FTZ cannot be requested on the reference RTX Vulkan path | Probe reports `shaderDenormFlushToZeroFloat32=false` | Explicit bit-level input/result normalization; standalone differential before real kernels |
| Approximate math differs across vendors | PTX uses approximate reciprocal/sqrt/rsqrt; native bits are known only on RTX CUDA | Per-operation exact/helper/ULP classification; no blanket epsilon |
| Finite parser misreads proprietary PTX | 17-module corpus is fixed but large | Typed strict parser, local manifest comparison, all-module gate, ZLUDA behavior reference |
| Default-stream visibility is under-synchronized | Vulkan queue order alone is not memory visibility | Conservative barriers and complete-step fixtures before optimization |
| Virtual u32 pointers diverge from host/Vulkan addresses | x86 PTX and API expose 32-bit values | Bounded virtual allocation records and offset-only shader lowering |
| AMD game is rejected before CUDA | Native startup performs NVAPI hardware selection | Separate narrow NVAPI slice; do not mix with Vulkan semantic debugging |
| Coarse lock harms performance | Thread behavior is not yet traced | Correct coarse locking first; measure and narrow only after correctness |
| Direct SPIR-V emitter produces invalid modules | No compiler frontend hides structural mistakes | Deterministic emitter plus `spirv-val` on every generated test module |

## Architecture gate conclusion

The reference-system Vulkan oracle and whole-corpus grammar, metadata, CFG,
register-dataflow, and finite result/use-type gates are complete. The next gate
is direct SPIR-V emission of the project-owned oracle from this IR.
Proprietary entries remain parse-only until that emitted oracle matches the
existing Vulkan fixture.

The gate is not open for a claim that Mirror's Edge runs on Vulkan, AMD
generated-SPIR-V execution, AMD game integration, or translated PhysX
correctness. Those claims remain behind the explicit frontend, AMD replay,
native-fixture, CUDA-facade, and NVAPI gates above.
