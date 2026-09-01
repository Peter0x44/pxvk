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
`tools/vulkan-probe/preflight.ps1`. It enumerates every Vulkan physical device
and applies one invariant gate for classic 32-bit GPU PhysX 2.8.x. Launch and
resource limits remain inputs derived from the selected PTX entry and actual
launch, not profiles copied from individual games or scenes. The script emits
hashed JSON and does not claim shader or PhysX correctness.

## Evidence synthesized

| Research result | Load-bearing conclusion |
|---|---|
| `mirrors-edge-native-trace.md` | Three routes exercise fluid, deformable, and visibly broken-glass workloads through one x86 context and default stream |
| `mirrors-edge-ptx-corpus.md` | Required frontend is finite: PTX 4.1, 17 modules, 16 launched entries, 120 launched variants, 32-bit addresses |
| `cuda-driver-contract.md` | Host contract is 35 active exports, six Dark tables, stable handles, caller-sized pools (32 MiB in Mirror's Edge; 128 MiB in official samples), packed launches, and ordered default-stream work |
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

The first vertical slice now emits a 3,086-word SPIR-V 1.5 module directly from
the schema-5 structured IR for `tools/cuda-oracle/probe.ptx`. The module uses
one 64-thread Vulkan entry point to invoke the original one-thread math and
constant routines, 64-thread barrier, 64-thread invocation-private local-memory,
and 12-thread integer and float texture routines against the oracle binding
profile. SDK `spirv-val --target-env vulkan1.2` accepts the binary, and Intel UHD
plus RTX 3070 executions match every native CUDA math, barrier, local-memory,
module-constant, and texture field.

The second slice lowers the captured 89-instruction
`KernelFluidCalcHashPacked` entry from `module_0013_01_05.ptx`. A conservative
structured state machine stores live PTX values in reusable function slots and
dispatches 14 PTX basic blocks through one SPIR-V loop/switch. Binding 0 is the
32-bit CUDA heap and binding 1 is the packed 52-byte launch parameter buffer.
The 2,807-word module is deterministic and passes Vulkan 1.2 validation.

The paired third slice lowers the captured 116-instruction
`KernelReorderPosVelAndFindCellStartPacked` entry. It adds shared `u32`
storage, predicate conjunction, integer equality/inequality, a 260-byte
dynamic-shared contract, workgroup barriers, and vector `f32` stores. Its
pre-barrier diamond is emitted as structured selections that reconverge before
one dynamically uniform `OpControlBarrier`; the remaining 12 blocks use the
state machine. The deterministic 3,126-word module passes Vulkan 1.2
validation.

Schema 5 retains every routine `.local`/`.shared` symbol with space, type,
aligned byte offset, size, alignment, and external status. Kernel emission now
uses those symbols rather than the former `sharedHash` special case, allocates
`.local` storage per invocation, and takes the captured dynamic-shared byte
count explicitly when sizing each Workgroup arena. The standalone local probe
stores and reloads the aligned symbol at byte offset 16; native RTX and
translated Intel/RTX return `0x5a000000 + tid` for all 64 invocations.

Local, module-constant, and texture storage are now all present in the
trace-ordered `KernelFluidDensity` path. Module `.const` declarations use a
read-only storage buffer at binding 2, separate from the binding-0 CUDA heap
and binding-1 launch parameters. Symbols retain schema-5 aligned byte offsets.
The standalone constant probe copies eight native words through
`cuModuleGetGlobal_v2`, then uses the density kernel's symbol-base plus
computed-offset address shape; native RTX and translated Intel/RTX preserve
`0x1234abcd` and NaN payload `0x7fc12345` exactly. The captured trace
independently shows a 1,024-byte device-to-device update into density's
10,240-byte `parameters_cmem`.

The selected-entry path independently emits the same `constant_probe` through
`KernelEntryLowerer` as a 446-word module with bindings 0/1/2. Vulkan 1.2
validation passes, and both local GPUs match the native payloads. The report is
stored at `vulkan.constant_memory` in
`tools/reference/expected/reference-workstation.json`; its original standalone
SHA-256 was
`b00dfe683ae265e8569be15aa05a2dabfd0ca4633d2b8c500f34dee5753620da`.

Selected legacy texture fetches add read-only storage-buffer binding 3. Each
module texture-reference index maps to three runtime `u32` words: base
binding-0 heap-word offset, element count, and component count. The shader
performs signed coordinate bounds checks before loading, returns positive zero
out of bounds, and bitcasts `f32` words without arithmetic conversion. A
729-word selected `u32` fixture covers the captured one-component layout; a
773-word selected `f32` fixture covers the captured four-component layout and
preserves negative zero, subnormals, infinities, and NaN payloads. Coordinates
-2 through 9 match on Intel and RTX. The reports are stored at
`vulkan.texture_u32` and `vulkan.texture_f32` in the reference bundle. Their
original standalone hashes were:

- integer: `f2f2af01930991fe61a4db237a7e0978953b26ab283331627b7c6bc90f349eb3`;
- float: `66fc00584b8371d0c7028bfcbc981c019ce4401b90ae1af1a55229eb4cb3f11b`.

`KernelFluidDensity` declares `%r<497>`, `%f<30>`, and `.s64 %rd<35>`,
but its final instructions reference 357 distinct `%rN` names, 23 distinct
`%fN` names, and no `%rdN` names. PTX bank declarations bound virtual-register
namespaces; they do not prove liveness or allocate hardware registers. The
exact old CUDA compiler pass ordering is unavailable, but the instruction
stream contains no hidden `%rd` dependency. Kernel lowering therefore discards
an unsupported bank only when it has zero definitions and uses; the CFG fixture
locks that rule, and a referenced unsupported type remains an error. No
`Int64` capability or 64-bit arithmetic was added for density.

Kernel emission computes standard CFG live-in/live-out sets, builds an
interference graph, and reuses one SPIR-V function variable only for
non-interfering PTX registers of the same scalar type and logical bit width.
This preserves the mutable PTX register namespace across branches and loops
while avoiding thousands of dead or non-overlapping function variables. The
collision entry exposed the practical requirement: literal bank-sized
allocation produced a valid module that the Intel driver could not create a
pipeline for; liveness allocation passes exact Intel/RTX differentials for all
nine lowered captured entries.

With the two captured texture variants plus the finite integer/FTZ-float tail,
all 411 density instructions lower. The deterministic 17,768-word SPIR-V
module, with bindings 0/1/2/3 and local size 192, passes Vulkan 1.2 validation;
SHA-256 is
`ff43010d9bc54639b4707a1eddffa77332aa63442a2e14cba887008244671abf`.

The next trace-ordered entry, `KernelFluidForce`, contains 499 instructions in
68 basic blocks with 100 CFG edges. It uses a 100-byte packed parameter block,
108 bytes of invocation-local storage, the 10,240-byte module constant, and six
linear texture references: sorted positions, velocities, densities, hash keys,
cell starts, and cell ends. The captured launch uses local size 128 and no
dynamic shared memory. Finite lowering adds ordered `lt`, unordered `le`/`ne`,
FTZ `max`, approximate reciprocal/square-root, and full float division. The
committed `force_math_probe.ptx` keeps those selected-entry paths in the
redistributable CTest suite. The deterministic 27,835-word force module passes
Vulkan 1.2 validation; SHA-256 is
`d9816251075a0b4cbc153bcd87fc6377a8d7ee1c79207fa2deb1fb4fcd07b235`.

The next fluid-spine entry, `PxgCudaFluidCollisionConstraintsBatch`, contains
725 instructions in 71 basic blocks with 107 CFG edges. It uses a 20-byte packed
parameter block, 1,840 bytes of static shared memory, local size 128, and no
module constants or textures. Finite lowering adds two- and four-component
shared float loads plus four-component shared integer loads. The deterministic
59,442-word module passes Vulkan 1.2 validation; SHA-256 is
`90c1fbec399219664e7a6b677b3231275af4c693297b5b5d2fa45ad151da270c`.

The focused one-particle fixture selects the `0x0000ffff` no-contact constraint
record. With time step `0.5f`, position `[1,2,3]`, and motion `[4,5,6]`, native
CUDA and PTX-derived Intel/RTX Vulkan produce position `[3,4.5,6]`, velocity
`[4,5,6]`, unit weight, twelve cleared output ranges, and intact guards. Three
native reports are byte-identical. Native and Vulkan report SHA-256 values are
`28c23397e099c4dc2db893253fbc292c733e0f414928d9812c5b29fc94d6e7eb`
and
`7de7425bff92b4cfb49266564ffac9ef611170a2bf608e2d815276ac17903cbb`.

The next trace-ordered entry,
`PxgCudaFluidCollisionDynamicResponseBatch`, contains 263 instructions in 36
basic blocks with 51 CFG edges. It uses a 20-byte packed parameter block, 304
bytes of static shared memory, local size 128, and no module constants or
textures. Finite lowering adds `cvt.rn.f32.u32` and `not.pred`; the
redistributable float-math and collision-type probes retain both selected-entry
paths. The deterministic 14,671-word module passes Vulkan 1.2 validation;
SHA-256 is
`1367057fa1d0379b45e1b7ed743440bafc8cbd45b065a3de0849946c0fc8eff9`.

The focused active-constraint fixture consumes the constraints output layout:
unit X normal, accumulated position `[7,8,9]`, count one, zero body velocity,
particle velocity `[-2,3,0]`, and impulse scale `0.5f`. Native CUDA and
PTX-derived Intel/RTX Vulkan clear the consumed normal and accumulators and
produce averaged position `[7,8,9]`, velocity `[1,3,0]`, response `[-3,0,0]`,
unit weight, copied destination ID, cleared flags, and intact guards. Three
native reports are byte-identical. Native and Vulkan report SHA-256 values are
`aadb855952a6fc76c9ac148841aa50e3366511eb9d08f8b7ac15269ed6e113f0`
and
`86fc752ef108674ae7adb086dfb570e4e7defd7098ff214626d79090231b6a87`.

`PxgCudaFluidCollisionTrianglesBatch` is the next trace-ordered fluid collision
entry. It contains 1,022 instructions in 148 basic blocks with 225 CFG edges, a
20-byte packed parameter block, 4,912 bytes of static shared memory, and local
size 64. Its triangle candidate mask requires native SPIR-V `Int64` add, shift,
bitwise, comparison, and select operations. `clz.b64` splits the value into
32-bit halves because Vulkan validation limits `GLSL.std.450 FindUMsb` to
32-bit components. The `Int64` capability is emitted only for a selected entry
that actually contains a 64-bit instruction. The deterministic 68,441-word
module passes Vulkan 1.2 validation; SHA-256 is
`e4e4cc84ee9f0a50a56b65610f4965e235e6d27aac504948dbfdec6e0a60e291`.

The focused one-particle fixture selects a mesh work unit with zero triangles.
It covers the captured 64-thread launch, three shared-memory setup barriers,
parameter and particle loads, empty-mesh branch, and state writeback. Native
CUDA and PTX-derived Intel/RTX Vulkan preserve the position, old position, and
two state words bit-for-bit with every guarded range intact. The native and
Vulkan report SHA-256 values are
`a9c53cf29be39d08e4eecb6096280be43672cfbc5a218ca906b1e9ac56ad82e9`
and
`fe47f95a513774bc49f316283b15b6f1e6077cd7e4a19944c2bd66ff15f5f36d`.
This fixture does not yet cover the non-empty triangle-contact path.

`PxgCudaFluidCollisionStaticResponseBatch` is the eleventh captured entry to
compile and execute. It contains 330 instructions in 47 basic blocks with 69
CFG edges, a 20-byte packed parameter block, 304 bytes of static shared memory,
and local size 128. Finite lowering adds two-component shared `u32` loads; the
redistributable collision-type probe retains that form plus the entry's 16-bit
mask and comparison forms. The deterministic 22,632-word module passes Vulkan
1.2 validation; SHA-256 is
`5dadeecbde444d20956e9d7a6ab8fcb3f8ab650890033f66fd1a2d00fbe7f37e`.

The focused active-contact fixture uses unit X normal, accumulated position
`[7,8,9]`, count one, zero body velocity, particle velocity `[-2,3,0]`,
collision radius `1.0f`, and restitution `0.5f`. Native CUDA and PTX-derived
Intel/RTX Vulkan normalize the contact, average and compact position `[7,8,9]`,
resolve and compact velocity `[1,3,0]`, emit unit X response and the
particle/source pair, and preserve every guard. Native and Vulkan report
SHA-256 values are
`1009b3ab0d85b6133d5053826852e3a213cb0c57111676689465b7e955e3f709`
and
`1381934743e92a95ecf61a98274df4d314f2481609b33679fd2e4834528ff937`.
Alternate contact flags, tangential damping, position-distance clamping, and
plane projection remain untested.

`PxgCudaFluidCollisionBoxShapesBatch` is the twelfth captured entry to compile
and execute. It contains 1,288 instructions in 133 basic blocks with 204 CFG
edges, a 20-byte packed parameter block, 6,752 bytes of static shared memory,
and local size 64. Its finite instruction surface was already covered by the
translated collision entries. The deterministic 61,420-word module passes
Vulkan 1.2 validation; SHA-256 is
`ee5f3e06f4e810cdf4ec54aaf7333cf29332927bec5ab226382414513a7713e9`.

The focused fixture uses the trace-observed two-work-unit grid with one particle
and zero boxes. It crosses five shared-memory barriers, copies both 80-byte
shape-start records and both parameter blocks, loads position and old position,
and takes the empty-shape exit. Native CUDA and PTX-derived Intel/RTX Vulkan
preserve both vectors and every guard bit-for-bit. Native and Vulkan report
SHA-256 values are
`2ff6c3f8215d59d39cfc0ef9b6041615aad883b755be724cf08a82faca93d121`
and
`fddd4e5d911c6d1d499eb03cccc8ef42cd003105c3d92b1c12b06fdfabe0e8b6`.
No box-contact geometry is covered.

`PxgCudaFluidCollisionCapsuleShapesBatch` is the thirteenth captured entry to
compile and execute. It contains 1,312 instructions in 159 basic blocks with
248 CFG edges, a 20-byte packed parameter block, 2,816 bytes of static shared
memory, and local size 64. Its finite instruction surface was already covered
by the translated collision entries. The deterministic 103,800-word module
passes Vulkan 1.2 validation; SHA-256 is
`fa57a13142c49fcbea4279acd9860aab2aa275a82dca702b658d30133594b3b0`.

The focused fixture uses the minimum trace-observed two-work-unit grid with one
particle and zero capsules. It crosses five shared-memory barriers, copies both
80-byte shape-start records and both parameter blocks, loads position and old
position, and takes the empty-shape exit. Native CUDA and PTX-derived Intel/RTX
Vulkan preserve both vectors and every guard bit-for-bit. Native and Vulkan
report SHA-256 values are
`e52b66988554fecd599f8ae48daf1726ee97a6c36228ea51f27890e98478d4a4`
and
`1642ce984d610bc860bdba6d3c1167d596fce99b10ba60ac61c25c773a5b1a2a`.
No capsule-contact geometry is covered.

`PxgCudaFluidCollisionConvexShapesBatch` is the fourteenth captured entry to
compile and execute. It contains 1,433 instructions in 149 basic blocks with
224 CFG edges, a 20-byte packed parameter block, 4,225 bytes of static shared
memory, and local size 64. Its finite instruction surface was already covered
by the translated collision entries. The deterministic 75,510-word module
passes Vulkan 1.2 validation; SHA-256 is
`7f10bd7049e51f9a4448ebb265064c9e26f54666e39ea6bde7a7f46cfac86cd8`.

The focused fixture uses the minimum trace-observed five-workgroup dispatch,
two work-unit records, one particle, and zero convexes. The surplus workgroups
take the out-of-range exit; the first two copy both work-unit and 80-byte
shape-start records, copy the active parameter block, load position and old
position, cross three shared-memory barriers, and take the empty-shape exit.
Native CUDA and PTX-derived Intel/RTX Vulkan preserve both vectors and every
guard bit-for-bit. Native and Vulkan report SHA-256 values are
`1e9ba7104210ffcd33d7d0dd60a4b8538447ca8544a8fea916370c346b186456`
and
`aa5a42405744e4dddd5e64f01f74eb22118caaa7200a1a56c6249cbfb67634b3`.
No convex-contact geometry is covered.

The first deformable-family entry, `PxgCudaDeformableUpdate`, contains the
finite unsigned `setp.lt.u32` comparison and bitwise `selp.b32` select variants.
Both lower directly to SPIR-V unsigned comparison and `OpSelect`; the committed
CFG probe retains those paths. The extracted entry emits a deterministic
9,792-word module at local size 64 with no dynamic shared memory. Vulkan 1.2
validation passes; SHA-256 is
`37f7ef69aa672681371b140a18e11c1d1a0e34abcc055342b52c989f2d437c14`.

A one-instance native fixture uses exact synthetic acceleration, time-step,
position, motion, and state-flag bits plus guards around every live allocation.
Three RTX captures are byte-identical. The integrated motion is
`[0x40440000, 0x408c0000, 0x40c80000, 0x40e00000]`, the position is unchanged,
and state flags become zero. PTX-derived Vulkan produces the same bits with
intact guards on Intel UHD and RTX 3070. The native report SHA-256 is
`30dfd7ef0bf3c7f17b1801e13a86a02e4f0a2cde21b36f5dc65e01816664774e`;
the Vulkan report SHA-256 is
`3aebb159586704b8767150940e9a55bbfee33540cce6cb358fd178658683274f`.

The solver deformable entry, `KernelDeformableUberSolver<512>`, has 1,191
instructions and five static `bar.sync` sites. Its finite surface adds
16-bit register banks and bitwise operations, signed 16-bit comparisons and
wide multiplication, byte parameter loads, packed `v4.u16` global loads,
vector shared stores, predicate OR, float absolute/negate, and reciprocal
square root. Narrow registers remain 32-bit SPIR-V function values with
explicit low-bit masking and sign extension, avoiding an unnecessary Vulkan
16-bit-storage requirement.

Barrier instructions terminate frontend CFG blocks. The kernel emitter runs
each invocation's CFG until its next barrier, reconverges the workgroup at one
structured `OpControlBarrier`, then resumes from the saved per-invocation
program counter. This preserves divergent work between barriers without
dropping the solver's repeated barrier sites. The local-size-64 entry includes
the captured 7,072-byte dynamic shared allocation and emits 80,354 SPIR-V
words. Vulkan 1.2 validation passes; SHA-256 is
`7a0a3341fe5f862f82379e09d4c008a78ffaf9bb9bed9f0a071bdbd6f783684e`.

The focused native fixture runs one synthetic particle with zero constraint
iterations. It still crosses the solver barriers, converts the initial
position against a reference position to exact delta bits
`[0xc1100000, 0xc1100000, 0xc1100000, 0x00000000]`, and writes the exact
64-byte solver record with all guards intact. PTX-derived Intel UHD and RTX
3070 Vulkan executions match the native result bit-for-bit. The native report
SHA-256 is
`83d4b38dd4112716e63311a06b7d040e9275c44654fd1bae6dbdefb33009c040`;
the Vulkan report SHA-256 is
`9b551fbc7296684b48645e8058c3a49d71a87f96733ca0690bed9e5484eb1d3c`.

The collision deformable entry, `KernelDeformableUberCollision`, contains
3,662 instructions, five `bar.sync` sites, 4,097 bytes of static shared memory,
and 17 legacy texture references. Its finite surface adds packed and scalar
`u8`/`u16` global, constant, and shared accesses; `v4.f32` loads; 16-bit
select/equality forms; ordered and unordered float comparisons; and FTZ float
minimum. Subword loads explicitly extract addressed bits; stores to the
observed word-aligned shared byte use workgroup atomic exchange, preserving the
benign CUDA same-value multi-writer behavior without a Vulkan data race. A
committed synthetic probe retains these paths.

The local-size-64 collision entry emits 175,407 SPIR-V words and passes Vulkan
1.2 validation; SHA-256 is
`bb32ba2c35a1b2d86b58bc921d9936eb1def25bbe0723fa6bdb7733aa249fd57`.
The focused native fixture runs one particle with no shapes, crosses all
barriers, copies its position/motion vectors, and clears collision state. The
exact collision scalar words are
`[0xbf800000, 0x00000000, 0x0000ffff, 0x00000000]`; all 17 live ranges remain
guarded. Native RTX and PTX-derived Intel/RTX results match bit-for-bit. Native
and Vulkan report SHA-256 values are
`0413b0348b49fe1b69376e6ed049f7011c2781366e3311ade51b5daaa0287369`
and
`cc9980da30b4f6f7f4208d42d404f517a4db3beac855ac82ca1ff38aea770ac2`.

The ordered deformable fixture loads all three proprietary modules at once and
uses one guarded 32-bit heap. Native CUDA queues update, collision, and solver
on the default stream before one final synchronization. Collision consumes
update's position/motion; solver consumes and updates collision's position
output. Vulkan records the same order in one command buffer with explicit
compute-write to compute-read/write barriers. Intel UHD and RTX 3070 match the
native updated motion, collision scalars, solver delta, and 64-byte record
bit-for-bit; all 25 heap ranges remain guarded. Native and Vulkan report SHA-256
values are
`4c71f0c3f14d0c9456c38a15fdf04a04271568cc77acdcc211a78e2d560ba2a8`
and
`d215b03ffd1938263653ecea5cef4e6653e4b621394fe8cf12c5ab64ea4ccbd3`.
The state deliberately has zero shapes and zero constraint iterations. This
proves ordered dispatch and inter-kernel dataflow, not captured collision
geometry or iterative constraint response.

`tools/cuda-physx-fixture` now runs a four-kernel one-particle pipeline against
local proprietary modules 0013, 0010, and 0016. Calc-hash and reorder produce
density's inputs. Density uses the trace-observed block size 192, 1,024-byte
module-constant update, 64-element position/hash texture bindings, and
524,288-element cell-start/end bindings. Force consumes that output with its
trace-observed block size 128, 1,024-byte constant update, three 64-element
float texture bindings, the 64-element hash, and the same cell tables.

The fixture constants are synthetic exact bits placed at captured offsets.
Density uses radius term and distance scale `1.0f` plus normalization `2.0f`.
Force uses time step `1.0f`, zero damping and acceleration, and maximum velocity
`100.0f`. This makes the one-particle self-neighbor contribute no pair force,
so the exact output is the input velocity. This is a trace-shaped
differential, not a captured in-game memory snapshot.

Native RTX produces:

- hash `0x00003081`, original index `0`;
- reordered position bits `[0x3f800000, 0x40000000, 0x40400000, 0x00000000]`;
- reordered velocity bits `[0x40800000, 0x40a00000, 0x40c00000, 0x40e00000]`;
- cell interval `[0, 1]` at hash `0x00003081`;
- direct and remapped density bits `0x40000000` (`2.0f`);
- force bits `[0x40800000, 0x40a00000, 0x40c00000, 0x00000000]`.

The Vulkan density and force fixtures start from the corresponding deterministic
post-reorder/post-density state and bind separate constant and texture-metadata
descriptor ranges. Intel UHD and RTX 3070 match every native output bit. These
reports are bundled under `cuda.physx_fluid_pipeline` and the four
`vulkan.fluid_*` keys. Their original standalone evidence hashes are:

- native four-kernel CUDA pipeline:
  `b27ef7d9030bcdff2a77d84e4df7af2a95766a1fe920bb74d4696e6cd3b8cb2c`;
- translated force Vulkan:
  `ce9dc03f572be9e5901c3cf6cc2f56caad7af3e1a2f5d2c5717b4b20c1f262b1`;
- translated density Vulkan:
  `67ca1b0f75a279a1a9ad00be197c9d5252494871739f3f8acd3fa05a3d9fa079`;
- translated calc-hash Vulkan:
  `ffec75c749d458b61c71712375300ab21aee723755ffbcb403273c038b953b1c`;
- translated reorder Vulkan:
  `a048d9dc1467761ae2c39b5d53e3b3c38351cfc3428e844b898b94965a5e847d`.

Reproduce the complete portable path from the repository root:

```powershell
.\run-tests.ps1 -CapturedPtxDir path\to\extracted\ptx
```

The directory must contain the extracted modules named in
`kernel-status.md`. The script parses, emits, validates when `spirv-val.exe` is
available, and executes all fourteen translated entries plus the ordered
deformable step. Proprietary PTX and generated derivatives remain local.

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
records. Reserve zero for null. Translate an address to a byte offset in its
owning caller-sized Vulkan storage allocation after validating the live range.

Use:

- caller-sized device-local buffers with storage and transfer usage;
- host-visible staging capacity sized or grown for the requested transfers;
- read-only storage buffers for module constants, the compact texture table,
  and oversized per-workgroup shared scratch when required;
- storage-buffer binding 1 for the exact packed parameter bytes.

Implement `cuMemHostAlloc` with checked, page-aligned Windows virtual memory.
It need not be Vulkan-mapped or physically pinned for this workload; ZLUDA's
successful Mirror's Edge path already proves host mapping capability is not
load-bearing. HtoD copies stage from it. DtoH copies complete into it at
`cuCtxSynchronize`.

Represent heap storage as `u32` words. Lower byte and halfword accesses with
shift/mask operations; native 8/16-bit storage features remain optional.
`shaderInt64` handles the observed 64-bit integer operations.

### Textures and module state

A texture reference is a compact runtime record: bound heap-word offset,
element count, and component count. Format comes from the finite PTX fetch
variant; the captured point-filter/clamp state requires no Vulkan image or
sampler. Generated shaders perform signed bounds checks, use safe indices for
storage-buffer reads, and explicitly select zero for OOB components.

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
deterministic PE32 query. Schema-4 `preflight.json` contains the compiled Vulkan
header version, version-wide invariant gate, raw runtime limits, FTZ path,
dependency source, and capability hash. Exit status is nonzero when no
enumerated device passes the invariant gate or when a requested execution mode
fails. It never turns captured block/grid sizes into startup thresholds.

`-RunOracle` runs the full 64-thread semantic shader on every device. It
allocates the real 32 MiB device and staging buffers, performs
transfer/compute barriers, and checks exact f32, FMA, conversion, workgroup
barrier, integer texture, and input-guard output. `-RunMemorySmoke` remains as
the smaller allocation/integer-only diagnostic. Neither mode validates PTX,
NVAPI, or real PhysX kernels.

Observed preflight devices:

| Device | Vendor | Result | f32 path | Subgroup size 32 |
|---|---|---|---|---:|
| Intel UHD Graphics | `0x8086` | invariant gate passed | native FTZ | native |
| RTX 3070 Laptop | `0x10de` | invariant gate passed | software FTZ required | native |
| AMD Radeon 8060S | `0x1002` | invariant gate passed | native FTZ | selectable |

The AMD submission is archived under
`tools/vulkan-probe/expected/amd-radeon-8060s-device1586-driver8389003/`.
Its raw capability SHA-256 is
`f35d24ad1404a2267e2dd5853466ef63960fb0de330b891d2de6c949c503fa7d`.
It proves the cross-vendor PE32 loader and hardware gate on AMD. It does not yet
prove SPIR-V execution, PTX semantics, NVAPI game startup, or PhysX output.

AMD's default subgroup size is 64, but subgroup-size control can require 32 for
compute. It therefore passes the version-wide warp invariant without turning
one game's block dimensions into hardware policy.

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
6. Compare against `cuda.semantic_oracle` in
   `tools/reference/expected/reference-workstation.json`.
7. Run each passing physical device independently.

### Current status

All seven items now run through `preflight.ps1 -RunOracle`. The PE32 runner
allocates device-local 32 MiB buffers, dispatches one 64-thread SPIR-V module,
and emits the CUDA oracle's math, barrier, invocation-private local-memory, and
integer/float texture field names per device. Intel and RTX match every covered
NVIDIA field bit-for-bit. Three runs produced byte-identical JSON.

The readable GLSL source is `oracle.comp`. `rebuild-oracle.ps1` compiles it,
adds the required `NoContraction` decoration to the GLSL.std.450 `Fma`, assembles
`oracle.spv`, and validates it for Vulkan 1.2. Without that explicit decoration,
Intel split the FMA and returned zero while RTX happened to fuse it; the patched
module returns native `0xa8800000` on both.

The committed report is `vulkan.semantic_oracle_static` in the reference
bundle. Its original standalone SHA-256 was
`782856f86951d704e40feceb0394321a444f7877b6623f6601700f8fcb271073`.
The SPIR-V SHA-256 is
`37782524d39e89aea01ce1b7c51462259fc14637ee7cd28bab7396186b37bb2a`.
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
is `c61f5cf9cdb96e0ec97547d2ee010e76d97fcf953dc7a77683eb037312fa02a2`.
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
4. **Sequence execution:** the ordered synthetic three-kernel deformable step
   passes; the nine-kernel fluid spine and captured-state replays remain.
5. **NVAPI and game integration:** only after the facade and real workload
   fixtures pass.

Each slice produces its own runnable artifact and cannot claim the next slice's
behavior.

## Blockers by acceptance point

| Blocker | Blocks | Does not block |
|---|---|---|
| Approximate math beyond fixed oracle inputs | accepting affected real kernels | exact committed reciprocal/sqrt/rsqrt cases |
| Captured-state deformable replay | claiming in-game deformable correctness | exact focused and ordered synthetic differentials |
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
