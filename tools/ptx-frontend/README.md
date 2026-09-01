# pxvk PTX frontend

`ptx-frontend` is the strict, finite PTX 4.1 parser and normalized-IR inspection
tool for the Mirror's Edge compatibility target. It is not a general PTX
compiler.

## Scope

The frontend currently accepts:

- the project-owned `tools/cuda-oracle/probe.ptx` module;
- all 17 locally extracted Mirror's Edge PTX modules;
- all 70 entry declarations and 184 function declarations;
- all 32 base instruction families and 147 exact variants in that corpus.

Raw game PTX is proprietary and must remain local. The committed corpus
manifest contains structural metadata only.

## Pipeline

```text
PTX bytes
  -> source-located tokens
  -> strict module/routine AST
  -> parameter and storage layouts
  -> labels and normalized instructions
  -> structured and typed operands
  -> basic-block CFG
  -> per-instruction register definitions and uses
  -> deterministic JSON manifests
  -> direct SPIR-V binary emission for the project oracle slice
```

`ptx-spirv-emitter` consumes schema-5 structured operands and routine storage
symbols. It does not parse the retained diagnostic operand strings.

## Build

CMake fetches pinned nlohmann/json commit
`9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03` and SPIRV-Headers commit
`29981f65241605e08b0ede4cfeb999fe3b723c6a`. PTX grammar, validation, CFG
construction, dataflow, semantic lowering, and binary emission are
project-owned.

MSVC Win32:

```powershell
cmake -S tools/ptx-frontend -B build/ptx-frontend -A Win32
cmake --build build/ptx-frontend --config Release
```

MinGW i686:

```powershell
cmake -S tools/ptx-frontend -B build/ptx-frontend-mingw `
  -G "MinGW Makefiles" `
  -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ `
  -DCMAKE_MAKE_PROGRAM=mingw32-make `
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/ptx-frontend-mingw --parallel
```

## Run

Self-tests:

```powershell
build/ptx-frontend/Release/ptx-frontend.exe --self-test
ctest --test-dir build/ptx-frontend -C Release --output-on-failure
```

One redistributable module:

```powershell
build/ptx-frontend/Release/ptx-frontend.exe `
  tools/cuda-oracle/probe.ptx `
  oracle-manifest.json
```

Emit the project oracle with explicit float32 RTE control:

```powershell
build/ptx-frontend/Release/ptx-spirv-emitter.exe `
  oracle-manifest.json `
  generated-oracle.spv
```

Emit one finite PTX entry with a structured control-flow state machine:

```powershell
build/ptx-frontend/Release/ptx-spirv-emitter.exe --entry `
  <entry-name> <local-size-x> <dynamic-shared-bytes> `
  module-manifest.json `
  generated-kernel.spv
```
When a launch exceeds the target device's native workgroup-memory limit, emit
the same entry with storage-backed shared memory:

```powershell
build/ptx-frontend/Release/ptx-spirv-emitter.exe --entry-shared-storage `
  <entry-name> <local-size-x> <dynamic-shared-bytes> `
  module-manifest.json `
  generated-kernel.spv
```

The runtime, not the frontend, selects this fallback from the actual launch
size and Vulkan device limit. It binds a scratch storage buffer at binding 4.
For the current X-only launch contract, each workgroup receives a disjoint
`ceil(required-shared-bytes / 4)`-word region and the caller allocates
`align4(required-shared-bytes) * grid_x` bytes. Shared atomics and barriers use
workgroup scope with storage-buffer memory semantics. The
`shared_storage_grid_probe` fixture dispatches two X workgroups and verifies
both guarded outputs and distinct scratch-slice heads on Intel and RTX.
Non-unit Y/Z dispatch remains outside the evidenced contract.

Kernel mode binds the 32-bit CUDA heap at descriptor binding 0, packed
launch-parameter words at binding 1, and module `.const` storage at read-only
storage-buffer binding 2 when the selected module declares constants. A
selected entry containing legacy texture fetches additionally reads binding 3.
Each manifest texture-reference index owns three `u32` metadata words: base
heap-word offset, element count, and component count. The current finite
lowering covers the captured one-dimensional, signed-coordinate, 32-bit
one-/four-component references. It checks coordinates explicitly, preserves
in-range `f32` bits, and returns positive-zero components out of bounds.

PTX byte addresses remain `u32` byte offsets within their memory space. The
dynamic-shared byte count is explicit pipeline input. Native mode combines it
with module/routine static shared layouts to size the Workgroup arena; storage
fallback mode uses the same byte layout within binding 4. Named `.shared`
symbols resolve to aligned byte offsets in either backing. Named `.local`
symbols resolve to aligned byte offsets in a per-invocation Function array.
Named `.const` symbols resolve to aligned byte offsets in binding 2.

Kernel lowering discards an unsupported register bank only when no instruction
defines or uses any register in that bank. A PTX bank declaration bounds a
virtual-register namespace; it does not assert that every name is live or
allocate hardware registers. `KernelFluidDensity` declares `%r<497>`,
`%f<30>`, and `%rd<35>`, while its final instructions reference 357 distinct
`%rN` names, 23 distinct `%fN` names, and no `%rdN` names. The unused `.s64`
bank is therefore compiler-output residue, not evidence of a missing operation.
A referenced unsupported type remains a deterministic error; no `Int64`
capability was added for this entry.

The current state-machine lowering computes CFG liveness and reuses function
storage only for non-interfering PTX registers with the same scalar type and
logical width. It executes one PTX basic block per structured loop iteration.
Entries containing barriers run each invocation until its next barrier,
reconverge at one structured workgroup barrier, and resume from the saved
program counter. This is intentionally conservative; SSA and general structured
CFG reconstruction remain later optimizations.

`KernelFluidForce` extends selected-entry lowering with the finite
`setp.lt.ftz.f32`, `setp.leu.ftz.f32`, `setp.neu.ftz.f32`,
`max.ftz.f32`, `rcp.approx.ftz.f32`, `sqrt.approx.ftz.f32`, and
`div.full.ftz.f32` forms. Its 499 PTX instructions lower to a deterministic
27,835-word module with local size 128, 108 bytes of invocation-local storage,
module constants, and six texture metadata records. The redistributable
`force_math_probe.ptx` exercises these added forms.

`KernelDeformableUberCollision` adds finite scalar/vector subword memory,
16-bit select/comparison, float comparison/minimum, five-barrier, and
17-texture-reference coverage. Its 3,662 instructions lower to a deterministic
175,407-word module with local size 64 and 4,097 bytes of static shared memory.
The redistributable `collision_types_probe.ptx` retains the added instruction
forms; local captured execution is available through
`run-tests.ps1 -CapturedPtxDir`.

`PxgCudaFluidCollisionConstraintsBatch` adds two- and four-component shared
float loads and four-component shared integer loads. Its 725 instructions lower
to a deterministic 59,442-word module with local size 128 and 1,840 bytes of
static shared memory. `collision_types_probe.ptx` retains each shared-vector
form. The focused one-particle fixture exercises the trace-ordered no-contact
path and matches native CUDA exactly on Intel and NVIDIA Vulkan adapters.

`PxgCudaFluidCollisionDynamicResponseBatch` adds finite
`cvt.rn.f32.u32` conversion and `not.pred` lowering. Its 263 instructions lower
to a deterministic 14,671-word module with local size 128 and 304 bytes of
static shared memory. `force_math_probe.ptx` and `collision_types_probe.ptx`
retain the added forms. The focused active-constraint fixture consumes the
constraints output layout and matches native CUDA exactly on Intel and NVIDIA
Vulkan adapters.

`PxgCudaFluidCollisionTrianglesBatch` adds finite 64-bit integer register,
shared-memory, arithmetic, comparison, select, conversion, and count-leading-zero
lowering. Its 1,022 instructions lower to a deterministic 68,441-word module
with local size 64 and 4,912 bytes of static shared memory. The selected entry
requires Vulkan `shaderInt64`; the focused zero-triangle fixture matches native
CUDA on Intel and NVIDIA Vulkan adapters but does not cover triangle contact.

`PxgCudaFluidCollisionStaticResponseBatch` adds two-component shared `u32`
loads. Its 330 instructions lower to a deterministic 22,632-word module with
local size 128 and 304 bytes of static shared memory. The focused active-contact
fixture covers normalization, averaging, restitution, output compaction, and
particle/source pairing and matches native CUDA exactly on Intel and NVIDIA
Vulkan adapters. Alternate flags, tangential damping, distance clamping, and
plane projection remain uncovered.

`PxgCudaFluidCollisionBoxShapesBatch` needs no additional instruction lowering
beyond the existing collision surface. Its 1,288 instructions lower to a
deterministic 61,420-word module with local size 64 and 6,752 bytes of static
shared memory. The trace-shaped two-work-unit zero-box fixture crosses five
barriers and matches native CUDA exactly on Intel and NVIDIA Vulkan adapters.
Box-contact geometry remains uncovered.

`PxgCudaFluidCollisionCapsuleShapesBatch` also needs no additional instruction
lowering beyond the existing collision surface. Its 1,312 instructions lower
to a deterministic 103,800-word module with local size 64 and 2,816 bytes of
static shared memory. The minimum trace-shaped two-work-unit zero-capsule
fixture crosses five barriers and matches native CUDA exactly on Intel and
NVIDIA Vulkan adapters. Capsule-contact geometry remains uncovered.

`PxgCudaFluidCollisionConvexShapesBatch` needs no additional instruction
lowering beyond the existing collision surface. Its 1,433 instructions lower
to a deterministic 75,510-word module with local size 64 and 4,225 bytes of
static shared memory. The minimum trace-observed five-workgroup, two-work-unit
zero-convex fixture crosses three barriers on its in-range path and matches
native CUDA exactly on Intel and NVIDIA Vulkan adapters.
Convex-contact geometry remains uncovered.

Validate and execute the module:

```powershell
spirv-val.exe --target-env vulkan1.2 generated-oracle.spv
build/vulkan-probe/Release/vulkan-memory-smoke.exe `
  generated-oracle-results.json `
  --ptx-oracle generated-oracle.spv
```

The module declares SPIR-V `RoundingModeRTE` and
`SignedZeroInfNanPreserve` capabilities with corresponding float32 execution
modes. Both properties are evidence-backed target requirements and are
supported by the Intel, NVIDIA, and AMD devices already probed. Explicit
software FTZ normalization remains necessary because native float32 FTZ is not
available on every target device.

A local directory containing extracted PTX:

```powershell
build/ptx-frontend/Release/ptx-frontend.exe --corpus `
  path/to/extracted/ptx `
  corpus-manifest.json
```

Input files are sorted by filename. Repeated runs over identical bytes must
produce byte-identical JSON.

## Strict failure behavior

The parser returns nonzero and reports `path:line:column` for unsupported or
malformed syntax. It does not recover into a partial successful module.

Validated failures include:

- unterminated block comments;
- missing module headers;
- malformed parameter/storage arrays;
- missing instruction/declaration semicolons;
- unsupported scalar types or instruction families;
- duplicate labels;
- branches to unknown labels;
- labels without an instruction;
- definitions or uses of undeclared registers.

## Manifest contracts

### Oracle manifest

`expected/oracle-manifest.json` includes the complete project-owned routine IR:

- module metadata;
- parameter and storage layouts;
- register banks and types;
- routine-local `.local`/`.shared` names, spaces, aligned offsets, sizes, and
  external status;
- labels;
- normalized diagnostic operands and backend-consumed structured operands;
- normalized instruction base/modifiers;
- register definitions and uses;
- basic blocks and successor edges.

SHA-256:

```text
c61f5cf9cdb96e0ec97547d2ee010e76d97fcf953dc7a77683eb037312fa02a2
```

### Corpus manifest

`expected/corpus-manifest.json` omits proprietary instruction operands and
function bodies. It records module declarations, entry summaries, parameter and
storage layouts, exact variant sets, and aggregate CFG counts.

SHA-256:

```text
d49317aaeaf1f39f080d0cfdaa5c56b96dba8fa71ea06d76c672792b831e02ee
```

Observed aggregate structure:

| Metric | Value |
|---|---:|
| Modules | 17 |
| Entries | 70 |
| Functions | 184 |
| Constant globals | 7 |
| Texture references | 33 |
| Base instruction families | 32 |
| Exact variants | 147 |
| Basic blocks | 4,174 |
| CFG edges | 5,935 |

All 16 launched entries match the independent captured analysis for packed
parameter offsets/sizes/alignments, effective static shared memory, and static
local memory. Schema 3 additionally retains every routine storage symbol and
its aligned offset without redistributing proprietary instructions.

Every result-producing instruction records an inferred finite PTX result type.
Definitions are checked against declared register-bank widths/categories,
including widening loads, `mul.wide`, `clz.b64`, and split bit destinations.
Register uses are checked against finite opcode operand contracts for memory,
conversion, comparison, selection, texture, shift/extract, and arithmetic
forms. All 147 corpus variants pass these contracts.

## Current exclusions

The frontend does not yet provide:

- SSA construction or phi placement;
- general type inference beyond declared register banks and the 147 finite
  opcode contracts;
- reachability or dead-code elimination;
- SPIR-V emission beyond the project oracle and the fourteen launched entries
  marked **Exact fixture** in
  [`../../docs/research/kernel-status.md`](../../docs/research/kernel-status.md);
- shader caching;
- CUDA module/runtime integration.

Those are subsequent independently testable gates. Grammar coverage alone is
not translated-kernel correctness.
