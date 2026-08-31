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

Kernel mode binds the 32-bit CUDA heap at descriptor binding 0, packed
launch-parameter words at binding 1, and module `.const` storage at read-only
storage-buffer binding 2 when the selected module declares constants. PTX byte
addresses remain `u32` byte offsets within their memory space. The
dynamic-shared byte count is explicit pipeline input; the emitter combines it
with module/routine static shared layouts to size the Workgroup arena. Named
`.shared` symbols resolve to aligned byte offsets in that arena. Named `.local`
symbols resolve to aligned byte offsets in a per-invocation Function array.
Named `.const` symbols resolve to aligned byte offsets in binding 2.

The current state-machine lowering stores virtual registers in function storage
and executes one PTX basic block per structured loop iteration. A finite prelude
structurizes the paired reorder kernel's divergent shared-memory setup so all
invocations reconverge before its sole workgroup barrier. This is intentionally
conservative; SSA and general structured CFG reconstruction remain later
optimizations.


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
- SPIR-V emission beyond the project oracle and selected
  `KernelFluidCalcHashPacked`/`KernelReorderPosVelAndFindCellStartPacked`
  finite slices;
- shader caching;
- CUDA module/runtime integration.

Those are subsequent independently testable gates. Grammar coverage alone is
not translated-kernel correctness.
