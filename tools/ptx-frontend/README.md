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
  -> basic-block CFG
  -> per-instruction register definitions and uses
  -> deterministic JSON manifests
```

The next stage will consume this IR to emit SPIR-V. It must not reparse operand
strings independently.

## Build

CMake fetches pinned nlohmann/json commit
`9cca280a4d0ccf0c08f47a99aa71d1b0e52f8d03`. That dependency is used only for
JSON serialization; PTX grammar, validation, CFG construction, and dataflow are
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
- labels;
- normalized instruction base/modifiers/operands;
- register definitions and uses;
- basic blocks and successor edges.

SHA-256:

```text
7ad88143948504e12d6cb0f2eae59fd1521749dab79237bd4511e064d2361054
```

### Corpus manifest

`expected/corpus-manifest.json` omits proprietary instruction operands and
function bodies. It records module declarations, entry summaries, parameter and
storage layouts, exact variant sets, and aggregate CFG counts.

SHA-256:

```text
b46479bf9a21121b448216eacfa66344551ddfeee60f352f6d9dc536e7682cde
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
local memory.

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
- SPIR-V emission;
- shader caching;
- CUDA module/runtime integration.

Those are subsequent independently testable gates. Grammar coverage alone is
not translated-kernel correctness.
