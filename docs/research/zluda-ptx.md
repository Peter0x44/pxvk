# ZLUDA PTX study for the Mirror's Edge corpus

## Scope and conclusion

Research is pinned to ZLUDA commit
[`8a7931be87d758835776f54ea2e33d30621c74ce`](https://github.com/vosen/ZLUDA/tree/8a7931be87d758835776f54ea2e33d30621c74ce).
The scope is the 17 PTX 4.1 modules, 147 whole-corpus opcode variants, and 120
launched variants cataloged in `mirrors-edge-ptx-corpus.md`.

ZLUDA's frontend accepts the complete corpus, and its documented 32-bit PhysX
support lists Mirror's Edge as tested; see
[`docs/src/physx32.md`, lines 67–72](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/docs/src/physx32.md#L67-L72).
No ZLUDA parser diagnostic file was emitted for any of the 17 captured PTX
modules. The useful parts are its typed
parsing, normalization strategy, dead-code removal, explicit parameter
metadata, u32 pseudo-pointer lowering, and backend-helper decomposition.

The LLVM AMDGPU emitter, HIP texture objects, AMD address spaces/intrinsics, and
32→64-bit HIP ABI are not reusable pxvk architecture. A corpus-specific
Vulkan/SPIR-V frontend can be substantially smaller.

## Parser coverage

ZLUDA uses a tokenized, strongly typed Winnow parser. `parse_module_checked`
returns structured errors; release builds use `parse_module_unchecked` with
recovery and retain the best PTX member; see
[`ptx_parser/src/lib.rs`, lines 392–501](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx_parser/src/lib.rs#L392-L501)
and
[`zluda/src/impl/module.rs`, lines 165–239](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/module.rs#L165-L239).

The parser directly covers the corpus's:

- `.version`, `.target`, and `.address_size` headers;
- global variables, constant arrays, texture references, and shared declarations;
- `.visible .entry`, `.weak .func`, `.param`, `.align`, `.maxntid`, and
  `.minnctapersm`;
- scalar/vector types and aggregate byte arrays;
- labels, predicates, branches, and instruction modifiers.

Directive/function/parameter parsing is implemented in
[`ptx_parser/src/lib.rs`, lines 503–856](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx_parser/src/lib.rs#L503-L856).
Instruction parser generation begins at
[lines 1867–1928](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx_parser/src/lib.rs#L1867-L1928)
and accepts modifiers in arbitrary order while producing typed AST variants.

The captured modules include weak CUDA runtime helper bodies. They are not
called by any kernel. ZLUDA's unreachable-code pass removes every non-kernel
function not reached by a call before 32-bit conversion; see
[`remove_unreachable_basic_blocks.rs`, lines 9–82](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/remove_unreachable_basic_blocks.rs#L9-L82).
This matters because the 32-bit conversion pass intentionally rejects retained
non-kernel bodies. A pxvk frontend selecting individual `.entry` functions can
avoid compiling these unused weak helpers altogether.

## Normalization pipeline

`ptx::to_llvm_module` applies this ordered pipeline; see
[`ptx/src/pass/mod.rs`, lines 64–132](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/mod.rs#L64-L132):

1. normalize identifiers/scopes;
2. replace known external functions;
3. convert predication to explicit control flow;
4. optimize function argument representation;
5. resolve function pointers;
6. replace special registers with helper calls;
7. expand vector and aggregate operands;
8. insert post-saturation behavior;
9. deparamize ordinary functions;
10. normalize reciprocal/FP-required helper operations;
11. normalize basic blocks and remove unreachable blocks/functions;
12. derive global floating-point/denormal modes;
13. make loads/stores and implicit conversions explicit;
14. convert 32-bit modules to a 64-bit backend ABI;
15. replace backend-sensitive instructions with helper calls;
16. hoist globals;
17. emit LLVM IR and metadata.

This is broader than pxvk needs, but the phase separation is sound:

```text
parse
→ resolve names/types
→ normalize predication/vectors/conversions
→ remove unreachable code
→ make memory and address semantics explicit
→ lower backend-sensitive operations
→ emit backend IR
```

A future C++ frontend can use this shape without copying Rust types, LLVM IR,
or every general-PTX pass.

## Predication and control flow

`normalize_predicates` converts every predicated instruction into explicit
conditional branches and labels. A predicated `bra` is folded directly into a
two-way branch; other predicated instructions receive generated true/false
blocks. See
[`normalize_predicates.rs`, lines 51–86](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/normalize_predicates.rs#L51-L86).

The parser accepts `.uni` on `bra` but discards it as an optimization hint; see
[`ptx_parser/src/lib.rs`, lines 1695–1707](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx_parser/src/lib.rs#L1695-L1707).
LLVM emission then uses ordinary branches and comparisons. This is reusable:
pxvk does not need a distinct semantic instruction for `bra.uni`.

Special-register operands are converted into explicit external helper calls by
`fix_special_registers`; see
[`fix_special_registers.rs`, lines 1–75](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/fix_special_registers.rs#L1-L75)
and
[lines 112–176](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/fix_special_registers.rs#L112-L176).
The captured launched entries use only `%tid.x`, `%ntid.x`, `%ctaid.x`, and
`%nctaid.x`. Vulkan/SPIR-V can map these directly to local invocation ID,
workgroup size, workgroup ID, and workgroup count rather than helper functions.

## 32-bit address conversion

ZLUDA must execute 32-bit PTX on a 64-bit HIP runtime. Its
`convert_32bit_to_64bit` pass:

- extracts `.global` and `.const` byte arrays as metadata;
- preserves shared variables and texture references;
- records every explicit kernel argument's `(size, align)`;
- removes module globals from generated code;
- adds a hidden 64-bit allocation-base argument to each kernel;
- adds one hidden u32 pseudo-pointer per global;
- rewrites global references to those hidden arguments.

See
[`convert_32bit_to_64bit.rs`, lines 1–53](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/convert_32bit_to_64bit.rs#L1-L53),
[lines 56–139](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/convert_32bit_to_64bit.rs#L56-L139), and
[lines 141–229](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/convert_32bit_to_64bit.rs#L141-L229).

During LLVM emission, a u32 global/generic/constant address is zero-extended,
added to the hidden base, and converted into a backend pointer; see
[`llvm/emit.rs`, lines 747–831](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L747-L831).

The reusable semantic rule is:

```text
CUDA u32 device pointer = byte offset into an explicitly bound memory object
```

pxvk does not need to turn that into a native 64-bit pointer. SPIR-V can use the
u32 offset to index a storage buffer, subject to the layout and robust-access
requirements to be established in `vulkan-requirements.md`.

## Module compilation and metadata

ZLUDA's module loader chooses PTX, parses it, runs the pipeline, emits LLVM,
links helper bitcode, compiles an AMDGPU ELF, and caches by PTX hash, compiler
version, ZLUDA version, device ISA, and compile attributes; see
[`zluda/src/impl/module.rs`, lines 165–363](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/module.rs#L165-L363).

The compiled object carries separate metadata for:

- original SM version;
- 32-bit globals and initializers;
- explicit argument size/alignment per entry.

The runtime exposes globals and function-argument metadata through its private
ZLUDA32 table; see
[`zluda/src/impl/driver.rs`, lines 482–558](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/driver.rs#L482-L558).

Reusable idea: backend shader bytes and CUDA-facing metadata should be cached
together. Cache format and compiler choice remain deferred.

## Instruction coverage strategy

All 147 whole-corpus variants are syntactically accepted and compile under
ZLUDA's tested PhysX path. Their lowering falls into three groups.

### Direct scalar/vector LLVM emission

The emitter directly handles the captured families for:

```text
abs add and bra clz cvt cvta fma ld mad max min mov mul neg not or
rcp ret rsqrt selp setp shl shr sqrt st sub xor
```

Examples:

- integer add/multiply/shift/boolean operations map to LLVM integer builders;
- `mul.wide.s16` sign-extends into a double-width integer, multiplies, and keeps
  the wide result;
- `clz.b64` maps to `llvm.ctlz.i64` and returns/truncates the PTX-required count;
- `selp` maps to select;
- `setp` maps signed/unsigned/ordered/unordered comparisons;
- `fma.rn.f32` maps to LLVM fused FMA;
- scalar/vector loads and stores preserve declared alignment and volatility.

Representative source is
[`llvm/emit.rs`, lines 617–706](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L617-L706),
[`llvm/emit.rs`, lines 710–744](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L710-L744), and
[`llvm/emit.rs`, lines 1431–1549](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L1431-L1549).

### Backend helper calls

`replace_instructions_with_functions` converts backend-sensitive operations to
named helper calls before LLVM emission; see
[`replace_instructions_with_functions.rs`, lines 340–455](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/replace_instructions_with_functions.rs#L340-L455) and
[lines 510–535](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/replace_instructions_with_functions.rs#L510-L535).

Observed mappings include:

| PTX | ZLUDA lowering |
|---|---|
| `div.full.ftz.f32` | `div_full_f32` helper |
| `bar.sync` | `bar_sync` helper |
| `tex.1d.v4.*.s32` | typed `texref_1d_v4_*_s32` helper |
| whole-corpus-only `bfe.u32` | `bfe_u32` helper |

The helpers are HIP/AMDGPU-specific and live in
[`ptx/lib/zluda_ptx_impl.cpp`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/lib/zluda_ptx_impl.cpp).
pxvk should reproduce semantics, not link this bitcode.

### Normalization-only forms

Vector operands, by-value byte arrays, bit types used as arithmetic types,
implicit pointer conversions, and predicated forms are expanded into simpler
typed operations before backend emission. This avoids multiplying backend cases
for every PTX spelling.

## Floating-point behavior

The corpus depends on explicit floating-point modes:

- pervasive `.ftz.f32` arithmetic;
- `fma.rn.ftz.f32`;
- `div.full.ftz.f32`;
- `sqrt.approx.ftz.f32`;
- `rsqrt.approx.ftz.f32`;
- `rcp.approx.ftz.f32`;
- `cvt.rmi`, `cvt.rzi`, and nearest-even conversions.

ZLUDA analyzes instruction modes across the CFG and records per-kernel denormal
and rounding defaults; see
[`instruction_mode_to_global_mode/mod.rs`, lines 959–1003](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/instruction_mode_to_global_mode/mod.rs#L959-L1003).
LLVM functions receive denormal-mode attributes and may become `strictfp`; see
[`llvm/emit.rs`, lines 119–159](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L119-L159).
Directed conversions use LLVM/constrained rounding helpers, while FMA uses
normal or constrained fused intrinsics; see
[`llvm/emit.rs`, lines 2387–2509](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L2387-L2509) and
[lines 2948–2981](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/emit.rs#L2948-L2981).

Approximate reciprocal/square-root operations use AMDGPU intrinsics when they
reach the emitter; the helper library contains explicit subnormal scaling for
non-FTZ helper paths. `div_full_f32` reconstructs division from mantissa,
exponent, reciprocal, and `ldexp`; see
[`zluda_ptx_impl.cpp`, lines 375–423](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/lib/zluda_ptx_impl.cpp#L375-L423) and
[lines 648–657](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/lib/zluda_ptx_impl.cpp#L648-L657).

This is the highest semantic risk for Vulkan. The requirements study must map
the observed modes to SPIR-V/Vulkan float controls and determine where shader
code must emulate NVIDIA approximate results. Ordinary arithmetic lowering is
not enough for a reliable oracle comparison.

## Memory-space lowering

ZLUDA maps PTX spaces to AMD LLVM address spaces:

| PTX | AMD LLVM space |
|---|---:|
| generic | 0 |
| global | 1 |
| shared | 3 |
| constant/entry parameter | 4 |
| local/register | 5 |

See
[`llvm/mod.rs`, lines 12–65](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/llvm/mod.rs#L12-L65).
These numeric spaces are not portable to SPIR-V and must not be copied.

Reusable semantics are:

- parameter bytes are immutable per dispatch;
- constants/globals have module lifetime;
- local memory is invocation-private;
- shared memory is workgroup-local;
- global memory uses u32 byte offsets into the CUDA virtual allocation;
- vector memory operations preserve component order, width, and alignment;
- volatile shared accesses must remain volatile.

The captured corpus has no atomics or memory fences, which keeps initial Vulkan
memory semantics narrow.

## Barrier lowering

`bar.sync` is legalized to a helper call. The AMD helper emits a sequentially
consistent workgroup fence followed by an AMD workgroup barrier; see
[`zluda_ptx_impl.cpp`, lines 276–280](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/lib/zluda_ptx_impl.cpp#L276-L280).

All captured barriers use barrier ID 0 and no thread-count operand. There are no
`bar.red`, `bar.warp`, atomics, `membar`, or cross-block synchronization in the
corpus. Vulkan lowering therefore needs one full-workgroup execution barrier
with workgroup-memory ordering; the precise SPIR-V scopes/semantics belong in
`vulkan-requirements.md`.

## Texture lowering

PTX texture instructions are converted to helper names based on texture kind,
dimension, result type, and coordinate type. For 32-bit U32 results, ZLUDA
normalizes the helper result to S32 bit-equivalent form; see
[`replace_instructions_with_functions.rs`, lines 365–395](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/replace_instructions_with_functions.rs#L365-L395).

HIP helpers dereference a `textureReference`, obtain its texture object, and for
signed integer coordinates perform a structured-buffer load; see
[`zluda_ptx_impl.cpp`, lines 1142–1189](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/lib/zluda_ptx_impl.cpp#L1142-L1189).

The captured texture surface is only:

```text
tex.1d.v4.f32.s32
tex.1d.v4.u32.s32
```

with linear bindings, point filtering, clamp mode, and integer coordinates.
This may map more naturally to typed storage-buffer loads than sampled Vulkan
images. That remains a candidate, not an architecture decision; format
reinterpretation and out-of-range clamp behavior need explicit validation.

## Whole-corpus support and unsupported paths

The captured PTX uses 147 variants across 32 base instruction families. ZLUDA
has a parser/emitter or helper path for every one. The 27 variants absent from
launched entries—including `bfe.u32`—are still present in loaded kernel entries
and are handled by the same module compilation.

No observed construct requires ZLUDA's implementations for atomics, memory
fences, warp votes/shuffles, tensor operations, surfaces, doubles, or modern
texture objects. Source TODO/error branches for those unrelated paths are not
pxvk requirements.

Relevant limitations to avoid copying blindly:

- release parsing can recover from invalid directives and compile a partial
  module; pxvk should report deterministic diagnostics for its finite corpus;
- 32-bit conversion assumes global/constant byte arrays with immediate
  initializers, which happens to match all seven corpus globals;
- the HIP emitter's address spaces and intrinsics are backend-specific;
- helper functions encode AMD approximation and texture-object behavior;
- LLVM's incomplete constrained-FP support forced ZLUDA-specific workarounds;
- whole-module generality creates many passes irrelevant to 16 launched entries.

## Reusable frontend ideas

1. Strongly typed AST nodes for opcode modifiers and operands.
2. Parsing modifiers independent of textual order.
3. Early source spans and deterministic diagnostics.
4. Normalize identifiers, predicates, vectors, conversions, and memory accesses
   before backend emission.
5. Remove unreachable blocks and uncalled weak functions before lowering.
6. Compile/select individual entry points while retaining module globals and
   texture metadata.
7. Record explicit `(size, align)` launch metadata beside compiled shaders.
8. Treat u32 CUDA addresses as offsets into explicitly bound memory.
9. Isolate backend-sensitive operations—barriers, textures, approximate math—
   behind narrow semantic lowering functions.
10. Cache backend code together with PTX hash, translator version, device key,
    and 32-bit module metadata.

## What not to reuse

1. LLVM AMDGPU address-space numbers.
2. HIP module, texture, stream, or event objects.
3. AMDGPU intrinsics and OCKL helper calls.
4. The 32→64-bit hidden pointer ABI in its exact form.
5. Linked HIP helper bitcode.
6. Global floating-mode register manipulation designed around AMD LLVM.
7. Release-mode partial parsing as normal success behavior.
8. General PTX instruction coverage beyond the finite corpus.

## Consequences for pxvk

A first translator can target the 16 launched entries and still parse module
metadata for all 17 modules. Its coherent frontend obligations are now:

```text
PTX 4.1 text
→ typed finite AST
→ entry/global/texref metadata
→ explicit predicated CFG
→ explicit scalar/vector conversions and memory operations
→ u32 device-offset lowering
→ backend-sensitive barrier/texture/FP lowering
→ Vulkan shader representation
```

The open decision is the final shader representation and emission route, not
what PTX must be understood. `vulkan-requirements.md` must next determine:

- SPIR-V float-control support for FTZ and rounding;
- approximate math strategy and oracle tolerance;
- workgroup barrier scopes/semantics;
- workgroup/private/storage-buffer layouts and limits;
- typed storage-buffer versus image lowering for the two texture operations;
- subgroup-size controls needed to keep CUDA-visible warp size 32 even though
  this corpus has no explicit warp intrinsic.

## Remaining validation questions

- Compare selected NVIDIA outputs for `div.full.ftz`, reciprocal, square root,
  reciprocal square root, and FMA against candidate Vulkan lowering.
- Verify texture out-of-range behavior for signed 1D coordinates and clamp mode.
- Verify static plus dynamic shared layout and alignment for the largest three
  kernels.
- Decide whether module `0017` self-collision entries are included in the first
  slice despite not launching in captured scenes.
- Confirm that compiling individual entries instead of whole modules preserves
  every referenced constant, texture, and shared declaration.
