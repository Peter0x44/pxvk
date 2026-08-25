# Mirror's Edge PTX corpus

## Status

The CUDA code can be extracted statically from the PhysX runtime DLL. The three
native captures resolve 70 CUDA entry points across 17 unique PTX modules; 16
entry points in 15 modules actually launched. Their packed parameter sizes map
exactly to the PTX declarations.

This document describes the captured system PhysX 2.8.0 runtime selected by the
GOG installation, not the obsolete bundled `PhysXCore.bak`.

## Provenance and static extraction

`MirrorsEdge.exe` has only ordinary PE `.text`, `.rdata`, `.data`, `.rsrc`, and
`.reloc` sections. Its CUDA kernels come from:

```text
C:\Program Files (x86)\NVIDIA Corporation\PhysX\Engine\v2.8.0\PhysXCore.dll
SHA-256: ed8a3f0cf46045699f05437348a1bd589a4e023ade5697c13dbc6c56befa545d
```

That DLL contains:

| PE section | Size | Purpose |
|---|---:|---|
| `.nv_fatb` | 2,104,736 B | Seventeen concatenated CUDA fatbins |
| `.nvFatBi` | 272 B | Fatbin registration metadata/wrappers |

A copy can be extracted without launching the game:

```bat
copy "C:\Program Files (x86)\NVIDIA Corporation\PhysX\Engine\v2.8.0\PhysXCore.dll" tmp\PhysXCore.dll
objcopy --dump-section .nv_fatb=tmp\physxcore.nv_fatb ^
        --dump-section .nvFatBi=tmp\physxcore.nvFatBi ^
        tmp\PhysXCore.dll
```

The `.nv_fatb` section begins with a valid `0xba55ed50` header and consists of
17 adjacent `header_size + files_size` records with no gaps. Every raw fatbin
dumped by the project tracer byte-for-byte matches one of those 17 static
records. Trace ordering varies, but the hash sets are identical.

CUDA 13.3 `cuobjdump --dump-ptx` cannot process the DLL because support for an
embedded `sm_20` image was removed. The current tool exits with:

```text
cuobjdump fatal: Support for 'sm_20' has been removed. Please use CUDA toolkit 9.1 or earlier.
```

This does not block extraction. The project tracer saves each raw fatbin, and
ZLUDA's parser split the same records into PTX and ELF images. The normalized
analysis is preserved locally as
`tmp/evidence/mirrors-edge-ptx-corpus-analysis.json`, SHA-256
`6ce4c5c05c45ceb9f010d3575f06cb9785420f9f3083197542be8b253f600da5`.
Raw fatbins and PTX remain local because they are proprietary runtime material.

## Corpus summary

Every PTX module declares:

```ptx
.version 4.1
.target sm_52
.address_size 32
```

| Metric | Count |
|---|---:|
| Unique raw fatbins | 17 |
| Unique PTX modules | 17 |
| PTX bytes | 1,273,679 |
| PTX `.entry` declarations | 70 |
| Runtime-resolved functions | 70 |
| Launched entries | 16 |
| Modules containing launched entries | 15 |
| Legacy texture references | 33 |
| Constant/global symbols | 7 |
| Launched opcode variants | 120 |
| Whole-corpus opcode variants | 147 |

All 17 PTX SHA-256 values are distinct:

| Module | Semantic content | Bytes | Entries (launched) | SHA-256 |
|---|---|---:|---:|---|
| `0001` | Fluid convex-shape collision | 52,461 | 1 (1) | `bbf86968be45a385f33e5d18f62bb0e0d49c0ba3613186ffbb9866c7d87fca6f` |
| `0002` | Fluid triangle collision | 43,886 | 1 (1) | `6f6b0226638b4743ef6c044b9f7f2c576f12756c2d1b856164d843971819b090` |
| `0003` | Fluid capsule-shape collision | 52,681 | 1 (1) | `21b20f25a59e97c8876a81edcd0c841ea5933cce5e62df009a99a00b8f69915d` |
| `0004` | Radix sort, scan, and helpers | 324,244 | 42 (1) | `2c8127a97432ebcd41b43c1a2502c34e03b8eaf89d94ae222d6db9da9089061b` |
| `0005` | Fluid collision constraints | 31,376 | 1 (1) | `67efc0147e095565d9575a92f8c5fa69e581f64838c52ec3ed2373aa5e32cbe2` |
| `0006` | Collision constants only | 1,496 | 0 (0) | `3b86db3e6567ceccd6fd6924597d626bdcd85de5a3d31a4811d69b05a9394cba` |
| `0007` | Fluid dynamic response | 13,987 | 1 (1) | `bae9511071b8ef27d4c25af12530a5ce633c115ba205e40156bb6ad69f1625d5` |
| `0008` | Deformable update | 9,439 | 1 (1) | `36624e3ea48fc608757992c7a622386d77b15fa6999593615ff7ab14b66ee03a` |
| `0009` | Fluid static response | 17,432 | 1 (1) | `ea76c3df93ab7546d244322234c20bdeb260c7611279f2a9b881718ab4028c09` |
| `0010` | Fluid density | 17,450 | 1 (1) | `f53b6418ca3f3e2a25b342b934b654cff497a4137120f4034dd81614ddd09289` |
| `0011` | Deformable collision | 374,091 | 1 (1) | `d08a3584a8f4bb25493b8987eaedbb18e351698dfa0d74bb3911dd1e6fbb0a49` |
| `0012` | Deformable solvers | 179,598 | 7 (1) | `8d751c7ec2eebe0e8a596117ff97744c8e842a05a529a2f557e0c5aa7476157a` |
| `0013` | Fluid hash and reorder | 23,300 | 4 (2) | `b1fbf0705466b24205ad8f453b53316ed04562a59d3209eecaa86b79f6598926` |
| `0014` | Simple particles | 13,676 | 1 (1) | `fef25e48953fc5193cf776efb6a1f6244eedbf47b062ec787c152aa697d3c44a` |
| `0015` | Fluid box-shape collision | 49,677 | 1 (1) | `48edfac30999f4b0d6ce7cf8f9f83fa1413f61bbb10c842cacf6b2580d7cd169` |
| `0016` | Fluid force | 28,532 | 1 (1) | `ce72dad51e4d4aaa52e27358380061727086d1515851573e4020357b0580f240` |
| `0017` | Deformable self-collision | 40,353 | 5 (0) | `17b39c7583106e502d60a0a8fb5aa36439e4f5f40c64b495581202e9fafacff7` |

Module `0006` is intentionally entry-less: it supplies the 160-byte
`cCollisionParameters` constant symbol. Module `0017` was loaded and its five
entries were resolved, but no captured scene launched them.

## Launched entries

Sixteen entries launched across the tutorial, subway, and server-room traces:

| Module | Entry stem | Parameter bytes | Static shared | Observed dynamic shared |
|---|---|---:|---:|---:|
| `0001` | `PxgCudaFluidCollisionConvexShapesBatch` | 20 | 4,225 B | 0 |
| `0002` | `PxgCudaFluidCollisionTrianglesBatch` | 20 | 4,912 B | 0 |
| `0003` | `PxgCudaFluidCollisionCapsuleShapesBatch` | 20 | 2,816 B | 0 |
| `0004` | `radixSortBlocks<32,0,false>` | 20 | 4 B | 4,096 B |
| `0005` | `PxgCudaFluidCollisionConstraintsBatch` | 20 | 1,840 B | 0 |
| `0007` | `PxgCudaFluidCollisionDynamicResponseBatch` | 20 | 304 B | 0 |
| `0008` | `PxgCudaDeformableUpdate` | 12 | 0 | 0 |
| `0009` | `PxgCudaFluidCollisionStaticResponseBatch` | 20 | 304 B | 0 |
| `0010` | `KernelFluidDensity` | 84 | 0 | 0 |
| `0011` | `KernelDeformableUberCollision` | 4 | 4,097 B | 0 |
| `0012` | `KernelDeformableUberSolver<512>` | 16 | 0 | 7,072 B |
| `0013` | `KernelFluidCalcHashPacked` | 52 | 0 | 0 |
| `0013` | `KernelReorderPosVelAndFindCellStartPacked` | 76 | 0 | 260 B |
| `0014` | `KernelFluidSimpleParticle` | 44 | 10,756 B | 0 |
| `0015` | `PxgCudaFluidCollisionBoxShapesBatch` | 20 | 6,752 B | 0 |
| `0016` | `KernelFluidForce` | 100 | 0 | 0 |

Maximum observed per-block shared memory is 10,756 bytes. Maximum static local
memory declaration in a launched entry is 108 bytes. The deformable solver
also declares `.maxntid 512, 1, 1` and `.minnctapersm 1`; its observed block X
was 64.

## Packed launch ABI

The captured packed-buffer length exactly equals the size obtained by laying
out each PTX `.param` declaration in order with its declared or natural
alignment. All 16 launched entries match; no unexplained trailing bytes remain.

Layout shorthand below is `offset:size/alignment`:

| Entry stem | PTX parameter layout | Total |
|---|---|---:|
| `radixSortBlocks<32,0,false>` | `0:4/4, 4:4/4, 8:4/4, 12:4/4, 16:4/4` | 20 |
| `KernelFluidForce` | `0:8/4, 8:32/4, 40:4/4, 44:8/4, 52:8/4, 60:8/4, 68:8/4, 76:8/4, 84:8/4, 92:8/4` | 100 |
| `KernelFluidDensity` | `0:8/4, 8:8/4, 16:24/4, 40:4/4, 44:8/4, 52:8/4, 60:8/4, 68:8/4, 76:8/4` | 84 |
| `PxgCudaDeformableUpdate` | `0:4/4, 4:4/4, 8:4/4` | 12 |
| `KernelDeformableUberSolver<512>` | `0:4/4, 4:1/1, 8:4/4, 12:4/4` | 16 |
| `KernelFluidCalcHashPacked` | `0:8/4, 8:8/4, 16:8/4, 24:8/4, 32:8/4, 40:8/4, 48:4/4` | 52 |
| `KernelFluidSimpleParticle` | `0:8/4, 8:8/4, 16:8/4, 24:8/4, 32:8/4, 40:4/4` | 44 |
| `KernelDeformableUberCollision` | `0:4/4` | 4 |
| Six fluid collision families | `0:8/4, 8:8/4, 16:4/4` | 20 |
| `KernelReorderPosVelAndFindCellStartPacked` | nine `8/4` records at offsets 0–64, then `72:4/4` | 76 |

Because `.address_size` is 32, device addresses loaded by the entries are
32-bit. An eight-byte `.b8[8]` parameter is a by-value aggregate and must not be
assumed to be a single 64-bit pointer. Its field meaning belongs in the CUDA
contract or a focused native probe; its byte layout is already known.

## Declaration surface

Required module-level syntax:

- `.version 4.1`, `.target sm_52`, `.address_size 32`;
- `.visible .entry` and `.weak .func`;
- typed and byte-array `.param` declarations with `.align`;
- `.reg`, `.local`, `.shared`, `.extern .shared`, `.const`, `.global .texref`;
- `.maxntid` and `.minnctapersm`;
- predicate registers and predicated instructions.

Declared scalar types are:

```text
.b8 .b16 .b32 .b64
.u8 .u16 .u32 .u64
.s8 .s16 .s32 .s64
.f32
.pred
```

There is no `.f64`, half precision, tensor operation, surface reference, or
modern object-based texture declaration. Launched entries use only `%tid.x`,
`%ntid.x`, `%ctaid.x`, and `%nctaid.x`; Y/Z thread and block built-ins are not
referenced.

## Instruction surface

The 16 launched entries use 31 base instruction families:

```text
abs add and bar bra clz cvt cvta div fma ld mad max min mov mul neg not or
rcp ret rsqrt selp setp shl shr sqrt st sub tex xor
```

The exact launched variants are:

```text
abs.ftz.f32
add.ftz.f32 add.s32 add.s64 add.u32
and.b16 and.b32 and.b64 and.pred
bar.sync
bra bra.uni
clz.b64
cvt.rmi.ftz.f32.f32 cvt.rn.f32.u32 cvt.rzi.ftz.s32.f32
cvt.s32.s16 cvt.u16.u32 cvt.u32.u64
cvta.to.global.u32
div.full.ftz.f32
fma.rn.ftz.f32
ld.const.f32 ld.const.u32 ld.const.u8
ld.global.f32 ld.global.u16 ld.global.u32 ld.global.u8
ld.global.v4.f32 ld.global.v4.u16 ld.global.v4.u32
ld.local.u32
ld.param.s8 ld.param.u32
ld.shared.f32 ld.shared.u32 ld.shared.u64 ld.shared.u8
ld.shared.v2.f32 ld.shared.v2.u32 ld.shared.v4.f32 ld.shared.v4.u32
ld.volatile.shared.u32
mad.lo.s32
max.ftz.f32 max.s32
min.ftz.f32 min.s32 min.u32
mov.b32 mov.f32 mov.pred mov.u16 mov.u32 mov.u64
mul.ftz.f32 mul.lo.s32 mul.wide.s16
neg.ftz.f32 not.pred
or.b16 or.b32 or.b64 or.pred
rcp.approx.ftz.f32
ret
rsqrt.approx.ftz.f32
selp.b16 selp.b32 selp.b64 selp.f32 selp.u16 selp.u32
setp.eq.b16 setp.eq.b32 setp.eq.ftz.f32 setp.eq.s16 setp.eq.s32 setp.eq.s64
setp.ge.ftz.f32 setp.ge.u32 setp.geu.ftz.f32
setp.gt.ftz.f32 setp.gt.s32 setp.gt.u32 setp.gtu.ftz.f32
setp.le.ftz.f32 setp.le.u32 setp.leu.ftz.f32
setp.lt.ftz.f32 setp.lt.s16 setp.lt.u32 setp.ltu.ftz.f32
setp.ne.s16 setp.ne.s32 setp.ne.s64 setp.neu.ftz.f32
shl.b32 shl.b64 shr.u32
sqrt.approx.ftz.f32
st.global.f32 st.global.u32 st.global.v2.f32 st.global.v4.f32 st.global.v4.u32
st.local.u32
st.shared.f32 st.shared.u32 st.shared.u64 st.shared.u8
st.shared.v2.f32 st.shared.v4.f32 st.volatile.shared.u32
sub.ftz.f32 sub.s32
tex.1d.v4.f32.s32 tex.1d.v4.u32.s32
xor.b64
```

Twenty-seven additional variants occur only in loaded but unlaunched entries or
weak helper functions:

```text
bfe.u32
cvt.s32.s8 cvt.u32.u16
ld.f32 ld.global.v2.u16 ld.global.v2.u32 ld.param.f32 ld.s16 ld.u16 ld.u32
ld.u8 ld.v2.f32 ld.v4.f32 ld.v4.u32 ld.volatile.u32
neg.s32 setp.ge.s32 setp.lt.s32 shr.s32
st.f32 st.param.b32 st.param.f32 st.shared.v2.u32 st.u32 st.v2.f32
st.v4.u32 st.volatile.u32
```

`bfe` is the only whole-corpus base family absent from launched entries. A
translator compiling only selected entries can defer these corpus-only forms;
a whole-module parser cannot.

## Memory, synchronization, and texture requirements

Launched memory operations cover:

- 32-bit parameter and address arithmetic;
- global, constant, local, shared, and parameter loads;
- scalar, `v2`, and `v4` global/shared loads and stores;
- `u8`, `u16`, `u32`, `u64`, and `f32` memory elements;
- volatile shared `u32` loads and stores;
- static and dynamically sized shared memory.

Synchronization is limited to `bar.sync`. There are no observed atomics,
reductions, `membar`, fences, CUDA calls from launched kernels, or cross-block
synchronization.

Texture use is legacy and narrow:

- 33 module texture references, 27 reachable from launched modules;
- only `tex.1d.v4.f32.s32` and `tex.1d.v4.u32.s32` instructions;
- signed 32-bit coordinates;
- driver traces bind linear addresses with point filtering and clamp mode.

No launched or loaded PTX uses `vote`, `shfl`, `match`, `%laneid`, `%warpid`, or
lane-mask registers. Warp size 32 remains part of the CUDA-visible device
contract, and the radix code names/algorithms contain 32-thread assumptions,
but this corpus has no explicit warp-intrinsic lowering requirement.

## Constant and texture modules

The seven constant/global symbols are:

| Module | Symbol | Bytes |
|---|---|---:|
| `0006` | `cCollisionParameters` | 160 |
| `0008` | `updateInstanceData` | 7,168 |
| `0010` | `parameters_cmem` | 10,240 |
| `0011` | `gDeformableCollisionInstanceData` | 17,920 |
| `0011` | `gDeformableCollisionBatchedData` | 72 |
| `0012` | `solverGpuData` | 5,888 |
| `0016` | `parameters_cmem` | 10,240 |

Texture references occur only in modules `0010` (4), `0011` (17), `0016` (6),
and unlaunched self-collision module `0017` (6).

## Minimum translator consequence

A first coherent PTX execution slice can be narrower than the complete corpus:

1. Parse PTX 4.1 modules and select individual `.entry` functions.
2. Support the 120 launched opcode variants and required declarations above.
3. Materialize packed `.param` bytes at the proven offsets.
4. Provide 32-bit global addresses, constant storage, private/local storage,
   static/dynamic shared storage, and legacy 1D texture bindings.
5. Implement block barriers and predicated control flow.
6. Preserve FTZ, rounding, approximate reciprocal/square-root, and fused FMA
   semantics closely enough for oracle comparison.

This is a workload contract, not a choice of IR, parser library, shader
language, or Vulkan architecture.

## Remaining questions

- Map aggregate `.b8[]` parameter fields to semantic host structures where
  validation needs named values.
- Capture a deterministic output-buffer oracle for at least one fluid and one
  deformable step.
- Decide whether unlaunched module `0017` belongs in the first compatibility
  slice or remains evidence-backed deferred coverage.
- Determine Vulkan feature/limit requirements for the exact shared memory,
  32-bit addressing model, texture reads, and floating-point modes above.
