# Native NVIDIA oracle probes

## Status

A project-owned x86 CUDA Driver API probe now records exact NVIDIA results for
the PTX floating-point, workgroup-barrier, and legacy linear-texture semantics
that are most backend-sensitive in the Mirror's Edge corpus.

Implemented at:

```text
tools/cuda-oracle/
├── CMakeLists.txt
├── main.c
├── probe.ptx
└── expected/rtx3070-driver610.88.json
```

The committed result fixture has SHA-256:

```text
a69570c48f5361afafbcca15408a350f19b9782730f613b35695692ce20141f9
```

Three native executions produced byte-identical JSON.

## Environment

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| Display driver | 610.88 |
| Process | PE32/i386 |
| Compiler | `i686-w64-mingw32-gcc` 16.2.0 |
| CUDA API | Driver API loaded dynamically from x86 `nvcuda.dll` |
| PTX | 4.1, `sm_52`, `.address_size 32` |

The probe does not link CUDART and does not pass through ZLUDA or pxvk. It
creates a native NVIDIA context, JIT-loads PTX text, allocates device memory,
launches through `cuLaunchKernel`, and copies exact result bits back.

## Build and run

```bat
cmake -S tools/cuda-oracle -B build/cuda-oracle -G "MinGW Makefiles" ^
  -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc ^
  -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build/cuda-oracle --parallel

build\cuda-oracle\cuda-oracle.exe ^
  build\cuda-oracle\probe.ptx ^
  native-result.json
```

The output can be compared byte-for-byte with the committed fixture.

## Probe questions and decisions

| Question | Native probe | Result | Contract consequence |
|---|---|---|---|
| Are f32 subnormal inputs/outputs flushed for `.ftz` arithmetic? | Divide minimum normal by two; operate on minimum subnormal | All selected subnormal results became zero | Initial backend must enable FTZ or emulate it |
| Is FMA genuinely fused? | Compare `fma.rn.ftz` with separate `mul.ftz` + `add.ftz` | FMA returned nonzero; separate path returned zero | Do not permit contraction policy to change explicit FMA/separate operations |
| What are native approximate math bits? | `rcp`, `sqrt`, and `rsqrt` on fixed inputs | Exact values recorded below | Vulkan results must be compared bitwise or against an explicitly approved tolerance |
| Does `bar.sync 0` make all shared writes visible? | 64 threads write, barrier, read neighbor | Exact cyclic neighbor sequence | Lower to full-workgroup execution + workgroup-memory barrier |
| How does an integer-coordinate linear texref handle OOB? | Coordinates -2 through 9 over eight `uint4` elements | In-range exact; negative and high OOB return zero | Storage-buffer lowering needs explicit bounds-to-zero, not edge clamping |
| How are NaNs handled by observed min/max order? | NaN as first operand, 1.0 as second | Both return 1.0 | Preserve NVIDIA operand-order behavior for these forms |
| How do directed/int conversions behave? | `cvt.rzi.s32.f32(-3.75)` and `cvt.rn.f32.u32(16777217)` | -3 and 16777216.0 | Match truncation and nearest-even rounding |

## Floating-point results

Exact returned f32/u32 bit patterns:

| Probe | Result bits | Interpretation |
|---|---|---|
| `div.full.ftz.f32(1, 3)` | `0x3eaaaaab` | 0.3333333432674408 |
| `div.full.ftz.f32(min_normal, 2)` | `0x00000000` | positive zero; subnormal output flushed |
| `rcp.approx.ftz.f32(3)` | `0x3eaaaaab` | native reciprocal approximation |
| `sqrt.approx.ftz.f32(2)` | `0x3fb504f3` | 1.4142135381698608 |
| `sqrt.approx.ftz.f32(min_subnormal)` | `0x00000000` | subnormal input flushed |
| `rsqrt.approx.ftz.f32(2)` | `0x3f3504f3` | 0.7071067690849304 |
| fused `fma.rn.ftz(a,b,-1)` | `0xa8800000` | -1.4210854715202004e-14 |
| separate `add.ftz(mul.ftz(a,b),-1)` | `0x00000000` | zero after intermediate rounding |
| `cvt.rzi.ftz.s32.f32(-3.75)` | `0xfffffffd` | -3 |
| `cvt.rn.f32.u32(16777217)` | `0x4b800000` | 16777216.0, nearest-even |
| `add.ftz(min_subnormal, 0)` | `0x00000000` | flushed |
| `mul.ftz(min_subnormal, 1)` | `0x00000000` | flushed |
| `min.ftz(NaN, 1)` | `0x3f800000` | 1.0 |
| `max.ftz(NaN, 1)` | `0x3f800000` | 1.0 |

The FMA case uses values chosen so an intermediate-rounded multiply followed by
add becomes zero while one fused operation remains nonzero. This is an
observable contract, not merely a performance/optimization preference.

These probes do not yet cover signed-zero FTZ, infinities, both NaN operand
orders, every directed conversion boundary, or a broad approximation-error
envelope. Add cases only when a candidate backend disagrees or a launched
kernel's validation points to one of those boundaries.

## Workgroup barrier result

One 64-thread block executes:

```ptx
shared[tid] = tid + 1;
bar.sync 0;
out[tid] = shared[(tid + 1) & 63];
```

Native output is exactly:

```text
2, 3, 4, ..., 63, 64, 1
```

This proves execution convergence and visibility for the corpus's only barrier
form. It does not establish semantics for divergent barriers, barrier IDs other
than zero, thread-count operands, reductions, warp barriers, atomics, or global
memory fences; none appear in the captured corpus.

## Legacy linear texture result

The probe binds eight `uint4` elements through a module `.texref` with:

- `CU_AD_FORMAT_UNSIGNED_INT32`;
- four packed components;
- `CU_TRSF_READ_AS_INTEGER`;
- point filtering;
- address mode `CLAMP`;
- a linear device-memory range.

It runs:

```ptx
tex.1d.v4.u32.s32
```

Coordinates -2 through 9 return:

| Coordinate | Native value |
|---:|---|
| -2 | `{0,0,0,0}` |
| -1 | `{0,0,0,0}` |
| 0–7 | exact corresponding `uint4` element |
| 8 | `{0,0,0,0}` |
| 9 | `{0,0,0,0}` |

For a linear texture reference, the address-mode setter does not clamp to the
edge element. Out-of-range integer fetches return an all-zero vector. A
storage-buffer implementation therefore needs an explicit signed bounds check
and zero result. Mapping directly to a sampled image with edge clamp would be
incorrect for this observed path.

The probe currently validates raw unsigned-integer reads. The corpus also uses
`tex.1d.v4.f32.s32`; its component bits and the same bounds rule should be
verified when the first Vulkan texture lowering exists.

## Decisions enabled

### Floating point

A candidate Vulkan backend must preserve:

- f32 FTZ for the observed kernels;
- explicit FMA fusion;
- separation of non-FMA multiply/add;
- directed and nearest-even conversion behavior;
- approximate operation results sufficiently close to the native oracle.

Whether this uses SPIR-V float controls, execution modes, operation decorations,
or software helpers remains a Vulkan requirements decision.

### Barrier

The only required barrier is full-workgroup `bar.sync 0` with workgroup-memory
visibility. No subgroup or device-scope memory primitive is justified by this
probe/corpus.

### Texture

The two legacy 1D texture operations can be modeled as typed linear range
reads with explicit bounds-to-zero behavior. This strengthens the
storage-buffer candidate and weakens the sampled-image candidate.

## Remaining native probes

Still open before validation/synthesis:

1. **Real workload output:** capture or construct one deterministic fluid kernel
   input/output and one deformable kernel input/output.
2. **Float texture:** verify `tex.1d.v4.f32.s32` bit interpretation and OOB zero.
3. **Queried-only exports:** determine whether null symbols are accepted or
   whether not-supported stubs are required. ZLUDA proves stubs work.
4. **NVAPI gate:** determine the smallest interface set required by the current
   system `PhysXDevice.dll` on a non-NVIDIA rendering path.
5. **Host threading:** add thread IDs to a game trace to record call-thread
   behavior, while still implementing thread-safe state.
6. **Backend differential:** rerun this exact fixture through the first Vulkan
   translator and classify every mismatch before broadening probes.

A full game-memory replay capture is not required to begin a thin translator.
The committed fixture now supplies deterministic oracle checks for the most
backend-sensitive standalone semantics.
