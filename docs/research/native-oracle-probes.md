# Native NVIDIA oracle probes

## Status

A project-owned x86 CUDA Driver API probe now records exact NVIDIA results for
the PTX floating-point, workgroup-barrier, invocation-private local-memory,
module constant-memory, and legacy integer/float linear-texture semantics that
are most backend-sensitive in the Mirror's Edge corpus.

Implemented by `tools/cuda-oracle/main.c` and `probe.ptx`. The runner emits a
standalone JSON report; `tools/build-reference-bundle.py` stores it at
`cuda.semantic_oracle` in
`tools/reference/expected/reference-workstation.json`.

The original native report had SHA-256
`38fc7ef4be8863cc55b0a5408cbeab5e20f6f81de852a9b7818339897cc88011`.
Nine native runner reports are stored in the deterministic reference bundle:
the semantic oracle, four-kernel fluid pipeline, focused fluid constraints,
dynamic response, and triangles, all three focused deformable workloads, and
the ordered deformable step. The bundle has SHA-256
`bb98605c68bd748d9179117d3cfedc6472eaf34956b9f78aee5c4af55ed990ac`.

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

The output is supplied to `tools/build-reference-bundle.py --cuda-oracle`.

The focused fluid-constraints report uses the extracted constraints module:

```powershell
.\build\cuda-physx-fixture\Release\cuda-fluid-constraints-fixture.exe `
  .\local-ptx\fatbin_0002.ptx `
  .\native-fluid-constraints.json
```

It launches `PxgCudaFluidCollisionConstraintsBatch` for one synthetic particle
with the no-contact key `0x0000ffff`, time step `0.5f`, position `[1,2,3]`, and
motion `[4,5,6]`. The exact outputs are position `[3,4.5,6]`, velocity
`[4,5,6]`, unit weight, eight zero vectors, and four zero scalars. All guarded
ranges remain intact; three reports are byte-identical. The native report
SHA-256 is
`28c23397e099c4dc2db893253fbc292c733e0f414928d9812c5b29fc94d6e7eb`;
the report is supplied as `--cuda-fluid-constraints`.

The focused dynamic-response report uses the next trace-ordered module:

```powershell
.\build\cuda-physx-fixture\Release\cuda-fluid-dynamic-response-fixture.exe `
  .\local-ptx\fatbin_0003.ptx `
  .\native-fluid-dynamic-response.json
```

It launches `PxgCudaFluidCollisionDynamicResponseBatch` for one active
constraint with unit X normal, accumulated position `[7,8,9]`, count one,
zero body velocity, particle velocity `[-2,3,0]`, and impulse scale `0.5f`.
The kernel clears the consumed normal and accumulators, writes averaged
position `[7,8,9]`, velocity `[1,3,0]`, response `[-3,0,0]`, unit weight,
the copied destination ID, and cleared constraint flags. All guards remain
intact; three native reports are byte-identical. The native report SHA-256 is
`aadb855952a6fc76c9ac148841aa50e3366511eb9d08f8b7ac15269ed6e113f0`;
the report is supplied as `--cuda-fluid-dynamic-response`.

The focused triangle-collision report uses the extracted triangle module:

```powershell
.\build\cuda-physx-fixture\Release\cuda-fluid-triangles-fixture.exe `
  .\local-ptx\fatbin_0005.ptx `
  .\native-fluid-triangles.json
```

It launches `PxgCudaFluidCollisionTrianglesBatch` with one particle and a mesh
work unit containing zero triangles. The captured 64-thread launch crosses the
three shared-memory setup barriers and takes the empty-mesh branch. Position,
old position, and the two particle state words are preserved exactly, and all
guards remain intact. The native report SHA-256 is
`a9c53cf29be39d08e4eecb6096280be43672cfbc5a218ca906b1e9ac56ad82e9`;
the report is supplied as `--cuda-fluid-triangles`. This focused fixture does
not cover a non-empty triangle-contact record.

The focused deformable report uses the extracted PhysX module on the native
NVIDIA reference system:

```powershell
cmake -S tools/cuda-physx-fixture -B build/cuda-physx-fixture -A Win32
cmake --build build/cuda-physx-fixture --config Release
.\build\cuda-physx-fixture\Release\cuda-deformable-fixture.exe `
  .\local-ptx\fatbin_0016.ptx `
  .\native-deformable.json
```

The runner launches `PxgCudaDeformableUpdate` with one synthetic instance,
checks exact output bits and guard ranges, and supplies its report to
`tools/build-reference-bundle.py --cuda-deformable`.

The solver fixture uses the extracted solver module and its captured 7,072-byte
dynamic-shared launch shape:

```powershell
.\build\cuda-physx-fixture\Release\cuda-deformable-solver-fixture.exe `
  .\local-ptx\fatbin_0017.ptx `
  .\native-deformable-solver.json
```

It launches `KernelDeformableUberSolver<512>` for one synthetic particle with
zero constraint iterations. The entry still crosses its workgroup barriers,
updates the particle/delta state, and writes the exact 64-byte solver record.
Every live range is guarded. The native report SHA-256 is
`83d4b38dd4112716e63311a06b7d040e9275c44654fd1bae6dbdefb33009c040`;
the report is supplied as `--cuda-deformable-solver`.

The collision fixture uses the extracted collision module:

```powershell
.\build\cuda-physx-fixture\Release\cuda-deformable-collision-fixture.exe `
  .\local-ptx\fatbin_0015.ptx `
  .\native-deformable-collision.json
```

It launches `KernelDeformableUberCollision` for one synthetic particle and no
shapes. The focused path copies the two particle vectors, clears collision
vectors/state, crosses all five barriers, and reaches response processing. All
17 live heap ranges are guarded. The exact collision scalar words are
`[0xbf800000, 0x00000000, 0x0000ffff, 0x00000000]`; three native runs were
byte-identical. The native report SHA-256 is
`0413b0348b49fe1b69376e6ed049f7011c2781366e3311ade51b5daaa0287369`;
the report is supplied as `--cuda-deformable-collision`.

The sequence fixture launches all three modules on one CUDA default stream with
one shared 32-bit heap and no intermediate host synchronization:

```powershell
.\build\cuda-physx-fixture\Release\cuda-deformable-sequence-fixture.exe `
  .\local-ptx\fatbin_0016.ptx `
  .\local-ptx\fatbin_0015.ptx `
  .\local-ptx\fatbin_0017.ptx `
  .\native-deformable-sequence.json
```

Collision consumes update's position/motion ranges; solver consumes collision's
position output. The trace-shaped one-particle path preserves the update motion,
collision scalars, solver delta, and 64-byte solver record bit-for-bit, with all
25 live heap ranges guarded. It intentionally uses zero shapes and zero solver
iterations, so it proves ordering and inter-kernel state flow rather than real
geometry or constraint response. The native report SHA-256 is
`4c71f0c3f14d0c9456c38a15fdf04a04271568cc77acdcc211a78e2d560ba2a8`;
the report is supplied as `--cuda-deformable-sequence`.

## Probe questions and decisions

| Question | Native probe | Result | Contract consequence |
|---|---|---|---|
| Are f32 subnormal inputs/outputs flushed for `.ftz` arithmetic? | Divide minimum normal by two; operate on minimum subnormal | All selected subnormal results became zero | Initial backend must enable FTZ or emulate it |
| Is FMA genuinely fused? | Compare `fma.rn.ftz` with separate `mul.ftz` + `add.ftz` | FMA returned nonzero; separate path returned zero | Do not permit contraction policy to change explicit FMA/separate operations |
| What are native approximate math bits? | `rcp`, `sqrt`, and `rsqrt` on fixed inputs | Exact values recorded below | Vulkan results must be compared bitwise or against an explicitly approved tolerance |
| Does `bar.sync 0` make all shared writes visible? | 64 threads write, barrier, read neighbor | Exact cyclic neighbor sequence | Lower to full-workgroup execution + workgroup-memory barrier |
| How does an integer-coordinate linear texref handle OOB? | Coordinates -2 through 9 over eight `uint4` elements | In-range exact; negative and high OOB return zero | Storage-buffer lowering needs explicit bounds-to-zero, not edge clamping |
| Does a float texref preserve fetched bits and OOB zero? | Normal values, signed zero, subnormals, infinities, NaN payloads, and signed coordinates -2 through 9 | Every in-range component was bit-identical; negative/high OOB returned four positive zeros | Lower `tex.1d.v4.f32.s32` as bounded four-word loads plus bit reinterpretation |
| Do indirect `.const` loads preserve integer and float payload bits? | Copy eight words into a 32-byte module constant, form an address through `mov` + `add`, then load `u32` and `f32` | Returned `0x1234abcd` and NaN payload `0x7fc12345` exactly | Represent module constants as a read-only bound byte range with 32-bit space-relative addressing |
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

The float texture probe covers the same signed coordinate range with signed
zero, subnormals, infinities, NaNs, and ordinary values. Every in-range
component preserves its original 32-bit payload; negative and high coordinates
return four positive-zero components.

## Invocation-private local result

The 64-thread local probe declares one `.u32` at offset 0 and a second
16-byte-aligned `.u32` at offset 16. Every invocation stores and reloads only
the aligned symbol before writing `0x5a000000 + tid` to global output. Native
CUDA returns the complete `0x5a000000` through `0x5a00003f` sequence. This
establishes the concrete private-storage and alignment contract needed by
`KernelFluidDensity`'s 108-byte local depot.

## Module constant result

The host resolves the 32-byte `probe_constants` symbol with
`cuModuleGetGlobal_v2`, copies eight words into it, and launches a kernel that
forms an indirect address from the symbol base. Native `ld.const.u32` returns
`0x1234abcd`; `ld.const.f32` preserves the `0x7fc12345` NaN payload. This is the
same symbol-base plus computed-offset shape used by `KernelFluidDensity`.

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

### Local storage

Routine `.local` declarations require an aligned, invocation-private arena.
They do not justify a descriptor, workgroup allocation, or cross-invocation
synchronization.

### Constant storage

Module `.const` declarations require immutable-per-dispatch storage with
module-relative 32-bit byte offsets. The Vulkan lowering uses read-only storage
buffer binding 2; this does not place constants in the CUDA heap or launch
parameter buffer.

### Texture

The two legacy 1D texture operations can be modeled as typed linear range
reads with explicit bounds-to-zero behavior. This strengthens the
storage-buffer candidate and weakens the sampled-image candidate.

## Vulkan differential result

The PTX-derived PE32 Vulkan oracle reproduces every committed field bit-for-bit
on Intel UHD and RTX 3070 Vulkan:

- all 14 f32/conversion results;
- the 64-thread shared-memory barrier sequence;
- the 64-thread invocation-private local-memory sequence;
- all 12 integer texture rows including signed OOB zero;
- all 12 float texture rows, including signed zero, subnormals, infinities, and
  NaN payload preservation.
- the indirect module-constant integer and float payloads.

Intel initially returned zero for the fused FMA because GLSL.std.450 `Fma`
without `NoContraction` may be split. Decorating the FMA result explicitly
restored native `0xa8800000`; the separate multiply/add remained zero on both
devices. Native CUDA evidence is stored at `cuda.semantic_oracle`; the
PTX-generated Vulkan result is stored at `vulkan.semantic_oracle_ptx` in
`tools/reference/expected/reference-workstation.json`.

## Remaining native probes

Still open at their later acceptance gates:

1. **Captured workload output:** focused synthetic fluid and deformable
   differentials are complete; captured in-game state and complete steps remain.
2. **Queried-only exports:** determine whether null symbols are accepted or
   whether not-supported stubs are required. ZLUDA proves stubs work.
3. **NVAPI gate:** determine the smallest interface set required by the current
   system `PhysXDevice.dll` on a non-NVIDIA rendering path.
4. **Host threading:** add thread IDs to a game trace to record call-thread
   behavior, while still implementing thread-safe state.

A full game-memory replay capture is not required to begin a thin translator.
The committed fixture now supplies deterministic oracle checks for the most
backend-sensitive standalone semantics.
