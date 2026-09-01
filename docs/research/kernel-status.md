# Mirror's Edge kernel translation status

## Purpose

This is the canonical implementation ledger for the finite Mirror's Edge GPU
PhysX workload. It tracks only entry points observed launching in the captured
tutorial, subway, and server-room traces. Corpus structure and instruction
coverage remain in [`mirrors-edge-ptx-corpus.md`](mirrors-edge-ptx-corpus.md);
launch chronology remains in
[`mirrors-edge-native-trace.md`](mirrors-edge-native-trace.md).

Status meanings:

- **Exact fixture:** PTX-derived SPIR-V executes one named synthetic case through
  Vulkan and matches native RTX CUDA bit-for-bit, including guard ranges. This
  is translation evidence, not complete branch or in-game workload coverage.
- **Lowering:** frontend/emitter work is active, but no exact native/Vulkan
  fixture has passed yet.
- **Not translated:** the captured entry is cataloged but not accepted by the
  selected-entry emitter.

The `Module` column uses the stable semantic numbering from the corpus report.
`Extracted PTX` is the static extraction filename for the pinned
`PhysXCore.dll` SHA-256
`ed8a3f0cf46045699f05437348a1bd589a4e023ade5697c13dbc6c56befa545d`.

## Launched entry ledger

| Module | Extracted PTX | Entry stem | Parameters | Static shared | Dynamic shared | Status | Validation |
|---|---|---|---:|---:|---:|---|---|
| `0001` | `fatbin_0008.ptx` | `PxgCudaFluidCollisionConvexShapesBatch` | 20 B | 4,225 B | 0 | Exact fixture | Minimum-grid two-work-unit zero-convex fixture |
| `0002` | `fatbin_0005.ptx` | `PxgCudaFluidCollisionTrianglesBatch` | 20 B | 4,912 B | 0 | Exact fixture | Focused one-particle zero-triangle fixture |
| `0003` | `fatbin_0006.ptx` | `PxgCudaFluidCollisionCapsuleShapesBatch` | 20 B | 2,816 B | 0 | Exact fixture | Captured two-work-unit zero-capsule fixture |
| `0004` | `fatbin_0009.ptx` | `radixSortBlocks<32,0,false>` | 20 B | 4 B | 4,096 B | Exact fixture | Captured 64-element in-place launch shape with duplicate-key stability |
| `0005` | `fatbin_0002.ptx` | `PxgCudaFluidCollisionConstraintsBatch` | 20 B | 1,840 B | 0 | Exact fixture | Focused one-particle no-contact fixture |
| `0007` | `fatbin_0003.ptx` | `PxgCudaFluidCollisionDynamicResponseBatch` | 20 B | 304 B | 0 | Exact fixture | Focused one-particle active-constraint response |
| `0008` | `fatbin_0016.ptx` | `PxgCudaDeformableUpdate` | 12 B | 0 | 0 | Exact fixture | Focused fixture and ordered deformable step |
| `0009` | `fatbin_0004.ptx` | `PxgCudaFluidCollisionStaticResponseBatch` | 20 B | 304 B | 0 | Exact fixture | Focused one-particle active static response |
| `0010` | `fatbin_0012.ptx` | `KernelFluidDensity` | 84 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0011` | `fatbin_0015.ptx` | `KernelDeformableUberCollision` | 4 B | 4,097 B | 0 | Exact fixture | Focused fixture and ordered deformable step |
| `0012` | `fatbin_0017.ptx` | `KernelDeformableUberSolver<512>` | 16 B | 0 | 7,072 B | Exact fixture | Focused fixture and ordered deformable step |
| `0013` | `fatbin_0010.ptx` | `KernelFluidCalcHashPacked` | 52 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0013` | `fatbin_0010.ptx` | `KernelReorderPosVelAndFindCellStartPacked` | 76 B | 0 | 260 B | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0014` | `fatbin_0011.ptx` | `KernelFluidSimpleParticle` | 44 B | 10,756 B | 0 | Exact fixture | Focused one-particle velocity-integration fixture |
| `0015` | `fatbin_0007.ptx` | `PxgCudaFluidCollisionBoxShapesBatch` | 20 B | 6,752 B | 0 | Exact fixture | Captured two-work-unit zero-box fixture |
| `0016` | `fatbin_0013.ptx` | `KernelFluidForce` | 100 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |

All sixteen Mirror's Edge launched entries have exact native/Vulkan fixtures
for their named synthetic cases. Module `0006` contains only collision
constants. Module `0017` contains the four deformable self-collision stages now
translated for the `SampleCloth` scene-2 runtime path. Those stages have an
end-to-end SDK workload check but no exact native/Vulkan differential fixture,
so they remain outside this exact-fixture ledger.

PhysX 2.8.0 `SampleCloth` scene 1 is a secondary proxy for the same three
deformable entries, not an expansion of this ledger. Its solver launch uses
local size 512 and 42,240 bytes of dynamic shared memory. The storage-backed
shared-memory fixture matches native CUDA exactly on Intel and RTX, so the
sample launch is accepted even where it exceeds native Vulkan Workgroup memory.

`SampleCloth` scene 2 exercises context-global reclamation, 128 MiB
host/device heap reuse, radix sorting, four self-collision dispatches, and the
15,376-byte storage-backed solver launch. Dispatch completion alone did not
validate the self-collision result.

`SampleCloth` scene 3 creates three hardware cloths and launches the same
512-thread solver with a third dynamic-shared shape: 23,616 bytes and a
three-block grid. The runtime originally recognized only the 42,240-byte
scene-1 and 15,376-byte scene-2 shapes. It returned
`CUDA_ERROR_NOT_SUPPORTED` for the scene-3 solver launch; PhysXCore then faulted
instead of recovering from that internal launch failure. A separately emitted
storage-backed solver shader now covers the observed 23,616-byte shape.

`SampleCloth` scene 4 uses the same solver with a 128-thread block, 3,168 bytes
of dynamic shared memory, and a two-block grid. This shape fits native Vulkan
Workgroup memory, but it was absent from the runtime catalog and produced the
same `CUDA_ERROR_NOT_SUPPORTED` followed by a PhysXCore access violation. A
separately emitted Workgroup-memory shader now covers the observed shape.

`SampleCloth` scene 5 launches three 64-thread solver blocks with 2,016 bytes
of dynamic shared memory. The missing catalog shape caused every solver launch
to return `CUDA_ERROR_NOT_SUPPORTED`; update and collision continued without a
solver, producing the detached triangular cloth geometry. An exact
Workgroup-memory shader now covers the attachment scene.

The first visual scene-2 comparison exposed a runtime resource-layout defect in
`PxgCudaDeformableSelfCollisionCompareParticles`. The shader indexes texture
metadata in the source module's seven-reference order:
`ciFromSiTex`, `uiFromSiTex`, the unused `iuiFromSiTex` slot,
`invMassSiTex`, `positionSiTex`, `newPositionSiTex`, and
`originalPositionSiTex`. The runtime supplied no self-collision texture
profile. Supplying only the six referenced textures also failed because it
shifted every live float-texture index. Supplying the complete seven-slot
layout restored sustained scene-2 motion without the previously observed
self-penetrating triangular fold. This is a visual runtime result, not an exact
native/Vulkan output differential.

The earlier 800-frame native/pxvk position series was scene 1: its trace
contains only update, deformable collision, and the 42,240-byte solver launch.
It contains no self-collision dispatch and therefore cannot qualify scene 2.

### Self-collision qualification strategy

1. Treat the seven-slot `SampleCloth` resource layout as the immediate runtime
   contract; retain unused manifest slots because shader indices include them.
2. Generate a resource-layout sidecar with every deployed SPIR-V shader rather
   than extending handwritten runtime profiles. The Mirror's Edge module has a
   different six-reference layout, so a global hardcoded order is not a safe
   final design.
3. Capture one scene-2 pre-step device heap, packed launch parameters, constant
   ranges, and texture bindings. Replay the eight-dispatch sequence after each
   native and Vulkan dispatch.
4. Compare integer stages exactly in order: create hash, radix sort, reorder,
   and cell-start. Then compare the self-collision response ranges after
   `CompareParticles`, followed by collision and the iterative solver. The
   first mismatching live range owns the next focused fixture.
5. Add a long-run mesh self-intersection check over fixed 20 ms steps. Visual
   screenshots remain a smoke test; the acceptance gate is no penetrations
   where the native reference has none.

## In-game readiness audit

The exact-fixture ledger answers whether each selected entry can translate and
match native CUDA on one controlled input. It does not answer whether the
fixtures exercise the state observed in the game. Comparing the fixture sources
with the first packed launch samples from the fluid, glass, and cloth captures
shows these remaining gaps:

| Family | Captured launch evidence | Current exact fixture | Workload confidence |
|---|---|---|---|
| Fluid hash/reorder/density/force | First glass sample carries 64 particles; traced grids reach 5/5/2/3 workgroups | [`main.c`](../../tools/cuda-physx-fixture/main.c) runs one particle in one workgroup | **Low-medium:** scalar, texture, and one-cell paths match; multi-particle neighbor cells and multi-workgroup tails do not |
| Radix sort | First glass and cloth samples sort 64 elements in place: output and input key/value pointers alias | [`radix_sort.c`](../../tools/cuda-physx-fixture/radix_sort.c) now matches the 64-element aliasing contract and checks stable duplicates | **Medium-high standalone:** all rank phases and the captured alias/count shape match; placement inside a captured full fluid step does not |
| Fluid constraint/response | Two work units in the first packed samples; traced grids reach three workgroups | One no-contact constraint plus one active dynamic and one active static response | **Low-medium:** response plumbing matches; alternate flags, multiple contacts, damping, clamping, and compaction contention do not |
| Triangle/box/capsule/convex collision | Broken-glass trace reaches 94 triangle workgroups, 34 convex workgroups, and repeated box/capsule launches | [`triangles.c`](../../tools/cuda-physx-fixture/triangles.c) and [`shape_batches.c`](../../tools/cuda-physx-fixture/shape_batches.c) use zero geometry | **Low:** setup, barriers, and empty exits match; no contact geometry executes |
| Simple particle | First glass sample carries 32 particles; traced grids range from 2 to 10 | [`simple_particle.c`](../../tools/cuda-physx-fixture/simple_particle.c) integrates one particle in one workgroup and skips the velocity clamp | **Low-medium:** parameter staging and integration match; multi-fluid lookup, clamp, tails, and traced grid sizes do not |
| Deformable update/collision/solver | Cloth sample has 6 update instances, grids 28/30/6, and 5 solver iterations | Focused and ordered fixtures use one particle, zero shapes, one workgroup, and zero solver iterations | **Low:** ordering and heap plumbing match, but collision geometry and the iterative solver core do not execute |

The deformable solver gap is structural, not just a smaller data set. With the
fixture's iteration count of zero, the pinned PTX branch at `BB14_6` jumps
directly to `BB14_75`, bypassing the constraint-iteration body. The captured
subway launch passes five iterations. Likewise, every geometry fixture reports
a zero shape or triangle count, so successful execution cannot validate its
contact math.

Current conclusion: all launched entries have credible parser, lowering,
resource-layout, and branch-specific differential evidence. They are **not yet
validated for in-game PhysX state**. That claim still requires the captured
pre/post device-pool replays and three game routes defined in
[`validation-plan.md`](validation-plan.md#real-physx-kernel-fixtures).

## Exact CUDA symbols

```text
_Z39PxgCudaFluidCollisionConvexShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z36PxgCudaFluidCollisionTrianglesBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK29PxgFluidCollisionWorkUnitMeshEPi
_Z40PxgCudaFluidCollisionCapsuleShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z15radixSortBlocksILj32ELj0ELb0EEvP5uint4S1_S1_S1_j
_Z38PxgCudaFluidCollisionConstraintsBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z42PxgCudaFluidCollisionDynamicResponseBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z24PxgCudaDeformableUpdateDPKhjPi
_Z41PxgCudaFluidCollisionStaticResponseBatchD11PxgGpuArrayIK26PxgFluidWorkUnitStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z19KernelFluidDensityD17PxgFixedSizeArrayIfES0_15DensityDebugGPUPiS_IK6float4ES_IKjES7_S7_S7_
_Z29KernelDeformableUberCollisionPi
_Z26KernelDeformableUberSolverILj512EEvjbPhPi
_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj
_Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej
_Z26KernelFluidSimpleParticleD17PxgFixedSizeArrayI12PxcVectorPadES_IKS0_E11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj
_Z36PxgCudaFluidCollisionBoxShapesBatchD11PxgGpuArrayIK32PxgFluidCollisionShapeStartAddrsES_IK16PxgFluidWorkUnitEPi
_Z17KernelFluidForceD17PxgFixedSizeArrayI12PxcVectorPadE13ForceDebugGPUPiS_IK6float4ES6_S_IKfES_IKjESA_SA_SA_
```

## Update rule

Change a row to **Exact fixture** only after all of these pass:

1. the pinned extracted PTX parses and emits deterministic SPIR-V;
2. `spirv-val` accepts the module for Vulkan 1.2;
3. a focused native CUDA fixture defines exact output bits and guard ranges;
4. the same fixture passes through Vulkan on the available Intel and NVIDIA
   adapters;
5. the portable validation suite includes the fixture and reference report.
