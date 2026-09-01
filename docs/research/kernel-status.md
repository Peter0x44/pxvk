# Mirror's Edge kernel translation status

## Purpose

This is the canonical implementation ledger for the finite Mirror's Edge GPU
PhysX workload. It tracks only entry points observed launching in the captured
tutorial, subway, and server-room traces. Corpus structure and instruction
coverage remain in [`mirrors-edge-ptx-corpus.md`](mirrors-edge-ptx-corpus.md);
launch chronology remains in
[`mirrors-edge-native-trace.md`](mirrors-edge-native-trace.md).

Status meanings:

- **Exact fixture:** PTX-derived SPIR-V executes through Vulkan and matches a
  native RTX CUDA fixture bit-for-bit, including guard ranges.
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
| `0004` | `fatbin_0009.ptx` | `radixSortBlocks<32,0,false>` | 20 B | 4 B | 4,096 B | Not translated | — |
| `0005` | `fatbin_0002.ptx` | `PxgCudaFluidCollisionConstraintsBatch` | 20 B | 1,840 B | 0 | Exact fixture | Focused one-particle no-contact fixture |
| `0007` | `fatbin_0003.ptx` | `PxgCudaFluidCollisionDynamicResponseBatch` | 20 B | 304 B | 0 | Exact fixture | Focused one-particle active-constraint response |
| `0008` | `fatbin_0016.ptx` | `PxgCudaDeformableUpdate` | 12 B | 0 | 0 | Exact fixture | Focused fixture and ordered deformable step |
| `0009` | `fatbin_0004.ptx` | `PxgCudaFluidCollisionStaticResponseBatch` | 20 B | 304 B | 0 | Exact fixture | Focused one-particle active static response |
| `0010` | `fatbin_0012.ptx` | `KernelFluidDensity` | 84 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0011` | `fatbin_0015.ptx` | `KernelDeformableUberCollision` | 4 B | 4,097 B | 0 | Exact fixture | Focused fixture and ordered deformable step |
| `0012` | `fatbin_0017.ptx` | `KernelDeformableUberSolver<512>` | 16 B | 0 | 7,072 B | Exact fixture | Focused fixture and ordered deformable step |
| `0013` | `fatbin_0010.ptx` | `KernelFluidCalcHashPacked` | 52 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0013` | `fatbin_0010.ptx` | `KernelReorderPosVelAndFindCellStartPacked` | 76 B | 0 | 260 B | Exact fixture | Focused fixture and four-kernel fluid fixture |
| `0014` | `fatbin_0011.ptx` | `KernelFluidSimpleParticle` | 44 B | 10,756 B | 0 | Not translated | — |
| `0015` | `fatbin_0007.ptx` | `PxgCudaFluidCollisionBoxShapesBatch` | 20 B | 6,752 B | 0 | Exact fixture | Captured two-work-unit zero-box fixture |
| `0016` | `fatbin_0013.ptx` | `KernelFluidForce` | 100 B | 0 | 0 | Exact fixture | Focused fixture and four-kernel fluid fixture |

Fourteen of sixteen launched entries have exact native/Vulkan fixtures. Two
remain. Module `0006` contains only collision constants. Module `0017` contains
resolved deformable self-collision entries, but none launched in the captured
routes, so they are outside this ledger and the initial compatibility target.

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
