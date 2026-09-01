# Classic GPU PhysX compatibility

`pxvk` is a focused compatibility project for running classic 32-bit NVIDIA
GPU PhysX 2.8.x-era CUDA workloads through Vulkan compute. Mirror's Edge
(2009) is the first target; this is not a general CUDA-on-Vulkan runtime.

The current finite PTX 4.1 frontend parses all 17 extracted Mirror's Edge
modules. Direct SPIR-V lowering covers the project oracle,
`KernelFluidCalcHashPacked`, `KernelReorderPosVelAndFindCellStartPacked`,
`KernelFluidDensity`, `KernelFluidForce`,
`PxgCudaFluidCollisionConstraintsBatch`,
`PxgCudaFluidCollisionDynamicResponseBatch`,
`PxgCudaFluidCollisionStaticResponseBatch`,
`PxgCudaFluidCollisionBoxShapesBatch`,
`PxgCudaFluidCollisionCapsuleShapesBatch`,
`PxgCudaFluidCollisionConvexShapesBatch`,
`PxgCudaFluidCollisionTrianglesBatch`, `KernelFluidSimpleParticle`,
`radixSortBlocks<32,0,false>`, `PxgCudaDeformableUpdate`,
`KernelDeformableUberCollision`, and `KernelDeformableUberSolver<512>`. Exact
native/Vulkan differential fixtures cover the oracle, all twelve translated
fluid kernels, the translated radix sort kernel, all three translated
deformable kernels, the ordered deformable update/collision/solver step, and
the PhysX 2.8.0 `SampleCloth` scene-1 solver launch shape. Oversized CUDA shared
memory uses per-workgroup storage-buffer scratch instead of rejecting a valid
launch against one Vulkan device's native Workgroup-memory limit.

See [`docs/charter.md`](docs/charter.md) for scope and
[`docs/research/README.md`](docs/research/README.md) for the evidence index.

The 32-bit [`pxvk-runtime`](tools/pxvk-runtime/README.md) now provides the
in-process `nvcuda.dll` facade: the captured CUDA export surface, six private
CUDART/PhysX export tables, 32-bit virtual allocations, module
globals/textures, packed launches, and Vulkan dispatch for the finite
translated kernel set.

## Build the DLL and stage SampleCloth

Given a local PhysX 2.8 `SampleCloth` directory and extracted PTX, one command
builds the x86 runtime and frontend, runs the runtime contract smoke test,
translates the three deformable shaders, and creates an isolated runnable
stage:

```powershell
.\tools\pxvk-runtime\prepare-sample.ps1 `
  -SampleDirectory .\local-samples\SampleCloth `
  -CapturedPtxDirectory .\local-ptx `
  -OutputDirectory .\local-run\SampleCloth
```

Run `run-pxvk-samplecloth.cmd` in the resulting application directory. The
launcher selects the staged `spirv` directory and writes `pxvk-runtime.log`
beside the sample. The original sample and proprietary PTX remain unchanged.

## Run the portable suite on AMD

Requirements:

- Windows with a 32-bit Vulkan loader and AMD x86 Vulkan driver installed;
- PowerShell, Python 3.9 or newer, CMake, Git, and either an i686 MinGW
  toolchain or Visual Studio with Win32 C++ tools;
- network access on the first configure so CMake can fetch pinned dependencies.

The Vulkan path has no CUDA or NVIDIA dependency. From the repository root,
the default i686 MinGW run is one command:

```powershell
.\run-tests.ps1
```

Visual Studio is also supported:

```powershell
.\run-tests.ps1 -Toolchain MSVC
```

The script builds the frontend and 32-bit Vulkan tools, runs all CTest frontend
and emitter checks, probes every visible Vulkan device, allocates and verifies
the 32 MiB heap path, and executes both the independent and directly generated
semantic oracles plus constant and integer/float texture fixtures. Results are
written under `test-output/results/`. `spirv-val.exe` is used when it is on
`PATH`; Vulkan execution remains mandatory either way.

Captured game PTX is proprietary and is not committed. The observed kernels are
embedded in the registered PhysX 2.8.0 `PhysXCore.dll`, not
`MirrorsEdge.exe`. Extract all 17 modules offline:

```powershell
python .\extract-physx-ptx.py `
  "C:\Program Files (x86)\NVIDIA Corporation\PhysX\Engine\v2.8.0\PhysXCore.dll" `
  .\local-ptx
```

The extractor needs no CUDA toolkit. It emits stable `fatbin_####.ptx` files
plus a source/module hash manifest. Keep that output local. Add the twelve
trace-shaped fluid fixtures, the focused deformable fixtures, the ordered
three-kernel deformable step, and the `SampleCloth` solver proxy by passing
either this directory or a ZLUDA trace module directory:

```powershell
.\run-tests.ps1 -CapturedPtxDir .\local-ptx
```

The suite resolves modules by their PTX entry names rather than trace-specific
module numbers. It compiles, validates, and executes calc-hash, reorder,
density, force, fluid collision constraints, dynamic response, static response,
boxes, capsules, convexes, triangles, simple particle, radix sort blocks, the
three focused deformable kernels, the ordered deformable step, and the
`SampleCloth` local-size-512/42,240-byte solver shape through storage-backed
shared memory.
Any failed device, shader validation, guard range, or exact result returns a
nonzero process exit code.

## Reference results

Each runner still writes a focused JSON report for failure diagnosis. The
committed reference is one deterministic bundle:
`tools/reference/expected/reference-workstation.json`. It contains sixteen
native CUDA reports and all twenty-four Vulkan reports under stable keys.

Compare a complete local run against it:

```powershell
.\run-tests.ps1 -CapturedPtxDir .\local-ptx `
  -ReferenceBundle .\tools\reference\expected\reference-workstation.json
```

After classifying a deliberate result change, the same command with
`-AcceptReferenceBundle` replaces only the Vulkan section. To assemble a new
bundle or replace native CUDA evidence, generate all sixteen native runner JSON
files and run:

```powershell
python .\tools\build-reference-bundle.py `
  --vulkan-results .\test-output\results `
  --cuda-oracle .\native-oracle.json `
  --cuda-physx .\native-physx.json `
  --cuda-fluid-constraints .\native-fluid-constraints.json `
  --cuda-fluid-dynamic-response .\native-fluid-dynamic-response.json `
  --cuda-fluid-static-response .\native-fluid-static-response.json `
  --cuda-fluid-simple-particle .\native-fluid-simple-particle.json `
  --cuda-fluid-boxes .\native-fluid-boxes.json `
  --cuda-fluid-capsules .\native-fluid-capsules.json `
  --cuda-fluid-convexes .\native-fluid-convexes.json `
  --cuda-fluid-triangles .\native-fluid-triangles.json `
  --cuda-radix-sort .\native-radix-sort.json `
  --cuda-deformable .\native-deformable.json `
  --cuda-deformable-solver .\native-deformable-solver.json `
  --cuda-sample-cloth-solver .\native-sample-cloth-solver.json `
  --cuda-deformable-collision .\native-deformable-collision.json `
  --cuda-deformable-sequence .\native-deformable-sequence.json `
  --output .\tools\reference\expected\reference-workstation.json
```
