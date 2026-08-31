# Classic GPU PhysX compatibility

`pxvk` is a focused compatibility project for running classic 32-bit NVIDIA
GPU PhysX 2.8.x-era CUDA workloads through Vulkan compute. Mirror's Edge
(2009) is the first target; this is not a general CUDA-on-Vulkan runtime.

The current finite PTX 4.1 frontend parses all 17 extracted Mirror's Edge
modules. Direct SPIR-V lowering covers the project oracle,
`KernelFluidCalcHashPacked`, `KernelReorderPosVelAndFindCellStartPacked`,
`KernelFluidDensity`, `KernelFluidForce`, `PxgCudaDeformableUpdate`,
`KernelDeformableUberCollision`, and `KernelDeformableUberSolver<512>`. Exact
native/Vulkan differential fixtures cover the oracle, all four translated fluid
kernels, and all three translated deformable kernels.

See [`docs/charter.md`](docs/charter.md) for scope and
[`docs/research/README.md`](docs/research/README.md) for the evidence index.

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
plus a source/module hash manifest. Keep that output local. Add the four
trace-shaped fluid fixtures and the deformable update/collision/solver fixtures
by passing either this directory or a ZLUDA trace module directory:

```powershell
.\run-tests.ps1 -CapturedPtxDir .\local-ptx
```

The suite resolves modules by their PTX entry names rather than trace-specific
module numbers. It compiles, validates, and executes calc-hash, reorder,
density, force, deformable-update, deformable-collision, and deformable-solver.
Any failed device, shader validation, guard range, or exact result returns a
nonzero process exit code.

## Reference results

Each runner still writes a focused JSON report for failure diagnosis. The
committed reference is one deterministic bundle:
`tools/reference/expected/reference-workstation.json`. It contains five native
CUDA reports and all thirteen Vulkan reports under stable keys.

Compare a complete local run against it:

```powershell
.\run-tests.ps1 -CapturedPtxDir .\local-ptx `
  -ReferenceBundle .\tools\reference\expected\reference-workstation.json
```

After classifying a deliberate result change, the same command with
`-AcceptReferenceBundle` replaces only the Vulkan section. To assemble a new
bundle or replace native CUDA evidence, generate all five native runner JSON
files and run:

```powershell
python .\tools\build-reference-bundle.py `
  --vulkan-results .\test-output\results `
  --cuda-oracle .\native-oracle.json `
  --cuda-physx .\native-physx.json `
  --cuda-deformable .\native-deformable.json `
  --cuda-deformable-solver .\native-deformable-solver.json `
  --cuda-deformable-collision .\native-deformable-collision.json `
  --output .\tools\reference\expected\reference-workstation.json
```
