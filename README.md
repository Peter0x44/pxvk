# Classic GPU PhysX compatibility

`pxvk` is a focused compatibility project for running classic 32-bit NVIDIA
GPU PhysX 2.8.x-era CUDA workloads through Vulkan compute. Mirror's Edge
(2009) is the first target; this is not a general CUDA-on-Vulkan runtime.

The current finite PTX 4.1 frontend parses all 17 extracted Mirror's Edge
modules. Direct SPIR-V lowering and exact native/Vulkan differential fixtures
currently cover the project oracle, `KernelFluidCalcHashPacked`,
`KernelReorderPosVelAndFindCellStartPacked`, `KernelFluidDensity`, and
`KernelFluidForce`.

See [`docs/charter.md`](docs/charter.md) for scope and
[`docs/research/README.md`](docs/research/README.md) for the evidence index.

## Run the portable suite on AMD

Requirements:

- Windows with a 32-bit Vulkan loader and AMD x86 Vulkan driver installed;
- PowerShell, CMake, Git, and either an i686 MinGW toolchain or Visual Studio
  with Win32 C++ tools;
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

Captured game PTX is proprietary and is not committed. If the three locally
extracted files are available in one directory, add the four trace-shaped
fluid fixtures with:

```powershell
.\run-tests.ps1 -CapturedPtxDir C:\path\to\extracted\ptx
```

That directory must contain `module_0013_01_05.ptx`,
`module_0010_01_05.ptx`, and `module_0016_01_05.ptx`. The added run compiles
and executes calc-hash, reorder, density, and force on the AMD Vulkan device.
Any failed device or exact result returns a nonzero process exit code.
