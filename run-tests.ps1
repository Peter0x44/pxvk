[CmdletBinding()]
param(
    [ValidateSet("MinGW", "MSVC")]
    [string]$Toolchain = "MinGW",
    [string]$OutputDir = "",
    [string]$CapturedPtxDir = "",
    [string]$ReferenceBundle = "",
    [switch]$AcceptReferenceBundle,
    [string]$Generator = "",
    [string]$CCompiler = "i686-w64-mingw32-gcc",
    [string]$CxxCompiler = "i686-w64-mingw32-g++",
    [string]$MakeProgram = "mingw32-make"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$File,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )

    & $File @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$File failed with exit code $LASTEXITCODE"
    }
}

function Configure-X86Build {
    param(
        [Parameter(Mandatory = $true)][string]$Source,
        [Parameter(Mandatory = $true)][string]$Build
    )

    $arguments = @("-S", $Source, "-B", $Build)
    if ($Toolchain -eq "MinGW") {
        $selectedGenerator = $Generator
        if ([string]::IsNullOrWhiteSpace($selectedGenerator)) {
            $selectedGenerator = "MinGW Makefiles"
        }
        $arguments += @(
            "-G", $selectedGenerator,
            "-DCMAKE_C_COMPILER=$CCompiler",
            "-DCMAKE_CXX_COMPILER=$CxxCompiler",
            "-DCMAKE_MAKE_PROGRAM=$MakeProgram",
            "-DCMAKE_BUILD_TYPE=Release"
        )
    }
    else {
        if ([string]::IsNullOrWhiteSpace($Generator)) {
            $arguments += @("-A", "Win32")
        }
        else {
            $arguments += @("-G", $Generator, "-A", "Win32")
        }
    }

    Invoke-Checked "cmake" $arguments
    Invoke-Checked "cmake" @("--build", $Build, "--config", "Release", "--parallel")
}

function Resolve-BuiltExecutable {
    param(
        [Parameter(Mandatory = $true)][string]$Build,
        [Parameter(Mandatory = $true)][string]$Name
    )

    $candidates = @(
        (Join-Path $Build "$Name.exe"),
        (Join-Path $Build "Release\$Name.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }
    throw "Built executable '$Name.exe' was not found under $Build"
}

function Validate-SpirvIfAvailable {
    param([Parameter(Mandatory = $true)][string[]]$Paths)

    $validator = Get-Command "spirv-val.exe" -ErrorAction SilentlyContinue
    if ($null -eq $validator) {
        Write-Host "spirv-val.exe not found; Vulkan execution remains the shader validation gate."
        return
    }
    foreach ($path in $Paths) {
        Invoke-Checked $validator.Source @("--target-env", "vulkan1.2", $path)
    }
}

function Resolve-PtxForEntry {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Entry
    )

    $matches = @(
        Get-ChildItem -LiteralPath $Directory -Filter "*.ptx" -File |
            Where-Object {
                [System.IO.File]::ReadAllText($_.FullName).Contains($Entry)
            }
    )
    if ($matches.Count -ne 1) {
        throw ("Expected one PTX file containing '{0}' under {1}; found {2}" -f
            $Entry, $Directory, $matches.Count)
    }
    return $matches[0].FullName
}

$repoDir = [System.IO.Path]::GetFullPath($PSScriptRoot)
if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path $repoDir "test-output"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$toolchainName = $Toolchain.ToLowerInvariant()
$frontendBuild = Join-Path $OutputDir "build-ptx-frontend-$toolchainName"
$vulkanBuild = Join-Path $OutputDir "build-vulkan-probe-$toolchainName"
$resultDir = Join-Path $OutputDir "results"
New-Item -ItemType Directory -Force -Path $resultDir | Out-Null

Write-Host "Building and testing the finite PTX frontend..."
Configure-X86Build (Join-Path $repoDir "tools\ptx-frontend") $frontendBuild
Invoke-Checked "ctest" @(
    "--test-dir", $frontendBuild,
    "-C", "Release",
    "--output-on-failure"
)

Write-Host "Building and preflighting the 32-bit Vulkan probes..."
$preflightParameters = @{
    Toolchain = $Toolchain
    SourceDir = (Join-Path $repoDir "tools\vulkan-probe")
    BuildDir = $vulkanBuild
    OutputDir = $resultDir
    CCompiler = $CCompiler
    CxxCompiler = $CxxCompiler
    MakeProgram = $MakeProgram
    RunMemorySmoke = $true
    RunOracle = $true
}
if (-not [string]::IsNullOrWhiteSpace($Generator)) {
    $preflightParameters.Generator = $Generator
}
& (Join-Path $repoDir "tools\vulkan-probe\preflight.ps1") @preflightParameters
if ($LASTEXITCODE -ne 0) {
    throw "Vulkan hardware preflight failed with exit code $LASTEXITCODE"
}

$frontend = Resolve-BuiltExecutable $frontendBuild "ptx-frontend"
$emitter = Resolve-BuiltExecutable $frontendBuild "ptx-spirv-emitter"
$memorySmoke = Resolve-BuiltExecutable $vulkanBuild "vulkan-memory-smoke"

$generatedOracle = Join-Path $frontendBuild "oracle-explicit-rte.spv"
$constantKernel = Join-Path $frontendBuild "constant-kernel.spv"
$textureKernel = Join-Path $frontendBuild "texture-kernel.spv"
$textureFloatKernel = Join-Path $frontendBuild "texture-float-kernel.spv"
Validate-SpirvIfAvailable @(
    $generatedOracle,
    $constantKernel,
    $textureKernel,
    $textureFloatKernel
)

Invoke-Checked $memorySmoke @(
    (Join-Path $resultDir "generated-oracle.json"),
    "--ptx-oracle",
    $generatedOracle
)
Invoke-Checked $memorySmoke @(
    (Join-Path $resultDir "constant-kernel.json"),
    "--constant-fixture",
    $constantKernel
)
Invoke-Checked $memorySmoke @(
    (Join-Path $resultDir "texture-kernel.json"),
    "--texture-fixture",
    $textureKernel
)
Invoke-Checked $memorySmoke @(
    (Join-Path $resultDir "texture-float-kernel.json"),
    "--texture-float-fixture",
    $textureFloatKernel
)

if (-not [string]::IsNullOrWhiteSpace($CapturedPtxDir)) {
    $CapturedPtxDir = [System.IO.Path]::GetFullPath($CapturedPtxDir)
    if (-not (Test-Path -LiteralPath $CapturedPtxDir -PathType Container)) {
        throw "Captured or extracted PTX directory not found: $CapturedPtxDir"
    }

    $calcHashEntry = "_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj"
    $reorderEntry = "_Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej"
    $densityEntry = "_Z19KernelFluidDensityD17PxgFixedSizeArrayIfES0_15DensityDebugGPUPiS_IK6float4ES_IKjES7_S7_S7_"
    $forceEntry = "_Z17KernelFluidForceD17PxgFixedSizeArrayI12PxcVectorPadE13ForceDebugGPUPiS_IK6float4ES6_S_IKfES_IKjESA_SA_SA_"
    $deformableUpdateEntry = "_Z24PxgCudaDeformableUpdateDPKhjPi"
    $deformableSolverEntry = "_Z26KernelDeformableUberSolverILj512EEvjbPhPi"
    $deformableCollisionEntry = "_Z29KernelDeformableUberCollisionPi"

    $calcHashPtx = Resolve-PtxForEntry $CapturedPtxDir $calcHashEntry
    $reorderPtx = Resolve-PtxForEntry $CapturedPtxDir $reorderEntry
    $densityPtx = Resolve-PtxForEntry $CapturedPtxDir $densityEntry
    $forcePtx = Resolve-PtxForEntry $CapturedPtxDir $forceEntry
    $deformableUpdatePtx = Resolve-PtxForEntry $CapturedPtxDir $deformableUpdateEntry
    $deformableSolverPtx = Resolve-PtxForEntry $CapturedPtxDir $deformableSolverEntry
    $deformableCollisionPtx = Resolve-PtxForEntry $CapturedPtxDir $deformableCollisionEntry

    Write-Host "Compiling local captured PhysX kernels and running available fixtures..."
    $fluidHashReorderManifest = Join-Path $resultDir "fluid-hash-reorder-manifest.json"
    $fluidDensityManifest = Join-Path $resultDir "fluid-density-manifest.json"
    $fluidForceManifest = Join-Path $resultDir "fluid-force-manifest.json"
    $deformableUpdateManifest = Join-Path $resultDir "deformable-update-manifest.json"
    $deformableSolverManifest = Join-Path $resultDir "deformable-solver-manifest.json"
    $deformableCollisionManifest = Join-Path $resultDir "deformable-collision-manifest.json"
    $calcHashSpirv = Join-Path $resultDir "fluid-calc-hash.spv"
    $reorderSpirv = Join-Path $resultDir "fluid-reorder.spv"
    $densitySpirv = Join-Path $resultDir "fluid-density.spv"
    $forceSpirv = Join-Path $resultDir "fluid-force.spv"
    $deformableUpdateSpirv = Join-Path $resultDir "deformable-update.spv"
    $deformableSolverSpirv = Join-Path $resultDir "deformable-solver.spv"
    $deformableCollisionSpirv = Join-Path $resultDir "deformable-collision.spv"

    Invoke-Checked $frontend @($calcHashPtx, $fluidHashReorderManifest)
    if ($reorderPtx -ne $calcHashPtx) {
        throw "Calc-hash and reorder entries must reside in one PTX module"
    }
    Invoke-Checked $frontend @($densityPtx, $fluidDensityManifest)
    Invoke-Checked $frontend @($forcePtx, $fluidForceManifest)
    Invoke-Checked $frontend @($deformableUpdatePtx, $deformableUpdateManifest)
    Invoke-Checked $frontend @($deformableSolverPtx, $deformableSolverManifest)
    Invoke-Checked $frontend @($deformableCollisionPtx, $deformableCollisionManifest)
    Invoke-Checked $emitter @(
        "--entry", $calcHashEntry, "64", "0", $fluidHashReorderManifest, $calcHashSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $reorderEntry, "64", "260", $fluidHashReorderManifest, $reorderSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $densityEntry, "192", "0", $fluidDensityManifest, $densitySpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $forceEntry, "128", "0", $fluidForceManifest, $forceSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $deformableUpdateEntry, "64", "0",
        $deformableUpdateManifest, $deformableUpdateSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $deformableSolverEntry, "64", "7072",
        $deformableSolverManifest, $deformableSolverSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $deformableCollisionEntry, "64", "0",
        $deformableCollisionManifest, $deformableCollisionSpirv
    )
    Validate-SpirvIfAvailable @(
        $calcHashSpirv,
        $reorderSpirv,
        $densitySpirv,
        $forceSpirv,
        $deformableUpdateSpirv,
        $deformableSolverSpirv,
        $deformableCollisionSpirv
    )

    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "fluid-calc-hash.json"),
        "--kernel-fixture",
        $calcHashSpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "fluid-reorder.json"),
        "--reorder-fixture",
        $reorderSpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "fluid-density.json"),
        "--density-fixture",
        $densitySpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "fluid-force.json"),
        "--force-fixture",
        $forceSpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "deformable-update.json"),
        "--deformable-fixture",
        $deformableUpdateSpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "deformable-solver.json"),
        "--deformable-solver-fixture",
        $deformableSolverSpirv
    )
    Invoke-Checked $memorySmoke @(
        (Join-Path $resultDir "deformable-collision.json"),
        "--deformable-collision-fixture",
        $deformableCollisionSpirv
    )
}

if ($AcceptReferenceBundle -and [string]::IsNullOrWhiteSpace($ReferenceBundle)) {
    $ReferenceBundle = Join-Path $repoDir "tools\reference\expected\reference-workstation.json"
}
if (-not [string]::IsNullOrWhiteSpace($ReferenceBundle)) {
    if ([string]::IsNullOrWhiteSpace($CapturedPtxDir)) {
        throw "Reference bundle validation requires -CapturedPtxDir for all Vulkan reports"
    }
    $ReferenceBundle = [System.IO.Path]::GetFullPath($ReferenceBundle)
    if (-not (Test-Path -LiteralPath $ReferenceBundle -PathType Leaf)) {
        throw "Reference bundle not found: $ReferenceBundle"
    }

    $candidateBundle = Join-Path $resultDir "reference-bundle-candidate.json"
    Invoke-Checked "python" @(
        (Join-Path $repoDir "tools\build-reference-bundle.py"),
        "--vulkan-results", $resultDir,
        "--base", $ReferenceBundle,
        "--output", $candidateBundle
    )
    if ($AcceptReferenceBundle) {
        Copy-Item -LiteralPath $candidateBundle -Destination $ReferenceBundle -Force
        Write-Host "Accepted reference bundle: $ReferenceBundle"
    }
    else {
        $candidateHash = (Get-FileHash -LiteralPath $candidateBundle -Algorithm SHA256).Hash
        $referenceHash = (Get-FileHash -LiteralPath $ReferenceBundle -Algorithm SHA256).Hash
        if ($candidateHash -ne $referenceHash) {
            throw "Generated reports differ from the reference bundle: $ReferenceBundle"
        }
        Write-Host "PASS: generated reports match $ReferenceBundle"
    }
    Remove-Item -LiteralPath $candidateBundle -Force
}

Write-Host ""
Write-Host "PASS: pxvk portable suite"
Write-Host "Results: $resultDir"
if ([string]::IsNullOrWhiteSpace($CapturedPtxDir)) {
    Write-Host "Captured PhysX kernels were not compiled or run; pass -CapturedPtxDir with traced or extracted PTX to add them."
}
