[CmdletBinding()]
param(
    [ValidateSet("MinGW", "MSVC")]
    [string]$Toolchain = "MinGW",
    [string]$OutputDir = "",
    [string]$CapturedPtxDir = "",
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
    $module0013 = Join-Path $CapturedPtxDir "module_0013_01_05.ptx"
    $module0010 = Join-Path $CapturedPtxDir "module_0010_01_05.ptx"
    $module0016 = Join-Path $CapturedPtxDir "module_0016_01_05.ptx"
    foreach ($path in @($module0013, $module0010, $module0016)) {
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
            throw "Captured PTX input not found: $path"
        }
    }

    Write-Host "Compiling and running the local captured PhysX fixtures..."
    $module0013Manifest = Join-Path $resultDir "module-0013-manifest.json"
    $module0010Manifest = Join-Path $resultDir "module-0010-manifest.json"
    $module0016Manifest = Join-Path $resultDir "module-0016-manifest.json"
    $calcHashSpirv = Join-Path $resultDir "fluid-calc-hash.spv"
    $reorderSpirv = Join-Path $resultDir "fluid-reorder.spv"
    $densitySpirv = Join-Path $resultDir "fluid-density.spv"
    $forceSpirv = Join-Path $resultDir "fluid-force.spv"

    $calcHashEntry = "_Z26KernelFluidCalcHashPackedD17PxgFixedSizeArrayIjES0_S_IK12PxcVectorPadE11PxgGpuArrayIKjES6_S_IK18PxgFluidParametersEj"
    $reorderEntry = "_Z42KernelReorderPosVelAndFindCellStartPackedD17PxgFixedSizeArrayI12PxcVectorPadES1_S_IjES2_S_IKS0_ES4_S_IKjES6_11PxgGpuArrayIS5_Ej"
    $densityEntry = "_Z19KernelFluidDensityD17PxgFixedSizeArrayIfES0_15DensityDebugGPUPiS_IK6float4ES_IKjES7_S7_S7_"
    $forceEntry = "_Z17KernelFluidForceD17PxgFixedSizeArrayI12PxcVectorPadE13ForceDebugGPUPiS_IK6float4ES6_S_IKfES_IKjESA_SA_SA_"

    Invoke-Checked $frontend @($module0013, $module0013Manifest)
    Invoke-Checked $frontend @($module0010, $module0010Manifest)
    Invoke-Checked $frontend @($module0016, $module0016Manifest)
    Invoke-Checked $emitter @(
        "--entry", $calcHashEntry, "64", "0", $module0013Manifest, $calcHashSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $reorderEntry, "64", "260", $module0013Manifest, $reorderSpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $densityEntry, "192", "0", $module0010Manifest, $densitySpirv
    )
    Invoke-Checked $emitter @(
        "--entry", $forceEntry, "128", "0", $module0016Manifest, $forceSpirv
    )
    Validate-SpirvIfAvailable @(
        $calcHashSpirv, $reorderSpirv, $densitySpirv, $forceSpirv
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
}

Write-Host ""
Write-Host "PASS: pxvk portable suite"
Write-Host "Results: $resultDir"
if ([string]::IsNullOrWhiteSpace($CapturedPtxDir)) {
    Write-Host "Captured PhysX fixtures were not run; pass -CapturedPtxDir to add them."
}
