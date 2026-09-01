[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SampleDirectory,
    [Parameter(Mandatory = $true)][string]$CapturedPtxDirectory,
    [string]$OutputDirectory = "",
    [ValidateSet("MinGW", "MSVC")][string]$Toolchain = "MinGW",
    [string]$Generator = "",
    [string]$CCompiler = "i686-w64-mingw32-gcc",
    [string]$CxxCompiler = "i686-w64-mingw32-g++",
    [string]$MakeProgram = "mingw32-make",
    [switch]$Force
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
    elseif ([string]::IsNullOrWhiteSpace($Generator)) {
        $arguments += @("-A", "Win32")
    }
    else {
        $arguments += @("-G", $Generator, "-A", "Win32")
    }
    Invoke-Checked "cmake" $arguments
    Invoke-Checked "cmake" @("--build", $Build, "--config", "Release", "--parallel")
}

function Resolve-BuiltFile {
    param(
        [Parameter(Mandatory = $true)][string]$Build,
        [Parameter(Mandatory = $true)][string]$Name
    )
    foreach ($candidate in @(
        (Join-Path $Build $Name),
        (Join-Path $Build "Release\$Name")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) { return $candidate }
    }
    throw "Built file '$Name' was not found under $Build"
}

function Resolve-PtxForEntry {
    param(
        [Parameter(Mandatory = $true)][string]$Directory,
        [Parameter(Mandatory = $true)][string]$Entry
    )
    $matches = @(
        Get-ChildItem -LiteralPath $Directory -Filter "*.ptx" -File |
            Where-Object { [System.IO.File]::ReadAllText($_.FullName).Contains($Entry) }
    )
    if ($matches.Count -ne 1) {
        throw ("Expected one PTX file containing '{0}' under {1}; found {2}" -f
            $Entry, $Directory, $matches.Count)
    }
    return $matches[0].FullName
}

$repo = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))
$sample = [System.IO.Path]::GetFullPath($SampleDirectory)
$captured = [System.IO.Path]::GetFullPath($CapturedPtxDirectory)
if (-not (Test-Path -LiteralPath $sample -PathType Container)) {
    throw "PhysX sample directory not found: $sample"
}
if (-not (Test-Path -LiteralPath $captured -PathType Container)) {
    throw "Captured PTX directory not found: $captured"
}
if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
    $OutputDirectory = Join-Path (Split-Path -Parent $sample) "pxvk-samplecloth-stage"
}
$stage = [System.IO.Path]::GetFullPath($OutputDirectory)
if (Test-Path -LiteralPath $stage) {
    if (-not $Force) { throw "Stage already exists: $stage (pass -Force to replace it)" }
    Remove-Item -LiteralPath $stage -Recurse -Force
}

$toolchainName = $Toolchain.ToLowerInvariant()
$runtimeBuild = Join-Path $repo "tmp\build-pxvk-runtime-$toolchainName"
$frontendBuild = Join-Path $repo "tmp\build-ptx-frontend-$toolchainName"
Configure-X86Build (Join-Path $repo "tools\pxvk-runtime") $runtimeBuild
Configure-X86Build (Join-Path $repo "tools\ptx-frontend") $frontendBuild
Invoke-Checked "ctest" @("--test-dir", $runtimeBuild, "-C", "Release", "--output-on-failure")

New-Item -ItemType Directory -Path $stage | Out-Null
Copy-Item -Path (Join-Path $sample "*") -Destination $stage -Recurse -Force
$sampleExecutable = Get-ChildItem -LiteralPath $stage -Filter "SampleCloth.exe" -File -Recurse |
    Select-Object -First 1
if ($null -eq $sampleExecutable) {
    throw "SampleCloth.exe was not found in the staged sample"
}
$applicationDirectory = $sampleExecutable.DirectoryName
$spirvDirectory = Join-Path $applicationDirectory "spirv"
$manifestDirectory = Join-Path $runtimeBuild "samplecloth-manifests"
New-Item -ItemType Directory -Force -Path $spirvDirectory, $manifestDirectory | Out-Null

$frontend = Resolve-BuiltFile $frontendBuild "ptx-frontend.exe"
$emitter = Resolve-BuiltFile $frontendBuild "ptx-spirv-emitter.exe"
$runtime = Resolve-BuiltFile $runtimeBuild "nvcuda.dll"
$runtimeLauncher = Resolve-BuiltFile $runtimeBuild "pxvk-launch.exe"
$frameLimiter = Resolve-BuiltFile $runtimeBuild "pxvk-frame-limit.dll"
$updateEntry = "_Z24PxgCudaDeformableUpdateDPKhjPi"
$collisionEntry = "_Z29KernelDeformableUberCollisionPi"
$solverEntry = "_Z26KernelDeformableUberSolverILj512EEvjbPhPi"
$solverInitEntry = "_Z26KernelDeformableInitSolverjPi"
$solverPacketsEntry = "_Z28KernelDeformableSolvePacketsILj512EEvPK31PxgDeformableSolverBlockControlbPi"
$solverParticlesEntry = "_Z30KernelDeformableSolveParticlesjbPi"
$solverPackEntry = "_Z26KernelDeformableSolverPackjPhPi"
$selfHashEntry = "_Z41PxgCudaDeformableSelfCollisionCreateHashD11PxgGpuArrayIjES0_S_IK12PxcVectorPadES_IKjEfPi"
$radixEntry = "_Z15radixSortBlocksILj32ELj0ELb0EEvP5uint4S1_S1_S1_j"
$selfReorderEntry = "_Z44PxgCudaDeformableSelfCollisionReorderInputsD11PxgGpuArrayIfES_I12PxcVectorPadES2_S2_S_IKjES4_S_IKfES_IKS1_ES8_S8_Pi"
$selfCellStartEntry = "_Z44PxgCudaDeformableSelfCollisionFindCellStartD11PxgGpuArrayIjES_IKjEPi"
$selfCompareEntry = "_Z47PxgCudaDeformableSelfCollisionCompareParticlesD11PxgGpuArrayI12PxcVectorPadES_IKjES3_S3_S_IKS0_ES5_S_IKfES5_fffPi"
$jobs = @(
    @($updateEntry, "64", "0", "deformable-update", "--entry"),
    @($collisionEntry, "64", "0", "deformable-collision", "--entry"),
    @($solverEntry, "64", "2016", "sample-cloth-attachment-solver", "--entry"),
    @($solverEntry, "128", "3168", "sample-cloth-pipe-solver", "--entry"),
    @($solverEntry, "512", "42240", "sample-cloth-solver", "--entry-shared-storage"),
    @($solverEntry, "512", "23616", "sample-cloth-damping-solver", "--entry-shared-storage"),
    @($solverInitEntry, "1024", "0", "sample-cloth-tearing-init-solver", "--entry"),
    @($solverPacketsEntry, "512", "33056", "sample-cloth-tearing-packets-large", "--entry-shared-storage-dynamic"),
    @($solverPacketsEntry, "512", "2624", "sample-cloth-tearing-packets-small", "--entry"),
    @($solverParticlesEntry, "1024", "0", "sample-cloth-tearing-solve-particles", "--entry"),
    @($solverPackEntry, "1024", "0", "sample-cloth-tearing-solver-pack", "--entry"),
    @($selfHashEntry, "128", "0", "self-collision-hash", "--entry"),
    @($radixEntry, "256", "4096", "radix-sort", "--entry"),
    @($selfReorderEntry, "128", "0", "self-collision-reorder", "--entry"),
    @($selfCellStartEntry, "128", "0", "self-collision-cell-start", "--entry"),
    @($selfCompareEntry, "128", "0", "self-collision-compare", "--entry"),
    @($solverEntry, "512", "15376", "sample-cloth-self-collision-solver", "--entry-shared-storage")
)
foreach ($job in $jobs) {
    $ptx = Resolve-PtxForEntry $captured $job[0]
    $manifest = Join-Path $manifestDirectory ($job[3] + ".json")
    $shader = Join-Path $spirvDirectory ($job[3] + ".spv")
    Invoke-Checked $frontend @($ptx, $manifest)
    Invoke-Checked $emitter @($job[4], $job[0], $job[1], $job[2], $manifest, $shader)
}
Copy-Item -LiteralPath $runtime -Destination (Join-Path $applicationDirectory "nvcuda.dll") -Force
Copy-Item -LiteralPath $runtimeLauncher -Destination (Join-Path $applicationDirectory "pxvk-launch.exe") -Force
Copy-Item -LiteralPath $frameLimiter -Destination (Join-Path $applicationDirectory "pxvk-frame-limit.dll") -Force

$launcher = Join-Path $applicationDirectory "run-pxvk-samplecloth.cmd"
$launcherText = @'
@echo off
setlocal
set "PXVK_SHADER_DIR=%~dp0spirv"
set "PXVK_LOG=%~dp0pxvk-runtime.log"
"%~dp0pxvk-launch.exe" "%~dp0SampleCloth.exe"
'@
[System.IO.File]::WriteAllText($launcher, $launcherText,
    [System.Text.Encoding]::ASCII)

$nativeLauncher = Join-Path $applicationDirectory "run-native-cuda-samplecloth.cmd"
$nativeLauncherText = @'
@echo off
setlocal
pushd "%~dp0"
set "PXVK_SHADER_DIR="
set "PXVK_LOG="
"%~dp0SampleCloth.exe" %*
set "PXVK_NATIVE_EXIT=%ERRORLEVEL%"
popd
exit /b %PXVK_NATIVE_EXIT%
'@
[System.IO.File]::WriteAllText($nativeLauncher, $nativeLauncherText,
    [System.Text.Encoding]::ASCII)

$limitedNativeLauncher = Join-Path $applicationDirectory "run-native-cuda-50fps-samplecloth.cmd"
$limitedNativeLauncherText = @'
@echo off
setlocal
set "PXVK_SHADER_DIR="
set "PXVK_LOG="
"%~dp0pxvk-launch.exe" --inject-dll "%~dp0pxvk-frame-limit.dll" "%~dp0SampleCloth.exe" %*
'@
[System.IO.File]::WriteAllText($limitedNativeLauncher, $limitedNativeLauncherText,
    [System.Text.Encoding]::ASCII)
Write-Host "Prepared pxvk SampleCloth stage: $applicationDirectory"
Write-Host "Run pxvk: $launcher"
Write-Host "Run native CUDA: $nativeLauncher"
Write-Host "Run native CUDA at 50 FPS: $limitedNativeLauncher"
