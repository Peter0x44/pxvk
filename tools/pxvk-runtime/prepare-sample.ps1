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
$updateEntry = "_Z24PxgCudaDeformableUpdateDPKhjPi"
$collisionEntry = "_Z29KernelDeformableUberCollisionPi"
$solverEntry = "_Z26KernelDeformableUberSolverILj512EEvjbPhPi"
$jobs = @(
    @($updateEntry, "64", "0", "deformable-update", "--entry"),
    @($collisionEntry, "64", "0", "deformable-collision", "--entry"),
    @($solverEntry, "512", "42240", "sample-cloth-solver", "--entry-shared-storage")
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
Write-Host "Prepared pxvk SampleCloth stage: $applicationDirectory"
Write-Host "Run: $launcher"
