[CmdletBinding()]
param(
    [ValidateSet("MinGW", "MSVC")]
    [string]$Toolchain = "MinGW",
    [switch]$RunMemorySmoke,
    [switch]$RunOracle,
    [string]$SourceDir = "",
    [string]$BuildDir = "",
    [string]$OutputDir = "",
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

if ([string]::IsNullOrWhiteSpace($SourceDir)) {
    $SourceDir = $PSScriptRoot
}
$SourceDir = [System.IO.Path]::GetFullPath($SourceDir)
$cmakeFile = Join-Path $SourceDir "CMakeLists.txt"
if (-not (Test-Path -LiteralPath $cmakeFile -PathType Leaf)) {
    throw "Source directory '$SourceDir' does not contain CMakeLists.txt. Keep preflight.ps1 beside the probe sources or pass -SourceDir."
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path ([System.Environment]::CurrentDirectory) "vulkan-preflight"
}
else {
    $OutputDir = [System.IO.Path]::GetFullPath($OutputDir)
}
if ([string]::IsNullOrWhiteSpace($BuildDir)) {
    $BuildDir = Join-Path $OutputDir ("build-" + $Toolchain.ToLowerInvariant())
}
else {
    $BuildDir = [System.IO.Path]::GetFullPath($BuildDir)
}
New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

$configureArguments = @(
    "-S", $SourceDir,
    "-B", $BuildDir
)
if ($Toolchain -eq "MinGW") {
    if ([string]::IsNullOrWhiteSpace($Generator)) {
        $Generator = "MinGW Makefiles"
    }
    $configureArguments += @(
        "-G", $Generator,
        "-DCMAKE_C_COMPILER=$CCompiler",
        "-DCMAKE_CXX_COMPILER=$CxxCompiler",
        "-DCMAKE_MAKE_PROGRAM=$MakeProgram",
        "-DCMAKE_BUILD_TYPE=Release"
    )
}
else {
    if ([string]::IsNullOrWhiteSpace($Generator)) {
        $configureArguments += @("-A", "Win32")
    }
    else {
        $configureArguments += @("-G", $Generator, "-A", "Win32")
    }
}

Invoke-Checked -File "cmake" -Arguments $configureArguments
Invoke-Checked -File "cmake" -Arguments @("--build", $BuildDir, "--config", "Release", "--parallel")

$executableCandidates = @(
    (Join-Path $BuildDir "vulkan-probe.exe"),
    (Join-Path $BuildDir "Release\vulkan-probe.exe")
)
$probeExecutable = $executableCandidates | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } | Select-Object -First 1
if ($null -eq $probeExecutable) {
    throw "Built vulkan-probe.exe was not found under $BuildDir"
}

$memorySmokePath = Join-Path $OutputDir "memory-smoke.json"
$oraclePath = Join-Path $OutputDir "oracle.json"
if ($RunMemorySmoke -or $RunOracle) {
    $memorySmokeCandidates = @(
        (Join-Path $BuildDir "vulkan-memory-smoke.exe"),
        (Join-Path $BuildDir "Release\vulkan-memory-smoke.exe")
    )
    $memorySmokeExecutable = $memorySmokeCandidates |
        Where-Object { Test-Path -LiteralPath $_ -PathType Leaf } |
        Select-Object -First 1
    if ($null -eq $memorySmokeExecutable) {
        throw "Built vulkan-memory-smoke.exe was not found under $BuildDir"
    }
}

$capabilityPath = Join-Path $OutputDir "capabilities.json"
$reportPath = Join-Path $OutputDir "preflight.json"
Invoke-Checked -File $probeExecutable -Arguments @($capabilityPath)

$capabilities = Get-Content -LiteralPath $capabilityPath -Raw | ConvertFrom-Json
if ([int]$capabilities.pointer_bits -ne 32) {
    throw "Probe is not PE32: pointer_bits=$($capabilities.pointer_bits)"
}

$requirements = [ordered]@{
    common = [ordered]@{
        minimum_vulkan = "1.2"
        minimum_compute_queues = 1
        signed_zero_inf_nan_preserve_float32 = $true
        rounding_mode_rte_float32 = $true
        minimum_compute_storage_buffers = 4
    }
    profiles = [ordered]@{
        mirrors_edge_captured = [ordered]@{
            requires_shader_int64 = $true
            minimum_workgroup_invocations = 256
            minimum_workgroup_size_x = 256
        }
        sample_cloth_scene_1 = [ordered]@{
            requires_shader_int64 = $false
            minimum_workgroup_invocations = 512
            minimum_workgroup_size_x = 512
            shared_backing = "storage_buffer"
            dispatch_dimensions = "x_only"
        }
    }
}

$deviceReports = @()
foreach ($device in $capabilities.devices) {
    $commonFailures = New-Object System.Collections.Generic.List[string]
    try {
        $apiVersion = [version]$device.api_version
    }
    catch {
        [void]$commonFailures.Add("invalid Vulkan API version: $($device.api_version)")
        $apiVersion = [version]"0.0"
    }

    if ($apiVersion -lt [version]$requirements.common.minimum_vulkan) {
        [void]$commonFailures.Add("Vulkan $($device.api_version) is below 1.2")
    }
    if ([uint64]$device.compute_queue_count -lt $requirements.common.minimum_compute_queues) {
        [void]$commonFailures.Add("no compute queue")
    }
    if (-not [bool]$device.float_controls.signed_zero_inf_nan_preserve_float32) {
        [void]$commonFailures.Add("f32 signed-zero/Inf/NaN preservation is false")
    }
    if (-not [bool]$device.float_controls.rounding_mode_rte_float32) {
        [void]$commonFailures.Add("f32 RTE is false")
    }
    if ([uint64]$device.limits.max_per_stage_storage_buffers -lt $requirements.common.minimum_compute_storage_buffers) {
        [void]$commonFailures.Add("compute-stage storage buffers are below 4")
    }
    if ([uint64]$device.limits.max_descriptor_set_storage_buffers -lt $requirements.common.minimum_compute_storage_buffers) {
        [void]$commonFailures.Add("descriptor-set storage buffers are below 4")
    }

    $mirrorFailures = New-Object System.Collections.Generic.List[string]
    $sampleClothFailures = New-Object System.Collections.Generic.List[string]
    foreach ($failure in $commonFailures) {
        [void]$mirrorFailures.Add($failure)
        [void]$sampleClothFailures.Add($failure)
    }
    if (-not [bool]$device.features.shader_int64) {
        [void]$mirrorFailures.Add("shaderInt64 is false")
    }
    if ([uint64]$device.limits.max_compute_work_group_invocations -lt $requirements.profiles.mirrors_edge_captured.minimum_workgroup_invocations) {
        [void]$mirrorFailures.Add("max workgroup invocations is below 256")
    }
    if ([uint64]$device.limits.max_compute_work_group_size[0] -lt $requirements.profiles.mirrors_edge_captured.minimum_workgroup_size_x) {
        [void]$mirrorFailures.Add("max workgroup size X is below 256")
    }
    if ([uint64]$device.limits.max_compute_work_group_invocations -lt $requirements.profiles.sample_cloth_scene_1.minimum_workgroup_invocations) {
        [void]$sampleClothFailures.Add("max workgroup invocations is below 512")
    }
    if ([uint64]$device.limits.max_compute_work_group_size[0] -lt $requirements.profiles.sample_cloth_scene_1.minimum_workgroup_size_x) {
        [void]$sampleClothFailures.Add("max workgroup size X is below 512")
    }

    $nativeFtz = [bool]$device.float_controls.denorm_flush_to_zero_float32
    $deviceReports += [pscustomobject][ordered]@{
        name = [string]$device.name
        vendor_id = [string]$device.vendor_id
        device_id = [string]$device.device_id
        api_version = [string]$device.api_version
        common_gate_pass = ($commonFailures.Count -eq 0)
        native_float32_ftz = $nativeFtz
        software_ftz_required = (-not $nativeFtz)
        common_failures = @($commonFailures)
        profiles = [ordered]@{
            mirrors_edge_captured = [ordered]@{
                pass = ($mirrorFailures.Count -eq 0)
                failures = @($mirrorFailures)
            }
            sample_cloth_scene_1 = [ordered]@{
                pass = ($sampleClothFailures.Count -eq 0)
                failures = @($sampleClothFailures)
            }
        }
    }
}

$eligibleCounts = [ordered]@{
    mirrors_edge_captured = @(
        $deviceReports |
            Where-Object { $_.profiles.mirrors_edge_captured.pass }
    ).Count
    sample_cloth_scene_1 = @(
        $deviceReports |
            Where-Object { $_.profiles.sample_cloth_scene_1.pass }
    ).Count
}
$report = [ordered]@{
    schema_version = 3
    probe_pointer_bits = [int]$capabilities.pointer_bits
    vulkan_header_version = [string]$capabilities.vulkan_header_version
    loader_api_version = [string]$capabilities.loader_api_version
    dependency_source = "CMake FetchContent with pinned commits"
    capabilities_sha256 = (Get-FileHash -LiteralPath $capabilityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    requirements = $requirements
    eligible_device_counts = $eligibleCounts
    devices = $deviceReports
    limitations = @(
        "Hardware preflight does not prove arbitrary caller-sized allocations; optional execution probes exercise a 32 MiB fixture.",
        "Hardware preflight does not execute SPIR-V or validate software FTZ.",
        "AMD game integration still requires the deferred non-NVIDIA NVAPI gate."
    )
}
$reportJson = ($report | ConvertTo-Json -Depth 8) + [System.Environment]::NewLine
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($reportPath, $reportJson, $utf8NoBom)
if ($RunMemorySmoke) {
    Invoke-Checked -File $memorySmokeExecutable -Arguments @($memorySmokePath)
}
if ($RunOracle) {
    $oracleShader = Join-Path (Split-Path -Parent $memorySmokeExecutable) "oracle.spv"
    if (-not (Test-Path -LiteralPath $oracleShader -PathType Leaf)) {
        throw "Built oracle.spv was not found beside vulkan-memory-smoke.exe"
    }
    Invoke-Checked -File $memorySmokeExecutable -Arguments @($oraclePath, $oracleShader)
}

Write-Host ""
Write-Host "pxvk Vulkan hardware preflight"
foreach ($device in $deviceReports) {
    $ftz = if ($device.software_ftz_required) { "software FTZ required" } else { "native f32 FTZ" }
    Write-Host ("DEVICE {0} ({1}, {2})" -f $device.name, $device.vendor_id, $ftz)
    foreach ($profileName in @("mirrors_edge_captured", "sample_cloth_scene_1")) {
        $profile = $device.profiles.$profileName
        if ($profile.pass) {
            Write-Host ("  PASS  {0}" -f $profileName)
        }
        else {
            Write-Host ("  FAIL  {0}: {1}" -f $profileName, ($profile.failures -join "; "))
        }
    }
}
Write-Host "Capabilities: $capabilityPath"
Write-Host "Report:       $reportPath"
if ($RunMemorySmoke) {
    Write-Host "Memory smoke: $memorySmokePath"
}
if ($RunOracle) {
    Write-Host "Oracle:       $oraclePath"
}
Write-Host ""
if ($RunOracle) {
    Write-Host "Hardware preflight and standalone Vulkan oracle completed."
}
elseif ($RunMemorySmoke) {
    Write-Host "Hardware preflight and integer/global-memory execution completed."
}
else {
    Write-Host "This checks cross-vendor hardware eligibility only; use -RunOracle for semantic execution."
}

if ($eligibleCounts.mirrors_edge_captured -eq 0) {
    exit 1
}
