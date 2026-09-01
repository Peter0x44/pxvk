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
    scope = [ordered]@{
        physx_version_family = "2.8.x"
        cuda_address_size = 32
        compatibility = "classic GPU PhysX PTX"
    }
    invariant = [ordered]@{
        minimum_vulkan = "1.2"
        minimum_compute_queues = 1
        signed_zero_inf_nan_preserve_float32 = $true
        rounding_mode_rte_float32 = $true
        shader_int64 = $true
        cuda_visible_warp_size = 32
    }
    runtime_derived = [ordered]@{
        local_size = "PTX launch block dimensions"
        dispatch_count = "PTX launch grid dimensions"
        shared_bytes_per_workgroup = "module static shared bytes + launch dynamic shared bytes"
        storage_scratch_bytes = "align4(shared bytes per workgroup) * total dispatched workgroups"
        storage_buffer_descriptors = "bindings used by the selected generated kernel"
        storage_ranges_and_allocations = "caller allocations and selected-kernel resources"
    }
}

$deviceReports = @()
foreach ($device in $capabilities.devices) {
    $failures = New-Object System.Collections.Generic.List[string]
    try {
        $apiVersion = [version]$device.api_version
    }
    catch {
        [void]$failures.Add("invalid Vulkan API version: $($device.api_version)")
        $apiVersion = [version]"0.0"
    }

    if ($apiVersion -lt [version]$requirements.invariant.minimum_vulkan) {
        [void]$failures.Add("Vulkan $($device.api_version) is below 1.2")
    }
    if ([uint64]$device.compute_queue_count -lt $requirements.invariant.minimum_compute_queues) {
        [void]$failures.Add("no compute queue")
    }
    if (-not [bool]$device.float_controls.signed_zero_inf_nan_preserve_float32) {
        [void]$failures.Add("f32 signed-zero/Inf/NaN preservation is false")
    }
    if (-not [bool]$device.float_controls.rounding_mode_rte_float32) {
        [void]$failures.Add("f32 RTE is false")
    }
    if (-not [bool]$device.features.shader_int64) {
        [void]$failures.Add("shaderInt64 is false")
    }

    $subgroup32Available = ([uint64]$device.subgroup.default_size -eq 32)
    if (-not $subgroup32Available -and [bool]$device.features.subgroup_size_control) {
        try {
            $requiredSizeStagesText = [string]$device.subgroup.required_size_stages
            $requiredSizeStages = if ($requiredSizeStagesText.StartsWith("0x")) {
                [Convert]::ToUInt64($requiredSizeStagesText.Substring(2), 16)
            }
            else {
                [Convert]::ToUInt64($requiredSizeStagesText, 10)
            }
            $subgroup32Available =
                ([uint64]$device.subgroup.min_size -le 32) -and
                ([uint64]$device.subgroup.max_size -ge 32) -and
                (($requiredSizeStages -band 0x20) -ne 0)
        }
        catch {
            $subgroup32Available = $false
        }
    }
    if (-not $subgroup32Available) {
        [void]$failures.Add("compute shaders cannot provide CUDA-visible subgroup size 32")
    }

    $nativeFtz = [bool]$device.float_controls.denorm_flush_to_zero_float32
    $deviceReports += [pscustomobject][ordered]@{
        name = [string]$device.name
        vendor_id = [string]$device.vendor_id
        device_id = [string]$device.device_id
        api_version = [string]$device.api_version
        invariant_gate_pass = ($failures.Count -eq 0)
        native_float32_ftz = $nativeFtz
        software_ftz_required = (-not $nativeFtz)
        subgroup_size_32_available = $subgroup32Available
        failures = @($failures)
        runtime_limits = [ordered]@{
            max_compute_work_group_invocations = [uint64]$device.limits.max_compute_work_group_invocations
            max_compute_work_group_size = @($device.limits.max_compute_work_group_size)
            max_compute_work_group_count = @($device.limits.max_compute_work_group_count)
            max_compute_shared_memory_size = [uint64]$device.limits.max_compute_shared_memory_size
            max_per_stage_storage_buffers = [uint64]$device.limits.max_per_stage_storage_buffers
            max_descriptor_set_storage_buffers = [uint64]$device.limits.max_descriptor_set_storage_buffers
            max_storage_buffer_range = [uint64]$device.limits.max_storage_buffer_range
            max_memory_allocation_size = [uint64]$device.limits.max_memory_allocation_size
            max_buffer_size = [uint64]$device.limits.max_buffer_size
        }
    }
}

$eligibleDeviceCount = @(
    $deviceReports | Where-Object { $_.invariant_gate_pass }
).Count
$report = [ordered]@{
    schema_version = 4
    probe_pointer_bits = [int]$capabilities.pointer_bits
    vulkan_header_version = [string]$capabilities.vulkan_header_version
    loader_api_version = [string]$capabilities.loader_api_version
    dependency_source = "CMake FetchContent with pinned commits"
    capabilities_sha256 = (Get-FileHash -LiteralPath $capabilityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    requirements = $requirements
    eligible_device_count = $eligibleDeviceCount
    devices = $deviceReports
    limitations = @(
        "Invariant preflight does not certify an unknown launch; block dimensions, grid dimensions, descriptors, shared storage, ranges, and allocations are checked from the selected module and launch.",
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
    if ($device.invariant_gate_pass) {
        Write-Host ("PASS  {0} ({1}, {2}, subgroup size 32 available)" -f $device.name, $device.vendor_id, $ftz)
    }
    else {
        Write-Host ("FAIL  {0} ({1}): {2}" -f $device.name, $device.vendor_id, ($device.failures -join "; "))
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

if ($eligibleDeviceCount -eq 0) {
    exit 1
}
