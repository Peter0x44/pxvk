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
    minimum_vulkan = "1.2"
    minimum_compute_queues = 1
    shader_int64 = $true
    signed_zero_inf_nan_preserve_float32 = $true
    rounding_mode_rte_float32 = $true
    minimum_workgroup_invocations = 256
    minimum_workgroup_size_x = 256
    minimum_workgroup_count_x = 94
    minimum_shared_memory_bytes = 10756
    minimum_storage_buffer_range_bytes = 33554432
    minimum_memory_allocation_bytes = 33554432
    minimum_push_constants_bytes = 100
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

    if ($apiVersion -lt [version]$requirements.minimum_vulkan) {
        [void]$failures.Add("Vulkan $($device.api_version) is below 1.2")
    }
    if ([uint64]$device.compute_queue_count -lt $requirements.minimum_compute_queues) {
        [void]$failures.Add("no compute queue")
    }
    if (-not [bool]$device.features.shader_int64) {
        [void]$failures.Add("shaderInt64 is false")
    }
    if (-not [bool]$device.float_controls.signed_zero_inf_nan_preserve_float32) {
        [void]$failures.Add("f32 signed-zero/Inf/NaN preservation is false")
    }
    if (-not [bool]$device.float_controls.rounding_mode_rte_float32) {
        [void]$failures.Add("f32 RTE is false")
    }
    if ([uint64]$device.limits.max_compute_work_group_invocations -lt $requirements.minimum_workgroup_invocations) {
        [void]$failures.Add("max workgroup invocations is below 256")
    }
    if ([uint64]$device.limits.max_compute_work_group_size[0] -lt $requirements.minimum_workgroup_size_x) {
        [void]$failures.Add("max workgroup size X is below 256")
    }
    if ([uint64]$device.limits.max_compute_work_group_count[0] -lt $requirements.minimum_workgroup_count_x) {
        [void]$failures.Add("max workgroup count X is below 94")
    }
    if ([uint64]$device.limits.max_compute_shared_memory_size -lt $requirements.minimum_shared_memory_bytes) {
        [void]$failures.Add("shared memory is below 10756 bytes")
    }
    if ([uint64]$device.limits.max_storage_buffer_range -lt $requirements.minimum_storage_buffer_range_bytes) {
        [void]$failures.Add("storage buffer range is below 32 MiB")
    }
    if ([double]$device.limits.max_memory_allocation_size -lt $requirements.minimum_memory_allocation_bytes) {
        [void]$failures.Add("maximum allocation is below 32 MiB")
    }
    if ([uint64]$device.limits.max_push_constants_size -lt $requirements.minimum_push_constants_bytes) {
        [void]$failures.Add("push constants are below 100 bytes")
    }

    $nativeFtz = [bool]$device.float_controls.denorm_flush_to_zero_float32
    $deviceReports += [pscustomobject][ordered]@{
        name = [string]$device.name
        vendor_id = [string]$device.vendor_id
        device_id = [string]$device.device_id
        api_version = [string]$device.api_version
        hardware_gate_pass = ($failures.Count -eq 0)
        native_float32_ftz = $nativeFtz
        software_ftz_required = (-not $nativeFtz)
        failures = @($failures)
    }
}

$eligibleCount = @($deviceReports | Where-Object { $_.hardware_gate_pass }).Count
$report = [ordered]@{
    schema_version = 2
    probe_pointer_bits = [int]$capabilities.pointer_bits
    vulkan_header_version = [string]$capabilities.vulkan_header_version
    loader_api_version = [string]$capabilities.loader_api_version
    dependency_source = "CMake FetchContent with pinned commits"
    capabilities_sha256 = (Get-FileHash -LiteralPath $capabilityPath -Algorithm SHA256).Hash.ToLowerInvariant()
    requirements = $requirements
    eligible_device_count = $eligibleCount
    devices = $deviceReports
    limitations = @(
        "Hardware preflight does not allocate the 32 MiB device/staging buffers.",
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
Write-Host "pxvk Mirror's Edge Vulkan hardware preflight"
foreach ($device in $deviceReports) {
    if ($device.hardware_gate_pass) {
        $ftz = if ($device.software_ftz_required) { "software FTZ required" } else { "native f32 FTZ" }
        Write-Host ("PASS  {0} ({1}, {2})" -f $device.name, $device.vendor_id, $ftz)
    }
    else {
        Write-Host ("FAIL  {0}: {1}" -f $device.name, ($device.failures -join "; "))
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

if ($eligibleCount -eq 0) {
    exit 1
}
