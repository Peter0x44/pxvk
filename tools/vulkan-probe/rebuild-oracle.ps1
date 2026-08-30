[CmdletBinding()]
param(
    [string]$ToolDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Resolve-Tool {
    param([string]$Name)
    if ([string]::IsNullOrWhiteSpace($ToolDir)) {
        return $Name
    }
    $path = Join-Path ([System.IO.Path]::GetFullPath($ToolDir)) $Name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required tool not found: $path"
    }
    return $path
}

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

$glslang = Resolve-Tool "glslangValidator.exe"
$disassembler = Resolve-Tool "spirv-dis.exe"
$assembler = Resolve-Tool "spirv-as.exe"
$validator = Resolve-Tool "spirv-val.exe"
$sourceDir = $PSScriptRoot
$glsl = Join-Path $sourceDir "oracle.comp"
$unpatchedSpv = Join-Path $sourceDir "oracle-unpatched.spv"
$unpatchedAssembly = Join-Path $sourceDir "oracle-unpatched.spvasm"
$assembly = Join-Path $sourceDir "oracle.spvasm"
$spirv = Join-Path $sourceDir "oracle.spv"

Invoke-Checked $glslang @("-V", "--target-env", "vulkan1.2", "-o", $unpatchedSpv, $glsl)
Invoke-Checked $disassembler @($unpatchedSpv, "-o", $unpatchedAssembly)

$lines = [System.Collections.Generic.List[string]](Get-Content -LiteralPath $unpatchedAssembly)
$fmaResults = @()
foreach ($line in $lines) {
    if ($line -match '^\s*(%\S+)\s*=\s*OpExtInst\s+%float\s+%\S+\s+Fma\s') {
        $fmaResults += $Matches[1]
    }
}
if ($fmaResults.Count -ne 1) {
    throw "Expected one GLSL.std.450 Fma result, found $($fmaResults.Count)"
}
$typeIndex = -1
for ($index = 0; $index -lt $lines.Count; ++$index) {
    if ($lines[$index] -match '=\s*OpType') {
        $typeIndex = $index
        break
    }
}
if ($typeIndex -lt 0) {
    throw "Could not locate SPIR-V type section"
}
$lines.Insert($typeIndex, "               OpDecorate $($fmaResults[0]) NoContraction")
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllLines($assembly, $lines, $utf8NoBom)

Invoke-Checked $assembler @(
    "--target-env", "spv1.5",
    "--preserve-numeric-ids",
    $assembly,
    "-o", $spirv
)
Invoke-Checked $validator @("--target-env", "vulkan1.2", $spirv)
Remove-Item -LiteralPath $unpatchedSpv, $unpatchedAssembly -Force
Write-Host "Rebuilt and validated $spirv"
