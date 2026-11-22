# Maximal reconvergence in the Windows drivers

How Intel's and NVIDIA's Vulkan compilers implement
`VK_KHR_shader_maximal_reconvergence`, why it makes some warp-synchronous
code slower on Intel but not on NVIDIA, and how the drivers were examined.
AMD is not covered yet; the last section lists what to look for.

Measured and read on the development machine in November 2026:

| GPU | Driver | Files examined (driver store folder) |
|---|---|---|
| Intel iGPU | 101.9175 | `igc32.dll`, `igvk32.dll` (`iigd_dch.inf_amd64_413e993a06fd9302`) |
| NVIDIA RTX 5070 Laptop | 621.01 | `nvoglv32.dll`, `nvgpucomp32.dll`, `nvoglv64.dll`, `nvvm32.dll`, `nvvm70.dll` (`nvlti.inf_amd64_360580ba71c61256`) |

Addresses below are for these builds and change with every driver. The
drivers' binaries stay out of the repository; copy them to `tmp\` to examine
them.

## Background

Warp-synchronous PTX (PhysX, APEX) relies on a warp's 32 lanes running
together. pxvk recreates that with subgroup control barriers, and makes the
barriers well-defined by declaring the `MaximallyReconvergesKHR` execution
mode (`architecture.md`). An execution mode is a flag on a SPIR-V entry
point that constrains how the driver compiles it, like `LocalSize` or
`DenormPreserve`. This one guarantees that lanes which split at an `if` or a
loop rejoin exactly at its structured merge point, and that barriers and
subgroup operations see exactly the lanes the control flow says are
together. In exchange the compiler may not reshape control flow in ways that
change where lanes rejoin: tail duplication, jump threading, moving code
across divergent branches, and some branch flattening.

## Summary

| | Intel | NVIDIA |
|---|---|---|
| Cost in `pxvk-warptest`'s scan | about 40% | none measurable |
| Cost in Arkham Origins | none measurable | none measurable |
| Compiler used for Vulkan | `igc32.dll`, LLVM 14.0.5 | `nvgpucomp32.dll`, LLVM 23.0.0 |
| LLVM convergence tokens | no (LLVM 14 predates them) | supported |
| What the mode triggers | a `spv.loop.merge` marker at every loop merge, turned into opaque intrinsic calls before optimization | dedicated reconvergence passes with a uniformity threshold and a clean-up pass |
| Granularity | every loop merge, whether or not anything depends on it | apparently only where something depends on it |

- **On Intel the execution mode costs, the barriers don't.** Any scan
  variant that declares the mode is about 40% slower, even with only memory
  fences; control barriers without the mode cost 0–3%.
- **Intel marks every loop merge.** Its SPIR-V front end inserts
  `spv.loop.merge` markers whenever the mode is declared. A closed-source
  pass, "Vulkan convergence control", turns them into calls to a private
  intrinsic (GenISA 0x2c0f) that the driver also uses for Direct3D's
  `dx.break` loop-exit markers and for cooperative-vector code, probably a
  general "keep these lanes together" marker. It runs before every
  optimization, and blocks containing such calls cannot be merged, threaded
  or flattened. The scan then compiles to fewer instructions but more basic
  blocks, and runs slower.
- **NVIDIA keeps reconvergence points only where needed.** Its LLVM 23
  compiler supports LLVM's convergence tokens, which restrict optimization
  only around barriers and subgroup operations, and it has its own
  reconvergence passes that appear to skip uniform branches and remove
  unneeded points. Its hardware also rejoins divergent lanes with explicit
  convergence barriers in every shader, so the guarantee adds little.
- **Intel's approach is coarse, not wrong.** Pinning every loop merge is
  safe, but stricter than the guarantee needs: the `mode` variant has no
  lane-to-lane operation at all and is still slowed down.

pxvk keeps maximal reconvergence on both: it makes warp-synchronous code
defined, and its cost shows only in a stress test built around
lane-dependent loops, on Intel.

## Measurements

`pxvk-warptest` has test-only modes that take `maximal`'s pieces separately
(`testing.md`). Intel iGPU, driver 101.9175, median of four alternating
rounds, scan at skew 0, against `fence`:

| Mode | Execution mode | Barriers | 128 blocks | 512 blocks |
|---|---|---|---|---|
| `fence` | no | memory fences | 3.23 ms | 11.85 ms |
| `control` | no | control at stores | +2% | +1% |
| `merges` | no | control at stores and merges | +3% | +0% |
| `mode` | yes | memory fences | +41% | +42% |
| `nomerge` | yes | control at stores | +43% | +44% |
| `maximal` | yes | control at stores and merges | +37% | +37% |

At skew 31, where per-lane busy work dominates, the cost is 13–16%. The
divergent test (stores under a branch, no lane-dependent loop before them)
stays within 3% in every mode. Driver 101.8331 gave the same pattern. On the
RTX 5070 every mode is within a few percent of `fence`.

Intel's pipeline statistics (`PXVK_PIPELINE_STATS=1`) for the scan:

| Mode | Instructions | Basic blocks |
|---|---|---|
| `fence` | 306 | 21 |
| `control`, `merges` | 304, 305 | 21, 22 |
| `mode`, `nomerge` | 244 | 23 |
| `maximal` | 247 | 25 |

## Intel

1. **The execution mode check.** `igc32.dll` 0x1216ccd0 loads 6023
   (`MaximallyReconvergesKHR`, 0x1787), looks it up in a set at offset 0x628
   of the module object and, if present, builds LLVM IR that uses the name
   `spv.loop.merge` (string at 0x129d9f54). `igvk32.dll` also compares
   against 6023 (0x10dd0d25, 0x10dd0d47), next to SPIRV-Tools validator
   messages ("In entry points using the MaximallyReconvergesKHR execution
   mode, ..."): it embeds the validator.
2. **The pass.** The name `vulkan-convergence-control` and description
   "Vulkan convergence control" (0x129d9f64) are registered at 0x12091999.
   The pass (code around 0x12091800–0x12092300) walks every call, collects
   calls to `spv.loop.merge` (string compare at 0x12092106) and to
   `llvm.genx.*` intrinsic 0x2c0f (0x12091e50), and itself creates calls to
   intrinsic 0x2c0f (0x12091f9b, through the GenISA declaration helper at
   0x109917c0). What exactly it emits per marker was not worked out.
3. **Pass order.** Its `PassInfo` (ID 0x130fb53c, constructor wrapper
   0x12091840) is created only from 0x12002b73. The passes created around
   it, named through their own `PassInfo` records, are in order:
   `spv-vector-access-legalization`, `vk-lower-image-atomic-int64`,
   **`vulkan-convergence-control`**, `spv-bindings-metadata`,
   `sampler-ycbcr-conversion`, `lower-image-ops`, `vulkan-fast-math-controls`,
   `lower-64-bit-instruction`, `vulkan-relaxed-precision-variables-lowering`,
   `spv-oob-access-prevention`, `always-inline`,
   `igc-code-gen-context-wrapper`, `vk-setup-input-output`, `vk-memory-model`.
   That is the Vulkan front end's lowering, before IGC's optimizer.
4. **Intrinsic 0x2c0f.** The embedded LLVM is 14.0.5. Its intrinsic name
   table (0x12766bb0) has 10,761 entries, and IGC numbers its GenISA
   intrinsics after LLVM's, so 0x2c0f is GenISA entry 518. The open-source
   list has only 488 entries, so this one is private to the driver. It is
   created by the Vulkan pass, by `ConvertDXILPass` (0x11fa83e0, after
   looking up the global `dx.break.cond`), by
   `igc-cooperative-vector-lowering` (0x1032e740) and at 0x1207ca01.
   0x108626a8 checks for it together with 0x2a1b, which in the open-source
   numbering is `GenISA_dpas`, the matrix multiply-accumulate that
   cooperative vectors use (if the driver's order matches the open list that
   early).
5. **What does the flattening.** Open-source IGC's `BranchToSelect`, which
   linearizes divergent branches and refuses regions with convergent calls,
   is not in this driver: its pass name `igc-branch-to-select` does not
   occur in `igc32.dll`, so it is newer than the driver's compiler. In this
   build LLVM 14's own CFG simplification (folding two-entry branches into
   selects, speculation, jump threading) is the likely flattening, which an
   opaque call in a block blocks as well.

## NVIDIA

1. **Which compiler.** A 32-bit Vulkan process (`pxvk-warptest`) loads only
   `nvoglv32.dll` and `nvgpucomp32.dll` (32.0.16.2101). `nvoglv32.dll`,
   `nvoglv64.dll`, `nvvm32.dll` and `nvvm70.dll` contain an older front end
   on LLVM 7.0.1; `nvgpucomp32.dll` (93 MB) is built on **LLVM 23.0.0** and
   is the one Vulkan uses.
2. **LLVM's convergence support.** `nvgpucomp32.dll` contains LLVM's
   convergence-control machinery: the verifier's token checks
   ("Convergence region is not well-nested", "Convergence token used by an
   instruction other than llvm.experimental.convergence.loop ..."), the code
   generator's `convergencectrl_entry`, `_loop`, `_anchor` and `_glue`
   nodes, and `StripConvergenceIntrinsicsPass`. Tokens tie each convergent
   operation to one reconvergence point, so optimizations are restricted
   only around those operations. The strings show the support is there, not
   that NVIDIA's Vulkan path emits tokens.
3. **Its own reconvergence passes.** Among its other Vulkan passes
   (`GLVHoistKillPass`, `GLVPostUnrollHoistPass`, `GLVLateInstTransformsPass`,
   ...) are `GLVReconvergencePass` and `GLVReconvergenceOptimizationPass`,
   with value names `sync`, `syncPhi` and `reconverge`. The per-shader
   properties include `reconvergenceEnabled` beside `usesSubgroupInsts` and
   `usesCooperativeMatrix`; the compiler settings include
   `reconvergenceUniformityLookbackThreshold` and `enableReconvergence`. A
   uniformity threshold suggests branches all lanes take the same way are
   left alone, and the optimization pass that unneeded reconvergence points
   are removed.
4. **The hardware.** Since Volta, NVIDIA GPUs rejoin divergent lanes with
   explicit convergence barriers the compiler places in every shader. The
   backend's instruction modifiers include `.reliable::reconvergent`, and its
   statistics count `WARPSYNC` instructions. Rejoining at SPIR-V's merge
   points is close to what it does anyway.

## What is open source

| Part | Open source? |
|---|---|
| LLVM 14 (Intel's base) and LLVM 23 (NVIDIA's base) | yes, upstream LLVM (Apache 2.0 with LLVM exceptions) |
| LLVM's convergence tokens, their verifier (`llvm/include/llvm/IR/GenericConvergenceVerifierImpl.h`) and `StripConvergenceIntrinsicsPass` (`llvm/lib/Transforms/Utils/StripConvergenceIntrinsics.cpp`) | yes; developed largely for LLVM's SPIR-V backend, and AMD's GPU backend uses tokens too |
| Intel's shader compiler (IGC): optimization pipeline, GenISA intrinsics list | yes, github.com/intel/intel-graphics-compiler, but newer than what the driver ships |
| Intel's Vulkan front end, `vulkan-convergence-control`, private GenISA intrinsics such as 0x2c0f | no |
| NVIDIA's `GLV*` passes, including the reconvergence passes | no; nothing by those names is in upstream LLVM |
| NVIDIA's code generator (convergence barriers, SASS) | no |

So in both drivers the decisions examined here, where reconvergence points
go and what they restrict, are closed source.

## How the drivers were examined

- **No machine code from the drivers.** `vkGetPipelineExecutableInternalRepresentationsKHR`
  returns nothing on Intel or NVIDIA, only statistics. Intel's compiler dump
  switch (`ShaderDumpEnable` under `HKLM\SOFTWARE\Intel\IGFX\IGC`, and under
  `WOW6432Node` for 32-bit processes) does nothing: the setting's name does
  not occur in the release `igc32.dll`, in ASCII or UTF-16, while strings
  such as `SPV_KHR_maximal_reconvergence` do, so the debug settings are
  compiled out.
- **Which DLLs a process uses.** List a running 32-bit process's modules
  from the 32-bit PowerShell (`C:\Windows\SysWOW64\WindowsPowerShell\v1.0`);
  a 64-bit PowerShell does not show them.
- **Finding the code.** Search the DLL for the execution mode's number as an
  immediate (`87 17 00 00` in x86 instructions) and for related strings,
  then follow references with `objdump -d --start-address=...
  --stop-address=...` from the MinGW toolchain. Pass names are reachable
  through LLVM's `PassInfo` records (name, argument, ID, constructor), which
  also give the pipeline order from the code that creates the passes.
- **LLVM version and intrinsics.** The embedded version appears as a string
  (`LLVM14.0.5`, `LLVM 23.0.0`). In LLVM 14 the intrinsic names are a
  pointer array starting with `not_intrinsic`, whose length gives the ID
  where a vendor's own intrinsics start.
- **The open sources** for context: what upstream LLVM and open-source IGC
  do, keeping in mind that drivers ship older or private versions.

## AMD: what to look for

Not examined yet. A similar pass would start with:

- **Cost first.** Run `pxvk-warptest --device radeon --blocks 512 --runs 40`
  in alternating rounds and compare `fence`, `mode` and `maximal`. If the
  mode costs nothing, as on NVIDIA, the rest is curiosity.
- **Which DLLs.** AMD's Windows Vulkan driver is `amdvlk32.dll` and
  `amdvlk64.dll` in the driver store; list a running `pxvk-warptest`'s
  modules to see which compiler it loads.
- **The compiler.** AMD's Vulkan pipeline compiler LLPC was published as part
  of AMDVLK (github.com/GPUOpen-Drivers), over LLVM's AMDGPU backend, so much
  of it may be readable in source. Whether the Windows driver ships the same
  code, and which LLVM version, is to be checked.
- **Strings.** `MaximallyReconverges`, `maximal_reconvergence`,
  `convergencectrl`, `llvm.experimental.convergence`, `reconverg`, and the
  embedded LLVM version. AMD's hardware rejoins lanes by managing the
  execution mask in software (`s_cbranch_execz`, saving and restoring
  `exec`), so its compiler already structures control flow, which may make
  the guarantee cheap.
- **On Linux,** RADV (Mesa, with the ACO compiler) is open source and an
  alternative to AMD's own driver.

## Open questions

- Intel: what the pass emits per marker (a 0x2c0f call per loop exit, or per
  merge), which late pass finally removes the 0x2c0f calls, and which LLVM
  14 transformation the markers block in the scan.
- Intel on Linux: whether Mesa's `anv` (a different compiler) pays a similar
  cost. It prints its assembly with `INTEL_DEBUG=cs`.
- NVIDIA: whether `GLVReconvergencePass` emits LLVM convergence tokens or its
  own `sync`/`reconverge` constructs, and what the uniformity threshold is.
- A minimal case for a report to Intel: `pxvk-warptest`'s `mode` variant, the
  scan with only memory fences and the execution mode declared, is about 40%
  slower than without the mode, although nothing in it communicates between
  lanes.

## Reproducing

```powershell
build\x86\pxvk-warptest.exe --device intel --blocks 512 --runs 40
build\x86\pxvk-warptest.exe --device nvidia --blocks 512 --runs 40
$env:PXVK_PIPELINE_STATS = "1"; build\x86\pxvk-warptest.exe --device intel --blocks 64 --runs 1
```
