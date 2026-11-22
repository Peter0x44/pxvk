# GPU notes

What pxvk has learned about each GPU family: the hardware it was tested on,
driver behaviour that matters for PhysX, and data on GPUs not yet tested.
Details and measurements live in the linked documents; this page collects
them per vendor.

## Tested hardware

| GPU | Driver | Role | Subgroups | Notes |
|---|---|---|---|---|
| NVIDIA GeForce RTX 5070 Laptop (Blackwell) | 621.01 | native reference for captures and opcheck (with the driver's 32-bit allowlist) | 32 | bit-exact target |
| NVIDIA GeForce RTX 3070 Laptop (Ampere) | — | earlier native reference (`charter.md`) | 32 | |
| Intel iGPU ("Intel(R) Graphics") | 101.8331, 101.9175 | second GPU in the RTX 5070 laptop | 8, 16, 32 | |
| Intel UHD | — | earlier test iGPU | 32 (required) | |
| AMD Radeon 8060S (RDNA 3.5) | 26.8.1–26.11.1 | AMD test laptop, no NVIDIA driver | 32, 64 | Arkham Origins not yet run |
| Mesa lavapipe (software) | Mesa 26.2.6 | fallback paths without a second GPU | 8 (vector width / 32) | `testing.md` |

## NVIDIA

- **Subgroups:** always 32 lanes, which is exactly a CUDA warp.
- **Denormals:** the Vulkan driver supports neither `DenormFlushToZero`
  nor `DenormPreserve`. By default it flushes `add`, `sub`, `mul` and `fma`
  but not `abs`, `neg`, `min`, `max` or `floor`. So `.ftz` is emulated per
  instruction, and plain arithmetic flushes where CUDA keeps denormals,
  which affects APEX near 10^-38 (`architecture.md`). D3D12 on the same
  hardware honours both modes; only Vulkan lacks them.
- **SM count:** reported by `VK_NV_shader_sm_builtins` (36 on the RTX 5070
  laptop) and passed to CUDA as on native. APEX 1.3 sizes its grids by it
  (198.8 against 207.4 fps in Arkham Origins with the persona's 16).
- **Threads per SM:** the driver reports 1536 where pxvk's occupancy
  answer assumes 2048. Arkham Origins' launches are the same either way
  (`roadmap.md`).
- **Warp synchronization:** `VK_KHR_shader_maximal_reconvergence` is
  supported (driver 621.01). `pxvk-warptest` never saw lanes out of step,
  with any barrier strategy, and with maximal reconvergence the regression
  check stays bit-exact and opcheck matches its native baseline (November
  2026).
- **32-bit CUDA on RTX 50:** the driver offers the 32-bit runtime only to
  allowlisted executables (driver 591.44 and later). Native tools have to
  run under an allowlisted name, such as `tmp\opcheck-native\borderlands2.exe`
  (`nvidia-32bit-physx.md`). `ptxas` cannot JIT 32-bit PTX for sm_90 and
  later, so two opcheck variants have no native result.
- **Pipeline statistics:** only register count and binary size; there is
  no machine code to inspect.
- **Timestamps:** a `BOTTOM_OF_PIPE` timestamp after a dispatch is written
  once its last workgroup has started, not finished, on Windows (621.01)
  as on Linux (620.73.04): 64 blocks of the scan read 0.00 ms of GPU time
  against 10.6 ms wall-clock. A memory barrier from shader writes to the
  host stage before the timestamp fixes it, so pxvk records one when timing
  (`testing.md`).
- **Host memory:** the readback buffer gets host-visible, cached and
  coherent memory, with no invalidate.

## Intel

- **Subgroups:** 8, 16 and 32 lanes; warp-synchronous kernels require 32.
  At 16 and 8 lanes (`PXVK_SUBGROUP_SIZE`) warp emulation runs PhysX 2.x
  correctly, but not 15 APEX kernels. At 8 lanes the limit of 64 subgroups
  per workgroup also rules out blocks above 512 threads. The width makes no
  speed difference (`testing.md`).
- **Denormals:** native `DenormFlushToZero` and `DenormPreserve`, so `.ftz`
  needs no emulation.
- **Approximation units:** `rcp`, `rsqrt` and `sqrt` differ from NVIDIA by
  1–2 ULP, and iterative solvers drift a little. The regression check
  tolerates this off NVIDIA.
- **Warp synchronization:**
  - Without an acquire before volatile shared loads, lanes read stale shared
    memory (APEX's `BasicIOS_compactKernel` lost particles).
  - Maximal reconvergence is supported (drivers 101.8331 and 101.9175).
    The regression check gives the same results with it as with fences,
    and `pxvk-warptest` never saw lanes out of step with any strategy
    (November 2026; 101.9175 gave identical results to 101.8331 in the
    regression check, opcheck and `pxvk-warptest`).
  - The `MaximallyReconvergesKHR` execution mode, not the barriers, makes
    `pxvk-warptest`'s scan about 40% slower: with it the compiler keeps
    more basic blocks and spends fewer instructions, apparently not
    flattening the scan's lane-dependent loops. Nothing measurable in
    Arkham Origins (`testing.md`). The driver marks every loop merge when
    the mode is declared, reusing its Direct3D `dx.break` mechanism
    (`maximal-reconvergence-drivers.md`). It is the Windows compiler, not
    the hardware: Mesa's ANV (26.3.1, under Wine) compiles the scan
    identically with and without the mode, though its code takes about
    1.5 times the Windows driver's GPU time in every mode.
- **Host memory:** the Windows driver (101.9175) has a host-visible type
  that is cached and coherent (and device-local), which pxvk's readback
  buffer uses with no invalidate. ANV (Mesa 26.3.1) offers cached or
  coherent host memory but never both, so there the readback buffer is
  cached and invalidated before each read; before that, uncached reads ran
  at about 120 MB/s (`architecture.md`).
- **Memory limits:**
  - `maxStorageBufferRange` is 4 bytes short of 1 GiB, so pxvk reports the
    device's memory as total memory: Arkham Origins enables GPU APEX only
    from 1 GiB.
  - Shared memory is 32 KiB against the cloth solver's 42 KiB, so the
    solver uses storage-backed shared memory.
- **SM count:** Vulkan exposes none, so pxvk uses the persona's 16.
- **Float atomics:**
  - Newer iGPUs have native global float atomic add, but it keeps a
    subnormal already in memory where NVIDIA flushes it (one opcheck
    variant).
  - There is no native shared float atomic. On the Intel UHD the shared
    compare-and-swap loop takes up to three quarters of the 2.8.4 cloth
    normal kernel's time.
- **Timing is noisy:** the GPU clocks down while `pxvk-replay --time`
  uploads between dispatches. Use `--time 200` or more and alternate the
  configurations being compared (`testing.md`).

## AMD

- **Subgroups:** RDNA offers 32 and 64 lanes; GCN always runs 64, two
  warps per subgroup. `PXVK_SUBGROUP_SIZE=64` tests that on RDNA, and the
  Radeon 8060S gives the same results either way.
- **Results:** the same as Intel on every capture set (approximation-unit
  drift only). Sweeps end every cloth and fluid scene as native does
  (`roadmap.md`).
- **Float atomics:** the native atomic flushes subnormals like NVIDIA, and
  the compare-and-swap loop is exact too.
- **Memory limits:** shared memory is 32 KiB, so the cloth solver uses
  storage-backed shared memory.
- **SM count:** pxvk reports the compute-unit count, from
  `VK_AMD_shader_core_properties2`: 40 on the Radeon 8060S (`sms=40`, and
  `cuDeviceGetAttribute` returns 40), its real count. Whether 40 or the
  persona's 16 suits APEX better on AMD is unmeasured: RDNA pairs compute
  units into workgroup processors, so the CU is not a certain match for an
  SM.
- **Warp synchronization:** `VK_KHR_shader_maximal_reconvergence` is
  supported (driver 26.11.1). `pxvk-warptest` never saw lanes out of step,
  with any barrier strategy, at 32 or 64 lanes, and the regression check
  gives the same results with maximal reconvergence as before (November
  2026).
- **Not yet run on AMD hardware:** Arkham Origins.

## lavapipe (Mesa software Vulkan)

- **Subgroups:** one size, its LLVM vector width divided by 32 bits: 8 on
  an AVX2 CPU by default. Above `LP_NATIVE_VECTOR_WIDTH=512` it reports 32
  or 64 lanes but really spans only 16, so warp-level results are wrong
  there.
- **Storage range:** 128 MiB, like most mobile drivers, which exercises the
  split heap.
- **Uses:** testing warp emulation and the split heap without a second GPU
  (`testing.md`).

## Untested GPUs

Default `subgroupSize` reported to the Vulkan hardware database
(vulkan.gpuinfo.org, November 2026). Drivers with subgroup size control may
offer other sizes. A few reports disagree with the rest for the same GPU
(a Mali-G715 at 4, a Mali-G710 at 128); their drivers do not use Arm's
version numbering, so they are taken to be translation layers or modified
drivers.

| Default size | GPUs | What pxvk would need |
|---|---|---|
| 1 | older PowerVR (Rogue GE8300/8320/8322, small A- and B-Series) | warp emulation; no `shfl` or `vote` |
| 4–8 | Mesa software drivers, Arm Mali Bifrost (4 on the first generation, G71/G72; 8 on G52/G76) | warp emulation; probably too slow for GPU PhysX to pay off |
| 16 | Arm Mali from Valhall (2019) to the newest (Immortalis-G925, Mali-G1-Ultra, Mali-G2-Ultra), Broadcom VideoCore (V3D, Raspberry Pi 4 and 5) | warp emulation; 15 APEX kernels fail as on Intel at 16 |
| 32 | NVIDIA, Intel, Apple | — |
| 64 | AMD, Qualcomm Adreno, Samsung Xclipse | whole warps per subgroup, as on GCN |
| 128 | Qualcomm Adreno (some modes), newer PowerVR (B-Series BXM-8/BXT, C-Series, D-Series) | whole warps per subgroup |

Other limits relevant to mobile GPUs (`roadmap.md`, Winlator on Adreno):

- Most report a 128 MiB `maxStorageBufferRange`, which needs the split heap
  and non-uniform storage-buffer indexing.
- 32 KiB of shared memory is common.
- Float-controls support is unknown.

`VK_KHR_shader_maximal_reconvergence` (Khronos, 2023) is reported by:

| Vendor | Drivers |
|---|---|
| NVIDIA | Windows/Linux 538.31 and later (January 2024); Mesa NVK 24.1, back to Kepler |
| AMD | Windows 2.0.299 and later (February 2024); Mesa RADV 24.1, back to GCN |
| Intel | Mesa ANV 24.1 (Gen9 to Arc); Windows 101.6xxx and later on Arc and Arc iGPUs |
| Arm Mali | r49 and later (September 2024); Mesa PanVK 25.1 (Valhall and later) |
| Qualcomm Adreno | 512.762 and later, including Snapdragon X; not Mesa's Turnip, which Winlator often uses |
| Imagination PowerVR | Pixel 10's D-Series and some MediaTek phones (2025) |

Without it pxvk uses memory-only barriers, which rely on the hardware
keeping a warp's lanes together (`architecture.md`).
