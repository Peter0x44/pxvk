# Vulkan requirements for the Mirror's Edge corpus

## Status

The captured Mirror's Edge workload fits a narrow Vulkan 1.2 compute target.
The first backend does not need CUDA-generic addressing, sampled images,
subgroup operations, atomics, buffer device addresses, or the Vulkan memory
model extension.

The hard device requirements are:

- a Vulkan 1.2 x86 loader path and a compute queue;
- `shaderInt64`;
- f32 signed-zero/Inf/NaN preservation plus RTE control;
- at least 256 workgroup invocations in X;
- at least 10,756 bytes of workgroup memory;
- a storage-buffer range and allocation of at least 32 MiB;
- at least 100 bytes of push constants.

The installed SDK and x86 loader reach both local GPUs through Volk. Both GPUs
clear the integer, workgroup, memory, and rounding requirements. The important
exception is f32 denormal handling: Intel advertises
`shaderDenormFlushToZeroFloat32`, while the RTX 3070 does not. Because the RTX
3070 is the native reference system, native Vulkan FTZ cannot be a hard device
requirement. The translator needs a software FTZ path whose output is checked
against the native CUDA oracle.

This is a requirements decision, not the permanent runtime or compiler
architecture.

## Evidence boundary

This study combines:

- the 16 launched PTX entries and 120 launched opcode variants cataloged in
  [mirrors-edge-ptx-corpus.md](mirrors-edge-ptx-corpus.md);
- the real allocation, launch, copy, and synchronization behavior in
  [cuda-driver-contract.md](cuda-driver-contract.md);
- exact NVIDIA f32, `bar.sync 0`, and linear-texture results in
  [native-oracle-probes.md](native-oracle-probes.md);
- Vulkan properties queried from a PE32/i386 process on the reference
  workstation.

It does not establish requirements for unlaunched PhysX entries, other games,
D3D/CUDA interop, warp intrinsics, or general CUDA.

## Reproducible x86 capability and memory probes

The probe is implemented at:

```text
tools/vulkan-probe/
├── CMakeLists.txt
├── main.c
├── memory_smoke.cpp
├── oracle.comp / oracle.spvasm / oracle.spv
├── preflight.ps1
├── rebuild-oracle.ps1
└── expected/
```

CMake FetchContent retrieves pinned Volk, Vulkan-Headers, and SPIRV-Headers
commits; no Vulkan SDK path or import library is required. Build and run
manually:

```bat
cmake -S tools/vulkan-probe -B build/vulkan-probe -G "MinGW Makefiles" ^
  -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc ^
  -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ ^
  -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build/vulkan-probe --parallel
build\vulkan-probe\vulkan-probe.exe vulkan-x86-capabilities.json
build\vulkan-probe\vulkan-memory-smoke.exe vulkan-memory-smoke.json
build\vulkan-probe\vulkan-memory-smoke.exe vulkan-oracle.json build\vulkan-probe\oracle.spv
```

The process reported `pointer_bits: 32`, Vulkan header version 1.4.357, and
loader API version 1.4.357. Two executions produced byte-identical JSON. `jq`
parsed the result and confirmed both devices report `shaderInt64`.

The normalized fetched-dependency output remains local at
`tmp/vulkan-preflight-fetch/capabilities.json`, SHA-256
`c6dffac9696fbb0a2165a0ca14505386e06d6a54969f884669b89c24225166dd`.

The generated integer/global-memory shader then allocated one 32 MiB
device-local buffer and one 32 MiB host-visible staging buffer per device.
Intel and RTX both returned exact `0xc7b13c62` while preserving the input guard.
The 123-word module passed `spirv-val --target-env vulkan1.2`; three runs
produced byte-identical JSON. The committed result is
`tools/vulkan-probe/expected/reference-workstation/memory-smoke.json`, SHA-256
`4527312167c098277ba0422cfcd87d48a5578fdeeea3883234dbac0bbed2d2a0`.

The combined 64-thread oracle then matched all committed native CUDA math,
barrier, and integer-texture fields on both devices. Intel exposed one
load-bearing SPIR-V rule: GLSL.std.450 `Fma` returned the separately rounded zero
until its result was decorated `NoContraction`; with the decoration Intel and
RTX both returned native `0xa8800000`. The combined result is
`tools/vulkan-probe/expected/reference-workstation/oracle.json`, SHA-256
`42a8778edb77a7a74429e30fc2ec68c87c677ac0b9ca51a604539b07a8953968`.

The direct compiler slice now reproduces that result from
`tools/cuda-oracle/probe.ptx`: the frontend emits schema-4 structured IR and
the project emitter produces a deterministic 2,605-word SPIR-V 1.5 binary.
SDK validation passes, and the generated module matches every math, barrier,
and texture field on Intel and RTX. The original GLSL/assembly oracle remains
the independent Vulkan-side reference rather than compiler input.

### Observed device results

| Property | Intel UHD Graphics | RTX 3070 Laptop |
|---|---:|---:|
| Device Vulkan version | 1.4.323 | 1.4.341 |
| Compute queues | 1 | 24 |
| Max workgroup invocations | 1,024 | 1,024 |
| Max workgroup size X | 1,024 | 1,024 |
| Max workgroup count X | 65,536 | 2,147,483,647 |
| Max workgroup shared memory | 32,768 B | 49,152 B |
| Max storage-buffer range | 1,073,741,820 B | 4,294,967,295 B |
| Max push constants | 256 B | 256 B |
| `shaderInt64` | yes | yes |
| storage-buffer 8/16-bit access | yes | yes |
| `robustBufferAccess2` | yes | yes |
| `vulkanMemoryModel` | yes | yes |

A later submitted AMD PE32 run also passed:

| Property | AMD Radeon 8060S |
|---|---:|
| Vendor / device ID | `0x1002` / `0x1586` |
| Device Vulkan version | 1.4.349 |
| Compute queues | 16 |
| Max workgroup invocations / size X | 1,024 / 1,024 |
| Max workgroup count X | 4,294,967,295 |
| Max workgroup shared memory | 32,768 B |
| Max storage-buffer range | 4,294,967,295 B |
| Max allocation / buffer | 2,147,483,648 B |
| Max push constants | 256 B |
| `shaderInt64` and storage-buffer 8/16-bit access | yes |
| f32 preserve / FTZ / RTE / RTZ | yes / yes / yes / yes |

The redistributable AMD fixtures are preserved under
`tools/vulkan-probe/expected/amd-radeon-8060s-device1586-driver8389003/`.
The raw capabilities SHA-256 is
`f35d24ad1404a2267e2dd5853466ef63960fb0de330b891d2de6c949c503fa7d`;
the normalized preflight SHA-256 is
`9989dc0a20d043d234a9c6b8e85a668571e16fa1c30bdc9a20a22667a34430d4`.

These are observed feasibility results, not portable minimums. Startup must
still query and reject a device that misses a hard requirement.

## Required, preferred, and rejected Vulkan capabilities

Vulkan 1.2 promotes several named extensions into core. On a 1.2 device the
runtime uses the core feature/property structs; it does not need to enable the
legacy extension name. SPIR-V capabilities still require the corresponding
feature or property to be supported.

| Capability or extension path | Policy | Enablement and workload reason |
|---|---|---|
| Vulkan 1.2 compute | required baseline | Supplies SPIR-V 1.5, properties2, storage-buffer storage class, and shader float-control properties |
| Core `shaderInt64` / SPIR-V `Int64` | required for real kernels | Enable `VkPhysicalDeviceFeatures::shaderInt64`; launched PTX uses `u64`/`s64`, `clz.b64`, 64-bit shifts, comparisons, and arithmetic |
| `VK_KHR_shader_float_controls` promoted to 1.2 | required properties | Require float32 RTE and signed-zero/Inf/NaN preservation; emit `RoundingModeRTE 32` and `SignedZeroInfNanPreserve 32` |
| `DenormFlushToZero 32` | preferred per device | Use only where `shaderDenormFlushToZeroFloat32` is true; retain software FTZ because the RTX reports false |
| `VK_KHR_8bit_storage` promoted to 1.2 | preferred | Enable `storageBuffer8BitAccess` for exact `ld.global.u8`/`ld.const.u8`; packed `u32` extraction is the fallback |
| `VK_KHR_16bit_storage` promoted to 1.1 | preferred | Enable `storageBuffer16BitAccess` for `ld.global.u16` and vector forms; packed `u32` extraction is the fallback |
| `VK_KHR_workgroup_memory_explicit_layout` | preferred after feature probing | All three devices advertise the extension. Its 8/16-bit access features can represent the PTX shared byte arena and `ld/st.shared.u8` directly; the probe must record those feature bits before use. Packed `u32` shared storage remains the fallback |
| `VK_EXT_subgroup_size_control` | conditional | All three devices can request compute subgroup size 32. Use only if a captured kernel gains warp/subgroup operations; the current corpus has none |
| `VK_KHR_vulkan_memory_model` | optional | All three devices support it, but current workgroup barriers and Vulkan queue dependencies cover the observed workload |
| `VK_KHR_shader_float_controls2` | optional, currently unused | All three devices support it. Strict PTX lowering uses explicit execution modes and `NoContraction`; enabling fast-math defaults would weaken compatibility |
| `VK_KHR_shader_fma` | not a baseline | Intel does not advertise it. `GLSL.std.450 Fma` plus `NoContraction` already matches the native oracle |
| `VK_KHR_synchronization2` and timeline semaphores | optional runtime optimization | Can simplify submission/dependency code later; they do not change PTX shader semantics |
| `robustBufferAccess2` | not required | Texture OOB needs an explicit zero result; invalid CUDA global accesses are not a robustness contract |
| Buffer device address | rejected for the first runtime | PTX is `.address_size 32`; device pointers remain 32-bit offsets into one bound heap |
| Sampled images and samplers | rejected for current textures | The observed one-dimensional point fetch requires explicit OOB zero, implemented by bounded storage-buffer loads |
| Float64, atomics, graphics queues, external-memory interop | not required | No launched PTX requires them; graphics interop remains outside the charter |

This is a feature-driven policy, not a vendor allowlist. Intel, RTX, and AMD
currently support the preferred 8/16-bit storage, subgroup-size, Vulkan-memory-
model, and float-controls2 paths. A module's cache key must include every
enabled lowering path because the emitted types, capabilities, and instructions
can differ.

## Quantitative limits

| Limit | Required value | Evidence | Local minimum |
|---|---:|---|---:|
| `maxComputeWorkGroupInvocations` | at least 256 | Largest captured launch; 512 is preferred headroom for a launched entry's `.maxntid` | 1,024 |
| `maxComputeWorkGroupSize[0]` | at least 256 | All captured launches and built-ins are X-only | 1,024 |
| `maxComputeWorkGroupCount[0]` | at least 94 | Largest captured grid X | 65,536 |
| `maxComputeSharedMemorySize` | at least 10,756 B | Maximum static plus observed dynamic memory for one launched entry | 32,768 B |
| `maxStorageBufferRange` | at least 33,554,432 B | The workload creates one 32 MiB CUDA device pool | 1,073,741,820 B |
| maximum memory allocation/buffer size | at least 33,554,432 B | Same device-pool requirement | 4,294,901,760 B |
| `maxPushConstantsSize` | at least 100 B | Largest packed kernel parameter block | 256 B |
| compute-stage storage buffers | at least 1 | Candidate lowering uses one byte-addressed device heap | 200 |

The native CUDA device query returned 1,024 threads per block, while the
largest captured launch is 256 and the largest launched-entry `.maxntid` is
512. The Vulkan hard gate follows executed work, not the emulated CUDA
attribute or a declaration's maximum. A device below 512 must reject an
unobserved launch above its real limit; supporting such a launch would first
promote the 512 value into the hard gate.

The 108-byte maximum PTX `.local` declaration maps to function-private shader
storage and is not a Vulkan device-limit gate. Descriptor offset alignment is
also not a heap-address gate when the 32 MiB descriptor is bound at offset zero
and CUDA addresses remain shader-side byte offsets.

The observed shared-memory maxima are per entry. They must not be summed across
kernels. Dynamic shared memory can be represented by a specialization-sized
Workgroup array or another validated per-pipeline scheme; that choice remains
implementation design.

## PTX-to-Vulkan mapping constraints

### Address and memory model

The minimum candidate representation is a 32 MiB storage buffer matching the
CUDA device pool. A PTX device address is a `u32` byte offset from that buffer's
base. This preserves 32-bit wrap and comparison behavior and avoids host
pointer width, SPIR-V physical pointers, and buffer device address.

Memory operations lower as follows:

| PTX space/form | Vulkan/SPIR-V candidate |
|---|---|
| `.global` | storage-buffer load/store at a checked/aligned byte offset |
| `.const` and module globals | read-only/module-state ranges in buffer storage |
| `.shared` and `.extern .shared` | Workgroup variables |
| `.local` | Function variables, with compiler spills left to the Vulkan driver |
| `.param` | push constants preserving the captured byte layout |
| `u8` / `u16` memory | `u32` load plus shift/mask; native narrow storage is an optional optimization |
| `u64` memory/arithmetic | `shaderInt64`, with byte layout preserved explicitly |

A staging allocation supplies host upload/download behavior. Host coherency is
not assumed: flush or invalidate non-coherent mapped ranges according to the
selected memory type.

### Synchronization

`bar.sync 0` lowers to `OpControlBarrier` with:

- execution scope `Workgroup`;
- memory scope `Workgroup`;
- workgroup-memory semantics with acquire/release ordering.

The native 64-thread oracle proves both full-workgroup rendezvous and shared
write visibility. No subgroup, device-scope, atomic, or cross-workgroup
primitive is justified.

CUDA default-stream ordering also requires dependencies between Vulkan
dispatches. The conservative first path inserts a compute-write to
compute-read/write memory barrier between launches that may share the device
pool. Khronos' synchronization example uses compute stage plus shader write on
the source and compute stage plus shader read on the destination. Hazard-based
barrier elision can come only after correctness.

Copies require transfer/compute dependencies around the staging and device
buffers. `cuCtxSynchronize` completes a submission and waits on a fence before
host reads or returns synchronization success.

Neither `VK_KHR_synchronization2` nor `vulkanMemoryModel` is essential: Vulkan
1.2 legacy pipeline barriers and the GLSL450 SPIR-V memory model can express the
required dependencies. Synchronization2 is preferred when available because
its stage/access pairs are clearer.

### Legacy linear textures

The corpus uses only:

```text
tex.1d.v4.f32.s32
tex.1d.v4.u32.s32
```

The native unsigned probe established point reads from a linearly bound range:
negative and high signed coordinates return an all-zero vector. Edge-clamped
sampled-image behavior is therefore wrong.

The first candidate lowering is a storage-buffer fetch with:

1. a signed coordinate bounds check against the bound element count;
2. an all-zero result when out of range;
3. four exact 32-bit component loads when in range;
4. bit reinterpretation for the f32 form, pending the deferred float-texture
   native probe.

This needs texture binding metadata but no Vulkan image, sampler, texel-buffer,
or format feature. The metadata layout belongs to implementation design.

### Floating point

The native CUDA fixture makes these observable requirements:

- `.ftz` input and result denormals become zero;
- `fma.rn.ftz` remains one fused operation;
- separate `mul.ftz` and `add.ftz` remain separately rounded;
- integer conversions preserve RTZ and RTE behavior;
- `min`/`max` preserve the observed NVIDIA NaN operand behavior;
- `rcp.approx`, `sqrt.approx`, and `rsqrt.approx` match exact native bits or an
  explicitly accepted error envelope.

SPIR-V provides the relevant controls, but the device properties decide which
execution modes may be requested. The observed device results are decisive:

| f32 property | Intel | RTX 3070 | AMD 8060S |
|---|---:|---:|---:|
| signed-zero/Inf/NaN preserve | yes | yes | yes |
| denorm preserve | yes | no | yes |
| denorm flush to zero | yes | no | yes |
| rounding mode RTE | yes | yes | yes |
| rounding mode RTZ | yes | yes | yes |
| shader float controls 2 | yes | yes | yes |

Therefore:

- request `DenormFlushToZero 32` only where advertised;
- on the RTX path, explicitly flush subnormal operands/results around every
  translated `.ftz` operation before accepting a kernel;
- lower explicit FMA through `GLSL.std.450 Fma` decorated with
  `NoContraction`, which makes it one operation;
- decorate each separate floating multiply/add with `NoContraction` and do not
  enable fast-math reassociation;
- implement ordered selects/helpers where a Vulkan min/max instruction does not
  guarantee the native NaN operand result;
- treat approximate math as an oracle-validated lowering, not a capability bit.
- lower `cvt.rzi` through an explicit truncation operation rather than changing
  the entry point's default RTE mode.

`cvt.rmi.ftz.f32.f32` does not create a hardware round-toward-negative
requirement in this corpus: source and destination have the same f32 width, so
the observable work is the `.ftz` normalization. This must still be verified by
kernel differential tests.

Software FTZ is the largest feasibility risk. The property result proves that
an execution-mode-only design cannot support the reference RTX path; it does
not yet prove that the proposed helper placement reproduces every intermediate.

## Subgroup size 32

The x86 probes reported:

| Property | Intel | RTX 3070 | AMD 8060S |
|---|---:|---:|---:|
| Default subgroup size | 32 | 32 | 64 |
| Minimum / maximum | 8 / 32 | 32 / 32 | 32 / 64 |
| Required-size support in compute | yes | yes | yes |
| `subgroupSizeControl` | yes | yes | yes |

This is useful headroom, not a current hard requirement. The PTX corpus has no
`vote`, `shfl`, `match`, `%laneid`, `%warpid`, or lane-mask use. `bar.sync 0` is
a workgroup operation and must not be lowered to a subgroup barrier.

The CUDA API surface must still report warp size 32, as required by the project
charter. That emulated device attribute does not require the Vulkan device's
subgroup size to be 32 when generated shaders contain no subgroup operations.
A future evidenced warp instruction would reopen the deferred subgroup design
and may make required subgroup size 32 a pipeline/device gate.

## Startup gate

A candidate device passes the Mirror's Edge gate only when all of these are
true:

1. Vulkan API version is at least 1.2 and a compute queue exists.
2. `shaderInt64`, f32 signed-zero/Inf/NaN preservation, and RTE are true.
3. the quantitative workgroup, shared-memory, storage-range, allocation, and
   push-constant limits meet the table above.
4. either f32 denorm FTZ is advertised or the validated software-FTZ lowering
   is enabled.
5. one 32 MiB device buffer plus required staging/module state can be allocated
   from compatible memory types.

The gate must produce a concrete diagnostic naming the first missing feature or
limit. It must not silently choose a semantically weaker floating-point mode.

## Acceptance gates and next implementation slice

The Vulkan research phase can proceed despite the deferred native probes. The
first implementation slice should remain small:

1. open the x86 Vulkan loader through Volk and apply the startup gate;
2. create one compute queue, command pool, device heap, and staging path;
3. run a generated SPIR-V microkernel for integer/global-memory addressing;
4. run the existing f32, barrier, and integer-texture oracle cases through
   Vulkan and compare outputs.

Do not accept translated real kernels until:

- the standalone Vulkan differential classifies every oracle mismatch;
- the deferred float-texture native probe fixes the f32 fetch contract;
- deterministic fluid and deformable input/output fixtures exist for real
  kernel differential testing.

The remaining queried-only export, NVAPI, threading, and PhysX-off trace probes
are integration or hardening gates, not Vulkan device-capability blockers.

## Primary references

Checked 2026-08-25:

- [Vulkan `VkPhysicalDeviceLimits`](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceLimits.html)
- [Vulkan float-control properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceFloatControlsProperties.html)
- [`VK_KHR_shader_float_controls2`](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_shader_float_controls2.html)
- [Vulkan subgroup properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupProperties.html)
- [Vulkan subgroup-size-control properties](https://docs.vulkan.org/refpages/latest/refpages/source/VkPhysicalDeviceSubgroupSizeControlProperties.html)
- [`vkCmdDispatch`](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdDispatch.html)
- [Khronos synchronization examples](https://docs.vulkan.org/guide/latest/synchronization_examples.html)
- [SPIR-V 1.6 unified specification](https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html)
- [GLSL.std.450 extended instructions](https://registry.khronos.org/SPIR-V/specs/1.0/GLSL.std.450.html)
