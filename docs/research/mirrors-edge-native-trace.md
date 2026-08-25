# Mirror's Edge native NVIDIA trace

## Status

Baseline capture `mirrors-edge-native-trace-001` succeeded against the native NVIDIA driver. It exercised the tutorial from a saved checkpoint through the swing. The user reported possible smoke, but no confirmed cloth, glass, or debris effects.

The trace proves a compact, active **fluid simulation** workload. It does not establish complete Mirror's Edge GPU PhysX coverage. Targeted cloth and debris/glass captures remain required before the compatibility contract can be closed.

## Capture identity

| Item | Value |
|---|---|
| Date | 2026-08-25 |
| Route | Continue saved tutorial; play forward through the swing; exit normally |
| Process result | Exit code 0; observed uptime 3m19s |
| Display | 1280×720, native windowed mode |
| PhysX settings | `PhysXEnhanced=True`; `bDisablePhysXHardwareSupport=False` |
| GPU | NVIDIA GeForce RTX 3070 Laptop GPU |
| NVIDIA driver | 610.88 |
| ZLUDA | v7-preview.9, commit `8a7931be87d758835776f54ea2e33d30621c74ce` |
| ZLUDA archive SHA-256 | `0e537e5bd0b0d22c1cc954bb8e870451645d490d820f65812f05da83b03cac43` |
| `MirrorsEdge.exe` SHA-256 | `58de3df21f40e9953e00be92f7749097832bad02e5b657d8485cdb61f360a3dd` |
| Raw log | 1,926,637 lines; 209,548,001 bytes |
| `log.txt` SHA-256 | `141f7c5f724dbce7e43b972af7923a94a9d74ee731847abfbc4f8a9bf5983395` |
| Raw archive SHA-256 | `330ba517492774b881b4377927b879926ef88e3ce45e072e5ecd0ae2e614a059` |

Command:

```bat
cd /d "C:\GOG Games\Mirror's Edge\Binaries"
C:\programming\homedir\pxvk\tmp\zluda-preview9\zluda\32\zluda.exe --nvidia-trace -- MirrorsEdge.exe
```

Raw evidence is preserved locally under `tmp/evidence/mirrors-edge-native-trace-001/`:

- `raw-trace.tar.gz` — untouched ZLUDA run directory;
- `SHA256SUMS` — per-file hashes;
- `capture-manifest.txt` — command, route, environment, and binary hashes.

The raw modules are proprietary game/runtime material and are intentionally not committed to this repository.

## Runtime preparation

The initial game run reported that hardware PhysX was unavailable. The current NVIDIA PhysX System Software was already installed; the game was loading its obsolete private runtime components first. The following reversible changes forced use of the registered system runtime:

- `PhysXDevice.dll` → `PhysXDevice.dll.bak`
- `PhysXLocal` → `PhysXLocal.bak`
- `PhysXCore.dll` was already present as `PhysXCore.bak`

Afterward, enabling PhysX produced no warning. Relevant hashes:

| Component | SHA-256 |
|---|---|
| Local `PhysXLoader.dll` | `2f8bc8adaabe6deec84cf374b96bbb47823de3d9a870bd8ffa41e4a939023e27` |
| Disabled local `PhysXDevice.dll.bak` | `7daa1ccbbaa71150bdb1e367bf0496de63862ee6f46eb7b2947b788e03d143b9` |
| System `Common\PhysXDevice.dll` | `163a6af59d0235a79b45c6ec9180211f8fa29cf47fc6d53cf9561b9f4f8d5600` |
| System `Engine\v2.8.0\PhysXCore.dll` | `ed8a3f0cf46045699f05437348a1bd589a4e023ade5697c13dbc6c56befa545d` |
| System `Engine\v2.8.0\PhysXCooking.dll` | `1371a94489d6b1ed2fd72138f25d15e4b234ea14028363eef2602d9d8cce9094` |

This runtime selection is part of the oracle definition. Restoring the private DLLs would produce a different workload or fail modern GPU detection.

## Capture result

| Metric | Observed |
|---|---:|
| Top-level traced calls | 1,926,525 |
| Distinct top-level call symbols | 57 |
| CUDA fatbins registered | 17 |
| Extracted PTX modules | 17 |
| Extracted ELF/cubin images | 85 |
| CUDA functions resolved | 70 |
| Distinct kernels launched | 11 |
| Kernel launches | 96,662 |
| Fluid simulation steps | 9,446 |
| Public CUDA failures | 0 |

All 17 extracted PTX modules declare:

```ptx
.version 4.1
.target sm_52
.address_size 32
```

The fatbins each contained one PTX image and five ELF images. Detailed deduplication and construct inventory belong in `mirrors-edge-ptx-corpus.md`.

## Chronology

### 1. NVAPI and CUDA discovery

The process first resolved 243 NVAPI interfaces. Most were queried by the NVAPI loader but never called and therefore are not compatibility requirements.

CUDA initialization then observed:

- `cuInit(0)`;
- one CUDA device;
- device name `NVIDIA GeForce RTX 3070 Laptop GPU`;
- CUDA driver version `13000`;
- compute capability 8.6;
- CUDA context API version `3020`;
- warp size 32;
- unified addressing reported as 0;
- managed memory reported as 0;
- host-memory mapping reported as supported;
- `cuDeviceTotalMem_v2` returned `4294967295`, the maximum 32-bit `size_t` value.

The driver returned 88 device-attribute queries. The complete values remain in the pinned log and will be normalized in `cuda-driver-contract.md`.

### 2. Dark API initialization

No public `cuCtxCreate*` call occurred. The runtime used `cuGetExportTable` six times and created the context through:

```text
{CTX_CREATE_BYPASS}::ctx_create_v2_bypass(flags: 20, dev: 0)
```

Other observed internal tables were:

- `{CUDART_INTERFACE}` — device setup and all 17 fatbin registrations;
- `{CONTEXT_LOCAL_STORAGE_INTERFACE_V0301}` — one context-local value and per-call lookup;
- `{TOOLS_RUNTIME_CALLBACK_HOOKS}` — two initialization buffers;
- `{HEAP_ACCESS}` — one allocation record around the simulation lifetime.

This is a hard compatibility finding: supporting only exported public CUDA functions will not initialize this workload. The required `cuGetExportTable` GUIDs and table behavior must be identified in `cuda-driver-contract.md` and `zluda-runtime.md`.

### 3. Module registration

`{CUDART_INTERFACE}::get_module_from_cubin` registered 17 fatbins. The runtime resolved:

- 70 kernel functions;
- 7 constant/global symbols;
- 33 legacy texture references.

The registered modules cover fluid collision, sorting, density/force, deformable collision, deformable solving, and self-collision. Only the fluid subset launched in this capture.

### 4. Memory pools

The workload allocated exactly two large pools:

| Call | Count | Size |
|---|---:|---:|
| `cuMemHostAlloc` | 1 | 32 MiB |
| `cuMemAlloc_v2` | 1 | 32 MiB |

No additional public CUDA allocation occurred. Suballocations were represented as offsets inside these pools. The same pools were freed during normal shutdown.

### 5. Steady-state fluid loop

There were 9,448 `cuCtxSynchronize` calls: two during setup and one after each of 9,446 active fluid steps.

Every active step launched the same nine-kernel spine:

| Kernel stem | Calls | Grid X | Block X | Dynamic shared memory |
|---|---:|---:|---:|---:|
| `KernelFluidCalcHashPacked` | 9,446 | 1–5 | 64 | 0 |
| `radixSortBlocks<32,0,false>` | 9,446 | 1 | 256 | 4096 B |
| `KernelReorderPosVelAndFindCellStartPacked` | 9,446 | 1–5 | 64 | 260 B |
| `KernelFluidDensity` | 9,446 | 1–2 | 192 | 0 |
| `KernelFluidForce` | 9,446 | 1–3 | 128 | 0 |
| `PxgCudaFluidCollisionConstraintsBatch` | 9,446 | 1–3 | 128 | 0 |
| `PxgCudaFluidCollisionDynamicResponseBatch` | 9,446 | 1–3 | 128 | 0 |
| `PxgCudaFluidCollisionTrianglesBatch` | 9,446 | 3–35 | 64 | 0 |
| `PxgCudaFluidCollisionStaticResponseBatch` | 9,446 | 1–3 | 128 | 0 |

Variable collision-shape work added:

| Kernel stem | Calls | Grid X | Block X |
|---|---:|---:|---:|
| `PxgCudaFluidCollisionBoxShapesBatch` | 10,730 | 2–6 | 64 |
| `PxgCudaFluidCollisionCapsuleShapesBatch` | 918 | 2 or 4 | 64 |

All launches used:

- grid Y/Z and block Y/Z equal to 1;
- the legacy default stream (`hStream: 0x0`);
- `kernelParams: NULL`;
- `extra: 0x1`.

The trace therefore does **not** contain the packed launch-parameter data. Recovering that data is a blocker for a faithful implementation and requires either a targeted tracer extension or a source-grounded understanding of the observed CUDART internal interface.

### 6. Transfers and texture binding

| Operation | Calls | Total | Per-call range |
|---|---:|---:|---:|
| `cuMemcpyHtoDAsync_v2` | 120,091 | 89.220 MiB | 8–5120 B |
| `cuMemcpyDtoDAsync_v2` | 18,892 | 18.449 MiB | exactly 1024 B |
| `cuMemcpyDtoHAsync_v2` | 47,230 | 103.233 MiB | 256–5120 B |
| `cuMemsetD8_v2` | 9,448 | 2 MiB per call | 2 MiB |

All transfers used the default stream. The trace records addresses and byte counts, not copied payload contents.

The workload repeatedly rebound legacy linear texture references. High-frequency calls were:

| Call | Count |
|---|---:|
| `cuTexRefSetAddress_v2` | 283,380 |
| `cuMemGetAddressRange_v2` | 94,460 |
| `cuTexRefSetFormat` | 94,460 |
| `cuTexRefSetFlags` | 94,460 |
| `cuTexRefSetFilterMode` | 94,460 |
| `cuTexRefSetMipmapFilterMode` | 94,460 |
| `cuTexRefSetMipmapLevelBias` | 94,460 |
| `cuTexRefSetMipmapLevelClamp` | 94,460 |
| `cuTexRefSetMaxAnisotropy` | 94,460 |
| `cuTexRefSetAddressMode` | 94,460 |

Observed bindings used linear device addresses, point filtering, and clamp address mode. No CUDA graphics-interoperability calls, arrays, surface objects, streams, or events appeared.

### 7. Shutdown

The run exited normally in this order:

1. final device-to-host copies;
2. `cuCtxSynchronize`;
3. `cuMemFreeHost`;
4. `cuMemFree_v2`;
5. `cuCtxDetach`;
6. internal heap-record release.

## Observed public CUDA surface

The active public driver API was limited to:

- initialization and device queries: `cuInit`, `cuDriverGetVersion`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute`;
- context access: `cuCtxGetCurrent`, `cuCtxGetApiVersion`, `cuCtxGetDevice`, `cuCtxSynchronize`, `cuCtxDetach`;
- module lookup: `cuModuleGetFunction`, `cuModuleGetGlobal_v2`, `cuModuleGetTexRef`;
- memory: `cuMemHostAlloc`, `cuMemAlloc_v2`, `cuMemGetAddressRange_v2`, `cuMemsetD8_v2`, three asynchronous copy directions, and the two matching frees;
- legacy texture references: address, format, flags, filter, mipmap, anisotropy, and address-mode setters;
- execution: `cuLaunchKernel`;
- internal interface discovery: `cuGetExportTable`.

Every observed public CUDA call returned `CUDA_SUCCESS`.

## Diagnostics

The trace contained four non-success diagnostics during initialization:

- two expected context-local lookups returned `CUDA_ERROR_INVALID_HANDLE` before the value was installed;
- two export tables produced tracer diagnostics: `Table length mismatch. Expected: 4, got: 3`.

NVAPI returned `-9` from `NvAPI_GPU_CudaEnumComputeCapableGpus` three times, but fallback discovery succeeded and CUDA execution proceeded. No module extraction or public CUDA execution failure was recorded.

## What this establishes

For the exercised fluid effect, the implementation target is much smaller than general CUDA:

- one x86 process and one device;
- one context created through a specific Dark API table;
- one 32 MiB pinned host pool and one 32 MiB device pool;
- one default stream with explicit synchronization per simulation step;
- legacy linear texture references;
- 11 launched kernels from PTX 4.1, `sm_52`, 32-bit address modules;
- small grids, one-dimensional blocks, and at most 4096 B dynamic shared memory;
- no observed D3D/CUDA interop.

These are observations, not yet architecture decisions.

## Missing evidence and required follow-up

This baseline does not close native trace research:

1. **Launch parameters:** every launch used `kernelParams: NULL` and opaque `extra: 0x1`; argument bytes were not captured.
2. **Transfer payloads:** copy contents were not captured.
3. **Cloth/deformables:** deformable modules loaded, but no deformable kernel launched.
4. **Glass/debris:** the route did not visibly exercise them.
5. **Coverage:** a short tutorial segment cannot establish all Mirror's Edge GPU PhysX paths.
6. **Timing:** ZLUDA tracing serializes and logs calls; this trace is not performance evidence.

Before deriving the final CUDA contract, collect targeted traces for:

- a confirmed cloth/deformable scene;
- a confirmed glass/debris scene;
- the same scene with enhanced PhysX disabled as a control;
- launch-parameter bytes, using a narrowly extended tracer if source study cannot recover them.
