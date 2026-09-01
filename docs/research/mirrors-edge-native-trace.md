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

## Project-owned tracer follow-up

The ZLUDA trace exposed launch geometry but rendered the packed `extra` list as
only its first token (`0x1`). A project-owned x86 proxy tracer now lives in
`tools/nvcuda-trace/`. It is C11, builds with MinGW through CMake, and does not
modify the game installation.

Build:

```bat
cmake -S tools/nvcuda-trace -B build/nvcuda-trace -G "MinGW Makefiles" ^
  -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc ^
  -DCMAKE_MAKE_PROGRAM=mingw32-make
cmake --build build/nvcuda-trace --parallel
```

Run:

```bat
cd /d "C:\GOG Games\Mirror's Edge\Binaries"
C:\programming\homedir\pxvk\build\nvcuda-trace\pxvk-trace.exe ^
  --output C:\path\to\trace-output ^
  -- MirrorsEdge.exe -windowed -ResX=1024 -ResY=576
```

`pxvk-trace.exe` starts the target suspended, injects its sibling `nvcuda.dll`,
then resumes and waits for the target. The proxy forwards into the native x86
NVIDIA driver. After `cuInit`, it patches loaded modules' `GetProcAddress` IAT
entries so requests made against the real driver handle are redirected to
matching tracer wrappers. Unknown CUDA functions remain native pass-throughs.
This was necessary because PhysX queried 190+ driver entry points and otherwise
bypassed an application-local proxy.

The successful verification capture produced:

| Artifact | Result |
|---|---:|
| Trace log | 27,216 lines; 6,341,335 bytes |
| Raw fatbins | 17 |
| Kernel launches | 26,898 |
| Distinct launch-parameter samples | 11 |
| Public CUDA failures | 0 |

The captured packed-buffer sizes were:

| Kernel stem | Bytes |
|---|---:|
| `KernelFluidCalcHashPacked` | 52 |
| `radixSortBlocks<32,0,false>` | 20 |
| `KernelReorderPosVelAndFindCellStartPacked` | 76 |
| `KernelFluidDensity` | 84 |
| `KernelFluidForce` | 100 |
| Fluid collision constraints/dynamic response/triangles/static response | 20 each |
| Fluid box-shape collision | 20 |
| Fluid capsule-shape collision | 20 |

The first buffer for each `(CUfunction, size)` pair is saved as a binary file.
Subsequent launches log the same parsed size without duplicating the sample.
The successful run is preserved locally at
`tmp/evidence/mirrors-edge-custom-trace-001/raw-trace.tar.gz`, SHA-256
`9704e2426d0dc53adac26d2605e43c410ce458efe9402e6f516f9235771d0c42`.

### Interactive window mode

Mirror's Edge ignored its generated `Fullscreen=False` setting and `-windowed`
argument during Direct3D device creation. Exclusive-mode focus changes also
rearranged desktop icons. For interactive capture, this workstation now uses
[dxwrapper v1.8.8600.25](https://github.com/elishacloud/dxwrapper/releases/tag/v1.8.8600.25)
as a reversible local D3D9 wrapper:

```text
Binaries\d3d9.dll  <- dxwrapper.dll
Binaries\d3d9.ini  <- dxwrapper.ini
```

Relevant settings:

```ini
EnableD3d9Wrapper        = 1
EnableWindowMode         = 1
WindowModeBorder         = 1
SetInitialWindowPosition = 1
InitialWindowPositionLeft = 100
InitialWindowPositionTop  = 100
FullscreenWindowMode     = 0
```

The wrapper log confirmed a bordered window positioned at `(100,100)`, and the
user confirmed normal window and focus behavior. Remove `d3d9.dll` and
`d3d9.ini` to restore the original renderer. Future trace manifests must record
whether this graphics wrapper was present; it is operational tooling, not a
pxvk runtime dependency.

## Targeted effect captures

Scene switching used local `TdInput.ini` bindings that issue Unreal `start`
commands only after an ordinary level has loaded:

```ini
F3 = start Subway_p?LoadCheckpoint=Subway_Station?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE
F4 = start Scraper_p?LoadCheckpoint=Server_room?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE
F11 = start Escape_p?LoadCheckpoint=Office?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE
```

### Subway station: deformable workload

F3 loaded the subway-station checkpoint. The user did not identify an obvious
cloth object, but the trace conclusively exercised a deformable simulation for
76 steps:

| Kernel | Calls | Grid X | Block X | Shared memory | Parameter bytes |
|---|---:|---:|---:|---:|---:|
| `PxgCudaDeformableUpdate` | 76 | 28 | 64 | 0 | 12 |
| `KernelDeformableUberCollision` | 76 | 30 | 64 | 0 | 4 |
| `KernelDeformableUberSolver<512>` | 76 | 6 | 64 | 7072 B | 16 |

The complete run recorded 44,537 launches, 6,563 synchronizations, 17 fatbins,
and 15 distinct launch-parameter samples with no nonzero CUDA result. Its
44,855-line log is 10,766,827 bytes. The archive is preserved locally at
`tmp/evidence/mirrors-edge-cloth-trace-001/raw-trace.tar.gz`, SHA-256
`40ac4732de8bdd7a392aaa36fcf309cda97e931025dadef8fca946fd10dc1ff9`.

This establishes the deformable kernel path, not visual cloth identification.

### Server room: confirmed glass workload

F4 loaded the server-room checkpoint. The user confirmed that glass was present
and that both the player and police gunfire broke it. Persistent shards and
smoke were not observed carefully enough to claim.

The run recorded 5,762 launches, 641 synchronizations, 17 fatbins, and 13
distinct launch-parameter samples with no nonzero CUDA result. Its 6,080-line
log is 1,412,649 bytes. Compared with the tutorial fluid baseline it added or
materially expanded:

| Kernel stem | Calls | Notable observation |
|---|---:|---|
| `KernelFluidSimpleParticle` | 514 | 44-byte parameter buffer; grid X 2–10 |
| Fluid convex-shape collision | 476 | 20-byte parameters; grid X 5–34; block X 64 |
| Fluid capsule-shape collision | 512 | 20-byte parameters |
| Fluid box-shape collision | 1,088 | largest collision-call family |
| Fluid triangle collision | 638 | grid X reached 94 |

The archive is preserved locally at
`tmp/evidence/mirrors-edge-glass-trace-001/raw-trace.tar.gz`, SHA-256
`eb68cc59f35cbddb0e3e125c56b6c7963b9c7ef9339580e4a84f0548072a1c29`.

The trace confirms the workload associated with a visibly broken-glass scene.
It does not prove which launches simulate glass fragments versus nearby smoke
or other particles; that correlation needs an enhanced-PhysX-off control or a
more focused trigger.


## Missing evidence and required follow-up

This baseline does not close native trace research:

1. **Launch parameters beyond the first sample:** buffer sizes and one sample per observed kernel are captured; dynamic changes across later launches are not preserved.
2. **Transfer payloads:** copy contents were not captured.
3. **Visual cloth identity:** the deformable path is proven, but the simulated object was not identified visually.
4. **Glass attribution:** glass destruction is confirmed, but persistent shards and smoke were not separately observed.
5. **Control run:** the same server-room checkpoint has not been captured with enhanced PhysX disabled.
6. **Coverage:** these three short scenes cannot establish every Mirror's Edge GPU PhysX path.
7. **Timing:** tracing changes execution and is not performance evidence.

Before deriving the final CUDA contract, collect targeted traces for:

- the server-room checkpoint with enhanced PhysX disabled as a control;
- a visually confirmed cloth object if deterministic visual validation is needed;
- additional launch samples only if later traces show a new function or packed-buffer size.
