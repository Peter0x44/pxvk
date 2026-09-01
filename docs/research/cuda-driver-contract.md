# Mirror's Edge CUDA driver contract

## Status

The captured fluid, deformable, and glass workloads query 196 public CUDA
symbols but execute only 35. They also depend on six `cuGetExportTable` Dark
API tables and on a narrow NVAPI hardware-detection path. This document defines
the observed x86 ABI and behavior; queried-only CUDA functionality is not
promoted into implementation scope.

The matching PhysX 2.8.0 SDK samples are secondary integration evidence. They
do not promote arbitrary CUDA, but they do prevent title-specific observations
from becoming runtime invariants where the same PhysX runtime demonstrably uses
another valid path.

Normalized evidence is preserved locally as
`tmp/evidence/cuda-driver-contract-analysis.json`, SHA-256
`ce60bf0516aab22e79b69cdf0276bd842069a1103a39ef3f8741551bd0df4a5d`.
The source traces and archive hashes are recorded in
`mirrors-edge-native-trace.md`.

## Boundary before CUDA: NVAPI and PhysX selection

Mirror's Edge initially reported that hardware PhysX was unavailable while its
obsolete private `PhysXDevice.dll` was present. Removing the private device DLL
and `PhysXLocal` directory made the game use the registered NVIDIA PhysX System
Software; the warning disappeared and CUDA work began.

The native startup trace resolved 243 NVAPI interfaces, but only this small
subset was called:

| Call | Count | Native result |
|---|---:|---|
| `NvAPI_Initialize` | 9 | `0` |
| `NvAPI_GetDisplayDriverVersion` | 1 | `0` |
| `NvAPI_EnumLogicalGPUs` | 1 | `0` |
| `NvAPI_EnumPhysicalGPUs` | 1 | `0` |
| `NvAPI_Stereo_IsEnabled` | 1 | `-219` |
| `NvAPI_SYS_GetDriverAndBranchVersion` | 3 | `0` |
| `NvAPI_GPU_CudaEnumComputeCapableGpus` | 3 | `-9` |
| `NvAPI_GetGPUIDfromPhysicalGPU` | 3 | `0` |
| `NvAPI_GetPhysicalGPUFromGPUID` | 1 | `0` |
| private interface `0x36e39e6b` | 6 | `0` |
| private interface `0x84244524` | 4 | `0` |
| private interface `0xcb7b09d2` | 4 | `0` |

`NvAPI_GPU_CudaEnumComputeCapableGpus` returning `-9` did not disable CUDA; the
runtime continued through private/fallback queries and enumerated the RTX 3070.

A non-NVIDIA pxvk target therefore needs one of:

1. a minimal `nvapi.dll` compatibility shim for this observed detection path;
2. a source-grounded alternate PhysX device-selection path.

Silently omitting NVAPI is not valid: the user-visible hardware-PhysX gate
occurs before the CUDA workload. ZLUDA's 32-bit package also contains an NVAPI
shim, so its exact implementation belongs in `zluda-runtime.md`. General NVAPI
support remains out of scope.

## x86 public ABI

The process is PE32/i386. CUDA's `CUDAAPI` is Windows `__stdcall` and exports
use undecorated names. Relevant type widths are:

| Type | x86 representation |
|---|---|
| `CUresult` | 32-bit enum/integer |
| `CUdevice` | signed 32-bit integer |
| `CUdeviceptr` | unsigned 32-bit integer |
| `size_t` | unsigned 32-bit integer |
| `CUcontext`, `CUmodule`, `CUfunction`, `CUtexref`, `CUstream` | opaque 32-bit handles/pointers |
| `CUuuid` | 16 raw bytes |

The 35 active exported signatures are:

```c
CUresult __stdcall cuInit(unsigned int flags);
CUresult __stdcall cuDriverGetVersion(int *version);
CUresult __stdcall cuDeviceGetCount(int *count);
CUresult __stdcall cuDeviceGet(CUdevice *device, int ordinal);
CUresult __stdcall cuDeviceGetName(char *name, int length, CUdevice device);
CUresult __stdcall cuDeviceTotalMem_v2(size_t *bytes, CUdevice device);
CUresult __stdcall cuDeviceGetAttribute(int *value, CUdevice_attribute attribute, CUdevice device);

CUresult __stdcall cuGetExportTable(const void **table, const CUuuid *id);

CUresult __stdcall cuCtxGetCurrent(CUcontext *context);
CUresult __stdcall cuCtxGetApiVersion(CUcontext context, unsigned int *version);
CUresult __stdcall cuCtxGetDevice(CUdevice *device);
CUresult __stdcall cuCtxSynchronize(void);
CUresult __stdcall cuCtxDetach(CUcontext context);

CUresult __stdcall cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name);
CUresult __stdcall cuModuleGetGlobal_v2(CUdeviceptr *pointer, size_t *bytes, CUmodule module, const char *name);
CUresult __stdcall cuModuleGetTexRef(CUtexref *texture, CUmodule module, const char *name);

CUresult __stdcall cuMemHostAlloc(void **pointer, size_t bytes, unsigned int flags);
CUresult __stdcall cuMemAlloc_v2(CUdeviceptr *pointer, size_t bytes);
CUresult __stdcall cuMemGetAddressRange_v2(CUdeviceptr *base, size_t *bytes, CUdeviceptr pointer);
CUresult __stdcall cuMemsetD8_v2(CUdeviceptr destination, unsigned char value, size_t count);
CUresult __stdcall cuMemcpyHtoDAsync_v2(CUdeviceptr destination, const void *source, size_t bytes, CUstream stream);
CUresult __stdcall cuMemcpyDtoHAsync_v2(void *destination, CUdeviceptr source, size_t bytes, CUstream stream);
CUresult __stdcall cuMemcpyDtoDAsync_v2(CUdeviceptr destination, CUdeviceptr source, size_t bytes, CUstream stream);
CUresult __stdcall cuMemFreeHost(void *pointer);
CUresult __stdcall cuMemFree_v2(CUdeviceptr pointer);

CUresult __stdcall cuTexRefSetAddress_v2(size_t *offset, CUtexref texture, CUdeviceptr pointer, size_t bytes);
CUresult __stdcall cuTexRefSetFormat(CUtexref texture, CUarray_format format, int components);
CUresult __stdcall cuTexRefSetFlags(CUtexref texture, unsigned int flags);
CUresult __stdcall cuTexRefSetFilterMode(CUtexref texture, CUfilter_mode mode);
CUresult __stdcall cuTexRefSetMipmapFilterMode(CUtexref texture, CUfilter_mode mode);
CUresult __stdcall cuTexRefSetMipmapLevelBias(CUtexref texture, float bias);
CUresult __stdcall cuTexRefSetMipmapLevelClamp(CUtexref texture, float minimum, float maximum);
CUresult __stdcall cuTexRefSetMaxAnisotropy(CUtexref texture, unsigned int maximum);
CUresult __stdcall cuTexRefSetAddressMode(CUtexref texture, int dimension, CUaddress_mode mode);

CUresult __stdcall cuLaunchKernel(
    CUfunction function,
    unsigned int grid_x, unsigned int grid_y, unsigned int grid_z,
    unsigned int block_x, unsigned int block_y, unsigned int block_z,
    unsigned int dynamic_shared_bytes,
    CUstream stream,
    void **kernel_parameters,
    void **extra);
```

The project tracer builds these exact exports with `i686-w64-mingw32-gcc` and
successfully forwards them to the native driver.

## Queried versus executed exports

The CUDA runtime made 196 distinct `GetProcAddress` requests:

| Category | Queried |
|---|---:|
| Initialization/device/context | 26 |
| Module/execution | 18 |
| Memory/copy/IPC | 38 |
| Textures/arrays | 31 |
| Streams/events | 16 |
| Graphics interop | 61 |
| Diagnostics/discovery | 6 |

All 35 active public functions were among those requests. The other 161
symbols were resolved but never called in any captured workload:

```text
cuArray3DCreate_v2 cuArray3DGetDescriptor_v2 cuArrayCreate_v2 cuArrayDestroy
cuArrayGetDescriptor_v2
cuCtxCreate_v2 cuCtxDisablePeerAccess cuCtxEnablePeerAccess cuCtxGetCacheConfig
cuCtxGetLimit cuCtxGetSharedMemConfig cuCtxGetStreamPriorityRange
cuCtxSetCacheConfig cuCtxSetCurrent cuCtxSetLimit cuCtxSetSharedMemConfig
cuD3D10CtxCreateOnDevice cuD3D10GetDevice cuD3D10GetDevices
cuD3D10GetDirect3DDevice cuD3D10MapResources cuD3D10RegisterResource
cuD3D10ResourceGetMappedArray cuD3D10ResourceGetMappedPitch_v2
cuD3D10ResourceGetMappedPointer_v2 cuD3D10ResourceGetMappedSize_v2
cuD3D10ResourceGetSurfaceDimensions_v2 cuD3D10ResourceSetMapFlags
cuD3D10UnmapResources cuD3D10UnregisterResource
cuD3D11CtxCreateOnDevice cuD3D11GetDevice cuD3D11GetDevices
cuD3D11GetDirect3DDevice
cuD3D9Begin cuD3D9CtxCreateOnDevice cuD3D9End cuD3D9GetDevice
cuD3D9GetDevices cuD3D9GetDirect3DDevice cuD3D9MapResources
cuD3D9MapVertexBuffer_v2 cuD3D9RegisterResource cuD3D9RegisterVertexBuffer
cuD3D9ResourceGetMappedArray cuD3D9ResourceGetMappedPitch_v2
cuD3D9ResourceGetMappedPointer_v2 cuD3D9ResourceGetMappedSize_v2
cuD3D9ResourceGetSurfaceDimensions_v2 cuD3D9ResourceSetMapFlags
cuD3D9UnmapResources cuD3D9UnmapVertexBuffer cuD3D9UnregisterResource
cuD3D9UnregisterVertexBuffer
cuDeviceCanAccessPeer cuDeviceGetByPCIBusId cuDeviceGetPCIBusId
cuEventCreate cuEventDestroy_v2 cuEventElapsedTime cuEventQuery cuEventRecord
cuEventSynchronize
cuFuncGetAttribute cuFuncSetCacheConfig cuFuncSetSharedMemConfig
cuGLCtxCreate_v2 cuGLGetDevices cuGLInit cuGLMapBufferObjectAsync_v2
cuGLMapBufferObject_v2 cuGLRegisterBufferObject cuGLSetBufferObjectMapFlags
cuGLUnmapBufferObject cuGLUnmapBufferObjectAsync cuGLUnregisterBufferObject
cuGetErrorName cuGetErrorString
cuGraphicsD3D10RegisterResource cuGraphicsD3D11RegisterResource
cuGraphicsD3D9RegisterResource cuGraphicsGLRegisterBuffer
cuGraphicsGLRegisterImage cuGraphicsMapResources
cuGraphicsResourceGetMappedMipmappedArray cuGraphicsResourceGetMappedPointer_v2
cuGraphicsResourceSetMapFlags_v2 cuGraphicsSubResourceGetMappedArray
cuGraphicsUnmapResources cuGraphicsUnregisterResource
cuIpcCloseMemHandle cuIpcGetEventHandle cuIpcGetMemHandle
cuIpcOpenEventHandle cuIpcOpenMemHandle
cuLinkAddData cuLinkAddFile cuLinkComplete cuLinkCreate cuLinkDestroy
cuMemAllocManaged cuMemAllocPitch_v2 cuMemGetInfo_v2
cuMemHostGetDevicePointer_v2 cuMemHostGetFlags cuMemHostRegister_v2
cuMemHostUnregister cuMemcpy cuMemcpy2DAsync_v2 cuMemcpy2DUnaligned_v2
cuMemcpy3DAsync_v2 cuMemcpy3DPeer cuMemcpy3DPeerAsync cuMemcpy3D_v2
cuMemcpyAsync cuMemcpyDtoD_v2 cuMemcpyDtoH_v2 cuMemcpyHtoD_v2
cuMemcpyPeer cuMemcpyPeerAsync cuMemsetD2D8Async cuMemsetD2D8_v2
cuMemsetD8Async
cuMipmappedArrayCreate cuMipmappedArrayDestroy cuMipmappedArrayGetLevel
cuModuleGetSurfRef cuModuleLoad cuModuleLoadData cuModuleLoadFatBinary
cuModuleUnload cuOccupancyMaxActiveBlocksPerMultiprocessor
cuPointerGetAttribute
cuProfilerInitialize cuProfilerStart cuProfilerStop
cuStreamAddCallback cuStreamAttachMemAsync cuStreamCreate
cuStreamCreateWithPriority cuStreamDestroy_v2 cuStreamGetFlags
cuStreamGetPriority cuStreamQuery cuStreamSynchronize cuStreamWaitEvent
cuSurfObjectCreate cuSurfObjectDestroy cuSurfObjectGetResourceDesc
cuSurfRefSetArray
cuTexObjectCreate cuTexObjectDestroy cuTexObjectGetResourceDesc
cuTexObjectGetResourceViewDesc cuTexObjectGetTextureDesc
cuTexRefCreate cuTexRefDestroy cuTexRefSetAddress2D_v3 cuTexRefSetArray
cuTexRefSetMipmappedArray
cuWGLGetDevice
```

Querying is not evidence that these functions need semantics. In particular,
61 graphics-interoperability symbols were queried and zero were called. They
remain deferred.

The unresolved symbol-presence contract is narrower: determine whether CUDART
accepts `NULL` for feature-specific queried functions, or requires non-null
stubs. Until probed, the conservative compatibility shape is:

- full semantics for the 35 active exports;
- correct symbol/signature presence for the 161 queried-only exports;
- a non-success result such as `CUDA_ERROR_NOT_SUPPORTED` if a queried-only
  stub is unexpectedly called;
- no implementation of the underlying unused feature.

## Active call frequencies

Counts from the baseline native run:

| Call | Count |
|---|---:|
| `cuTexRefSetAddress_v2` | 283,380 |
| `cuMemcpyHtoDAsync_v2` | 120,091 |
| `cuLaunchKernel` | 96,662 |
| `cuMemGetAddressRange_v2` | 94,460 |
| each non-address texture setter | 94,460 |
| `cuMemcpyDtoHAsync_v2` | 47,230 |
| `cuMemcpyDtoDAsync_v2` | 18,892 |
| `cuMemsetD8_v2` | 9,448 |
| `cuCtxSynchronize` | 9,448 |
| `cuDeviceGetAttribute` | 88 |
| `cuModuleGetFunction` | 70 |
| `cuModuleGetTexRef` | 33 |
| `cuModuleGetGlobal_v2` | 7 |
| `cuGetExportTable` | 6 |
| `cuInit` | 5 |
| `cuDeviceGetCount` | 3 |
| `cuDeviceGet`, `cuCtxGetCurrent` | 2 each |
| each remaining initialization/allocation/free/detach call | 1 |

The high counts are a performance concern for the compatibility layer, not a
reason to introduce a generalized dispatch abstraction before implementation.

## Dark API contract

The six observed table layouts are corroborated by ZLUDA's pinned
[`dark_api/src/lib.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/dark_api/src/lib.rs#L224-L303).
A table's slot 0 `SIZE_OF` value is pointer-width times its declared length.

### CUDART interface

```text
GUID {6BD5FB6C-5BF4-E74A-8987-D93912FD9DF9}
length 13; SIZE_OF = 52 bytes
```

Observed slots:

| Slot | Signature | Calls |
|---:|---|---:|
| 1 | `CUresult get_module_from_cubin(CUmodule *, const FatbincWrapper *)` | 17 |
| 2 | `CUresult cudart_interface_fn2(CUcontext *, CUdevice)` | 1 |

Slot 1 receives the raw fatbin wrapper and returns a module handle. Active
module loading does not call public `cuModuleLoad*`.

The x86 wrapper layout is:

```c
struct FatbincWrapper {
    uint32_t magic;                 // observed 0x466243b1
    uint32_t version;               // observed 1
    const struct FatbinHeader *data;
    const void *filename_or_fatbins;
};

struct FatbinHeader {
    uint32_t magic;                 // 0xba55ed50
    uint16_t version;               // 1
    uint16_t header_size;           // 16
    uint64_t files_size;
};
```

### Tools runtime callbacks

```text
GUID {A094798C-2E74-2E74-93F2-0800200C0A66}
length 7; SIZE_OF = 28 bytes
```

| Slot | Signature | Observation |
|---:|---|---|
| 2 | `void get_unknown_buffer1(void **, size_t *)` | returned 1024-byte buffer |
| 6 | `void get_unknown_buffer2(void **, size_t *)` | returned 14-byte buffer |

### Tools TLS

```text
GUID {42D85A81-23F6-CB47-8298-F6E78A3AECDC}
length 4; SIZE_OF = 16 bytes
```

No callable slot was observed. ZLUDA reported a table-length diagnostic against
the current driver, but native execution continued.

### Context-local storage V0301

```text
GUID {C693336E-1121-DF11-A8C3-68F355D89593}
length 4; SIZE_OF = 16 bytes
```

| Slot | Signature | Observation |
|---:|---|---|
| 0 | `CUresult put(CUcontext, void *key, void *value, destructor)` | one successful install |
| 1 | `CUresult delete(CUcontext, void *key)` | not called in the captured run |
| 2 | `CUresult get(void **value, CUcontext, void *key)` | high-frequency lookup |

Two `get` calls before `put` returned `CUDA_ERROR_INVALID_HANDLE`; subsequent
lookups returned the stored value and `CUDA_SUCCESS`.

### Context creation bypass

```text
GUID {0CA50B8C-1004-929A-89A7-D0DF10E77286}
length 2; SIZE_OF = 8 bytes
```

| Slot | Signature | Observation |
|---:|---|---|
| 1 | `CUresult ctx_create_v2_bypass(CUcontext *, unsigned int flags, CUdevice)` | flags `20`, device `0`, success |

No public `cuCtxCreate*` call occurred.

### Heap access

```text
GUID {195BCBF4-D67D-024A-ACC5-1D29CEA631AE}
length 3; SIZE_OF = 12 bytes
```

| Slot | Signature | Observation |
|---:|---|---|
| 1 | `CUresult heap_alloc(const void **record, size_t arg2, size_t arg3)` | one allocation record |
| 2 | `CUresult heap_free(const void *record, size_t *arg2)` | one release at shutdown |

## Runtime lifecycle

The observed state machine is:

```text
NVAPI hardware gate
  -> cuInit / device enumeration
  -> cuGetExportTable tables
  -> dark ctx_create_v2_bypass(flags=20, device=0)
  -> current-context queries and CLS install
  -> 17 dark CUDART fatbin registrations
  -> 70 function + 7 global + 33 texture lookups
  -> one 32 MiB pinned host pool + one 32 MiB device pool
  -> repeated copies / texture binds / packed launches / context synchronize
  -> final DtoH copies and synchronize
  -> free host pool
  -> free device pool
  -> cuCtxDetach
  -> dark heap release
```

### PhysX SDK proxy differences

The official PhysX 2.8.0 `SampleCloth` and `SampleParticleFluid` binaries use
the public `cuCtxCreate_v2` path and request 128 MiB pinned-host and device
pools. These are not different CUDA semantics: public and Dark context creation
must construct the same internal context state, and allocation sizes remain
caller-supplied runtime values. The 32 MiB Mirror's Edge pools are fixture
evidence, not allocator constants.

`SampleCloth` scene 1 loads the same 17-module PhysX 2.8.0 corpus and launches
the same three deformable entry names as the game. Scene 2 adds the four-stage
self-collision path and radix sort, and provides the context teardown/recreation
qualification case. `SampleParticleFluid` remains a later qualification case;
neither sample justifies unrelated CUDA features.

No public stream, event, module-unload, array, surface, graphics-resource, or
peer/IPC lifecycle occurred. Modules remain valid until context teardown.

## Device enumeration and properties

Native initialization returned:

| Query | Value |
|---|---|
| device count | `1` |
| selected ordinal | `0` |
| name | `NVIDIA GeForce RTX 3070 Laptop GPU` |
| driver version | `13000` |
| context API version | `3020` |
| compute capability | `8.6` |
| total memory | `4294967295` (`size_t` saturation at 32 bits) |
| warp size | `32` |

CUDART made 88 attribute calls representing 85 distinct attribute/value pairs;
all returned `CUDA_SUCCESS`. The complete native response is:

| Attribute | Native value |
|---|---:|
| private numeric `0x20080402` | `256` |
| `COMPUTE_CAPABILITY_MAJOR` | `8` |
| `COMPUTE_CAPABILITY_MINOR` | `6` |
| `GPU_OVERLAP` | `1` |
| `ASYNC_ENGINE_COUNT` | `1` |
| `MULTIPROCESSOR_COUNT` | `40` |
| `KERNEL_EXEC_TIMEOUT` | `1` |
| `INTEGRATED` | `0` |
| `CAN_MAP_HOST_MEMORY` | `1` |
| `MAXIMUM_TEXTURE1D_WIDTH` | `131072` |
| `MAXIMUM_TEXTURE1D_MIPMAPPED_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE1D_LINEAR_WIDTH` | `268435456` |
| `MAXIMUM_TEXTURE2D_WIDTH` | `131072` |
| `MAXIMUM_TEXTURE2D_HEIGHT` | `65536` |
| `MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT` | `32768` |
| `MAXIMUM_TEXTURE2D_LINEAR_WIDTH` | `131072` |
| `MAXIMUM_TEXTURE2D_LINEAR_HEIGHT` | `65000` |
| `MAXIMUM_TEXTURE2D_LINEAR_PITCH` | `2097120` |
| `MAXIMUM_TEXTURE2D_GATHER_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE2D_GATHER_HEIGHT` | `32768` |
| `MAXIMUM_TEXTURE3D_WIDTH` | `16384` |
| `MAXIMUM_TEXTURE3D_HEIGHT` | `16384` |
| `MAXIMUM_TEXTURE3D_DEPTH` | `16384` |
| `MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE` | `8192` |
| `MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE` | `8192` |
| `MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE` | `32768` |
| `MAXIMUM_TEXTURECUBEMAP_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE1D_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE1D_LAYERED_LAYERS` | `2048` |
| `MAXIMUM_TEXTURE2D_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_TEXTURE2D_LAYERED_HEIGHT` | `32768` |
| `MAXIMUM_TEXTURE2D_LAYERED_LAYERS` | `2048` |
| `MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS` | `2046` |
| `MAXIMUM_SURFACE1D_WIDTH` | `32768` |
| `MAXIMUM_SURFACE2D_WIDTH` | `131072` |
| `MAXIMUM_SURFACE2D_HEIGHT` | `65536` |
| `MAXIMUM_SURFACE3D_WIDTH` | `16384` |
| `MAXIMUM_SURFACE3D_HEIGHT` | `16384` |
| `MAXIMUM_SURFACE3D_DEPTH` | `16384` |
| `MAXIMUM_SURFACE1D_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_SURFACE1D_LAYERED_LAYERS` | `2048` |
| `MAXIMUM_SURFACE2D_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_SURFACE2D_LAYERED_HEIGHT` | `32768` |
| `MAXIMUM_SURFACE2D_LAYERED_LAYERS` | `2048` |
| `MAXIMUM_SURFACECUBEMAP_WIDTH` | `32768` |
| `MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH` | `32768` |
| `MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS` | `2046` |
| `CONCURRENT_KERNELS` | `1` |
| `ECC_ENABLED` | `0` |
| `PCI_BUS_ID` | `1` |
| `PCI_DEVICE_ID` | `0` |
| `PCI_DOMAIN_ID` | `0` |
| `TCC_DRIVER` | `0` |
| `UNIFIED_ADDRESSING` | `0` |
| `MEMORY_CLOCK_RATE` | `7001000` |
| `GLOBAL_MEMORY_BUS_WIDTH` | `256` |
| `L2_CACHE_SIZE` | `4194304` |
| `MAX_THREADS_PER_MULTIPROCESSOR` | `1536` |
| `SURFACE_ALIGNMENT` | `512` |
| `TEXTURE_PITCH_ALIGNMENT` | `32` |
| `MAX_SHARED_MEMORY_PER_BLOCK` | `49152` |
| `MAX_SHARED_MEMORY_PER_MULTIPROCESSOR` | `102400` |
| `MAX_REGISTERS_PER_BLOCK` | `65536` |
| `MAX_REGISTERS_PER_MULTIPROCESSOR` | `65536` |
| `WARP_SIZE` | `32` |
| `MAX_PITCH` | `2147483647` |
| `MAX_THREADS_PER_BLOCK` | `1024` |
| `MAX_BLOCK_DIM_X` | `1024` |
| `MAX_BLOCK_DIM_Y` | `1024` |
| `MAX_BLOCK_DIM_Z` | `64` |
| `MAX_GRID_DIM_X` | `2147483647` |
| `MAX_GRID_DIM_Y` | `65535` |
| `MAX_GRID_DIM_Z` | `65535` |
| `TOTAL_CONSTANT_MEMORY` | `65536` |
| `CLOCK_RATE` | `1560000` |
| `TEXTURE_ALIGNMENT` | `512` |
| `STREAM_PRIORITIES_SUPPORTED` | `1` |
| `GLOBAL_L1_CACHE_SUPPORTED` | `1` |
| `LOCAL_L1_CACHE_SUPPORTED` | `1` |
| `MANAGED_MEMORY` | `0` |
| `MULTI_GPU_BOARD` | `0` |
| `MULTI_GPU_BOARD_GROUP_ID` | `0` |
| `COMPUTE_MODE` | `0` |

These are oracle values, not all mandatory constants. The minimum response
policy must be derived from checks in ZLUDA/PhysX behavior or native probes.
Clearly load-bearing values include warp size 32, 32-bit non-UVA addressing,
host mapping, block/grid limits above observed launches, shared memory above
10,756 bytes, and texture/constant capacities above the corpus requirements.
Clock, cache, PCI, and unused surface/array limits should not be copied into a
portable implementation unless caller behavior proves they matter.

## Memory contract

Observed allocation behavior:

- `cuMemHostAlloc(..., 32 MiB, flags=0)` once;
- `cuMemAlloc_v2(..., 32 MiB)` once;
- no other public allocation;
- 32-bit device pointers used as suballocation addresses;
- `cuMemGetAddressRange_v2` must return the owning pool base and full 32 MiB
  size when passed any in-pool suballocation;
- both pools survive all simulation steps and are freed at shutdown.

Observed transfers:

| Call | Calls | Bytes | Size range |
|---|---:|---:|---:|
| `cuMemcpyHtoDAsync_v2` | 120,091 | 93,554,248 | 8–5120 |
| `cuMemcpyDtoDAsync_v2` | 18,892 | 19,345,408 | exactly 1024 |
| `cuMemcpyDtoHAsync_v2` | 47,230 | 108,247,680 | 256–5120 |
| `cuMemsetD8_v2` | 9,448 | 2 MiB per call | exactly 2 MiB |

Every transfer used `stream = NULL`. Host-visible completion is established by
`cuCtxSynchronize`; the contract does not require non-default streams from
current evidence. Async entry points must still preserve issue order within the
default stream.

## Module and symbol contract

Seventeen raw fatbins are registered through CUDART Dark API slot 1. For each
returned `CUmodule`, callers may request:

- 70 exact function names;
- 7 exact constant/global names;
- 33 exact legacy texture-reference names.

Unknown names were not observed. Module handles, function handles, globals, and
texture references remain stable for the context lifetime. No module unload
occurs.

`cuModuleGetGlobal_v2` returns a 32-bit device address. Its `bytes` output was
passed as `NULL` in the trace and must be accepted. The seven symbol sizes and
names are cataloged in `mirrors-edge-ptx-corpus.md`.

## Legacy texture contract

The active workload uses module-declared texture references, not texture
objects or arrays.

Per binding cycle:

1. `cuMemGetAddressRange_v2` identifies the containing pool;
2. `cuTexRefSetAddress_v2(..., pointer=0, bytes=0)` unbinds the old range;
3. format and range are set for a linear device suballocation;
4. flags/filter/mipmap/anisotropy/address mode are applied;
5. the kernel launches;
6. the address is reset to zero after use.

Observed settings:

| Property | Values |
|---|---|
| format | `CU_AD_FORMAT_FLOAT` or `CU_AD_FORMAT_UNSIGNED_INT32` |
| packed components | `1` or `4` |
| flags | `0` for float reads, `1` (`READ_AS_INTEGER`) for integer reads |
| filter | `CU_TR_FILTER_MODE_POINT` |
| mipmap filter | `CU_TR_FILTER_MODE_POINT` |
| mip bias | `0` |
| mip clamp | `0, 0` |
| max anisotropy | `0` |
| address mode dimension | `0` only |
| address mode | `CU_TR_ADDRESS_MODE_CLAMP` |

The caller passed `byte_offset = NULL` to every observed
`cuTexRefSetAddress_v2` call; this optional output must be accepted. The PTX
corpus uses only 1D four-component reads.

## Launch contract

Every launch uses:

- `kernel_parameters = NULL`;
- `extra` containing `CU_LAUNCH_PARAM_BUFFER_POINTER`, a packed buffer,
  `CU_LAUNCH_PARAM_BUFFER_SIZE`, a pointer to x86 `size_t`, then
  `CU_LAUNCH_PARAM_END`;
- `stream = NULL`;
- one-dimensional grids and blocks;
- Y/Z dimensions equal to 1.

All 16 packed layouts and their exact offsets match the PTX `.param`
declarations; see `mirrors-edge-ptx-corpus.md`. `cuLaunchKernel` must copy or
consume the argument bytes according to CUDA launch semantics before the caller
reuses its host buffer.

Mirror's Edge uses block X values 64, 128, 192, and 256 and dynamic shared
sizes 0, 260, 4096, and 7072 bytes. PhysX 2.8.0 `SampleCloth` adds solver
launches with block X 512 and 42,240 bytes in scene 1, 15,376 bytes in scene 2,
and 23,616 bytes in scene 3. Scene 4 uses block X 128 and 3,168 bytes. Launch
dimensions and shared size are runtime inputs. The backend combines dynamic and
static layouts, then selects native Workgroup memory or per-workgroup storage
scratch from the physical Vulkan limit.

## Result and error behavior

Every observed public CUDA call returned `CUDA_SUCCESS`.

Observed non-success behavior is limited to Dark API initialization:

- CLS `get` before `put` returned `CUDA_ERROR_INVALID_HANDLE` twice;
- after `put`, the same lookup returned `CUDA_SUCCESS` and the stored value;
- NVAPI feature queries returned negative statuses described above without
  aborting CUDA initialization.

No public invalid-argument, allocation-failure, missing-symbol, launch-failure,
or asynchronous-error path was exercised. Do not invent detailed error
compatibility from CUDA documentation alone. Targeted probes should define only
errors that PhysX observes or that tests need.

## Minimum host-side implementation scope

Required semantics from current evidence:

1. NVAPI hardware gate sufficient to select GPU PhysX.
2. One CUDA device and stable ordinal 0.
3. The 35 active public exports with the x86 signatures above.
4. Six Dark API tables with the active slots above.
5. One current context and CLS state.
6. Raw fatbin registration and lookup of exact entries/globals/texture refs.
7. One long-lived pinned host pool and one long-lived device pool, with general
   allocation behavior sufficient to produce them.
8. Ordered default-stream copies, memset, launches, and context synchronization.
9. Legacy linear texture-reference binding.
10. Packed launch-buffer handling for the 16 observed entries.
11. Clean pool release and `cuCtxDetach` teardown.

Not required from current evidence:

- D3D, OpenGL, or general graphics interop;
- arrays, mipmapped arrays, surfaces, texture objects;
- events or non-default streams;
- managed memory or UVA;
- IPC, peer access, linking, profiling, occupancy APIs;
- general CUDA runtime coverage;
- semantics for queried-only exports unless a probe shows that CUDART invokes
  them under a target path.

## Remaining probes before freezing the contract

1. Return `NULL` for queried-only symbol groups to determine whether CUDART
   requires pointer presence or merely probes capabilities.
2. If non-null is required, verify that a uniform `CUDA_ERROR_NOT_SUPPORTED`
   stub is accepted when an unused symbol is never called.
3. Determine the minimum NVAPI interface set that passes hardware selection on
   a non-NVIDIA renderer.
4. Test conservative device-property values rather than cloning RTX-specific
   clocks/cache/PCI fields.
5. Add thread IDs to one trace to determine whether PhysX calls CUDA from one
   host thread or several; implementation should remain thread-safe regardless.
6. Capture one deterministic output-buffer oracle for fluid and deformable
   validation.
