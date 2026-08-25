# ZLUDA runtime study for Mirror's Edge

## Scope and conclusion

Research is pinned to ZLUDA commit
[`8a7931be87d758835776f54ea2e33d30621c74ce`](https://github.com/vosen/ZLUDA/tree/8a7931be87d758835776f54ea2e33d30621c74ce).
Only behavior relevant to the captured 32-bit PhysX workload is considered.

ZLUDA proves that this workload tolerates substantial simplification:

- public symbol stubs for unused CUDA functions;
- fabricated/local context handles;
- one 32-bit virtual device-address space;
- packed launch arguments decoded from PTX metadata;
- no-op legacy texture settings that do not affect linear point reads;
- synchronous implementation of nominally asynchronous device-to-host copies;
- backend-specific device properties rather than cloned NVIDIA values.

ZLUDA's separate 64-bit server is not reusable for pxvk. Current source does
not use a named pipe: it spawns `zluda64_server.exe` and communicates through
named shared-memory mappings plus named events. That boundary exists so the
x86 CUDA facade can call ZLUDA's 64-bit HIP/ROCm backend. A 32-bit Vulkan
implementation can remain in process, so pxvk should not adopt the server,
serialization protocol, shared-memory transport, or forced cross-process
copies.

## Why ZLUDA uses a 64-bit server

The x86 DLL's `GlobalState` owns an `ipc::Server`, and the first state access
spawns `zluda64_server.exe`; see
[`zluda32/src/lib.rs`, lines 153–220](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L153-L220).

`Server::start` creates two endpoints, launches the sibling server with four
endpoint names, and arranges for the child to die with the client; see
[`zluda32/src/ipc.rs`, lines 14–67](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/ipc.rs#L14-L67).
Each endpoint is a named event plus a named file mapping. Calls serialize an
opcode/body, then use `SignalObjectAndWait`; see
[`zluda32/src/ipc.rs`, lines 69–137](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/ipc.rs#L69-L137) and
[`zluda_server_common/src/lib.rs`, lines 424–500](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_server_common/src/lib.rs#L424-L500).

The server links against the 64-bit ZLUDA `nvcuda.dll`, whose implementation
uses `hip_runtime-sys` and HIP objects. The raw-dylib imports are generated in
[`zluda64_server/src/main.rs`, lines 1269–1339](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L1269-L1339),
while the main runtime maps CUDA contexts, memory, modules, streams, events, and
textures onto HIP; for example
[`zluda/src/impl/context.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/context.rs) and
[`zluda/src/impl/memory.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/memory.rs).

### Boundary costs that pxvk avoids

The protocol encodes every CUDA handle, device pointer, and `size_t` as a
little-endian `u32`; see
[`zluda_server_common/src/lib.rs`, lines 118–196](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_server_common/src/lib.rs#L118-L196).
It also:

- copies every host-to-device payload into shared memory;
- returns every device-to-host payload as a serialized vector;
- synchronizes a stream before returning a nominally async DtoH copy;
- serializes client state behind one mutex and server request loop;
- requires wire handle translation and version-matched client/server binaries.

These are consequences of crossing x86→x64 process and pointer boundaries, not
CUDA or PhysX requirements. Vulkan objects and mapped host memory can be owned
by the x86 process directly. This session decision is explicit: **pxvk will not
use ZLUDA's server boundary.** Vulkan x86 availability and required features
still need to be recorded in `vulkan-requirements.md`.

## Export strategy and queried-only symbols

ZLUDA generates exported wrappers for an explicit implemented set. Every other
CUDA export is still present but returns `CUDA_ERROR_NOT_SUPPORTED` (or
`CUDA_ERROR_DEINITIALIZED` after detach); see the `implemented!` and
`not_implemented!` macros in
[`zluda32/src/lib.rs`, lines 19–115](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L19-L115).

This directly supports the provisional policy in `cuda-driver-contract.md`:
Mirror's Edge/CUDART can query a broad symbol surface while using a much smaller
semantic implementation. ZLUDA does not require semantics for the 161
queried-only functions in our trace. A generated export table with uniform
not-supported stubs is evidence-backed; implementing the unused APIs is not.

## Local x86 state versus remote backend state

ZLUDA deliberately keeps some CUDA-visible state in the x86 DLL:

- a thread-local context stack;
- fabricated 32-bit context handles;
- context-local-storage keys, values, and destructor callbacks;
- packed function-argument metadata;
- CUDART/Tools/CLS/context-bypass/heap Dark API tables;
- host allocations made by `cuMemHostAlloc`;
- unknown 1024- and 14-element callback buffers.

`ContextStack` and `GlobalState` are defined in
[`zluda32/src/lib.rs`, lines 117–222](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L117-L222).
`cu_ctx_create_v2` ignores flags, creates an integer-like local handle, pushes
it, and never asks the server to create a matching context; see
[lines 229–268](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L229-L268).
The 64-bit server instead retains one primary backend context during `State::new`;
see
[`zluda64_server/src/main.rs`, lines 262–275](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L262-L275).

This proves that PhysX treats CUDA context handles as opaque and does not depend
on native context flag semantics in the captured path. pxvk can keep context,
CLS, and heap bookkeeping entirely local, with no server analogue.

## Active public API mapping

### Initialization and device queries

`cuInit`, device enumeration, name, most attributes, total memory, driver
version, and context API version are RPCs into the server; see
[`zluda32/src/lib.rs`, lines 224–444](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L224-L444)
and server handlers in
[`zluda64_server/src/main.rs`, lines 1089–1179](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L1089-L1179).

Important local compatibility overrides:

- private attribute `0x20080402` returns `256` exactly;
- `CAN_MAP_HOST_MEMORY` returns `0`, although the NVIDIA oracle returned `1`;
- 64-bit total memory is saturated to `u32::MAX` for x86.

ZLUDA documents Mirror's Edge as tested while this x86 path reports
`CAN_MAP_HOST_MEMORY=0`; see
[`docs/src/physx32.md`, lines 67–72](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/docs/src/physx32.md#L67-L72).
That proves native RTX device properties are not all load-bearing. pxvk should
report capability values from its real backend plus targeted compatibility
overrides, not copy the RTX table.

### Context operations

| Active function | ZLUDA behavior |
|---|---|
| `cuCtxGetCurrent` | local thread-local stack |
| `cuCtxGetDevice` | local stack's device |
| `cuCtxGetApiVersion` | remote backend query |
| `cuCtxSynchronize` | remote backend synchronize |
| `cuCtxDetach` | local CLS removal and destructor callbacks |

There is no per-context remote lifecycle corresponding to each local handle.

### Dark API tables

`cuGetExportTable` is local and returns statically generated tables; unknown
GUIDs return `CUDA_ERROR_NOT_SUPPORTED`; see
[`zluda32/src/lib.rs`, lines 471–490](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L471-L490).

`DarkApi32` implements the observed slots in
[`zluda32/src/lib.rs`, lines 1128–1344](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L1128-L1344):

- CUDART fatbin registration calls local `cu_module_load_data`;
- CUDART setup returns an opaque fake context value `0xDEADBEEF`;
- callback-buffer slots return static local buffers;
- CLS is a map keyed by local context and caller pointer;
- context-bypass delegates to local context creation;
- heap slots allocate/free a two-field host record;
- unobserved Dark API slots return not-supported or zero.

The fake CUDART context value is especially useful evidence: PhysX does not
interpret it. pxvk needs stable opacity, not NVIDIA's actual context bits.

## Fatbin and module lifecycle

The x86 `cu_module_load_data`:

1. parses the input with `CodeLibraryRef`;
2. walks fatbin members;
3. keeps PTX members and ignores non-PTX images;
4. forwards the last PTX module as a null-terminated byte vector.

See
[`zluda32/src/lib.rs`, lines 963–1003](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L963-L1003).
Each Mirror's Edge fatbin contains one PTX member, so the last-module shortcut
works for this corpus. pxvk can select the sole PTX member directly; it should
not inherit the ambiguous general-case policy.

The 64-bit ZLUDA runtime compiles PTX to a HIP-loadable ELF and stores extra
32-bit metadata. `ModuleLaunchData::new_module` asks a private table for global
names, initializers, sizes, and alignments, allocates each in the virtual
32-bit address space, copies initializers, and records per-module globals; see
[`zluda64_server/src/main.rs`, lines 72–147](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L72-L147).
The main runtime exposes that metadata through
[`zluda/src/impl/driver.rs`, lines 482–558](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda/src/impl/driver.rs#L482-L558).

Function lookup performs two operations:

1. resolve and virtualize the backend function handle;
2. retrieve explicit argument size/alignment metadata and cache it in the x86
   client.

See
[`zluda32/src/lib.rs`, lines 713–745](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L713-L745) and
[`zluda64_server/src/main.rs`, lines 961–982](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L961-L982).

Reusable idea: module compilation should emit an immutable metadata record
containing entry parameter layouts, globals, initializers, texture references,
and static shared requirements. It need not be exposed through a private
server-side Dark API in pxvk; the local compiled-module object can own it.

## 32-bit device-pointer virtualization

The server reserves one 512 MiB backend allocation and exposes 32-bit byte
offsets in 256-byte allocation units. Offset zero remains null. A range map
supports allocation, free, translation to the real pointer, and
`cuMemGetAddressRange_v2`; see
[`zluda64_server/src/main.rs`, lines 170–260](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L170-L260).

Opaque backend handles are placed in a separate u32 namespace beginning at
512 MiB; see
[`zluda64_server/src/main.rs`, lines 32–69](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L32-L69).

This is the most directly reusable runtime concept:

- CUDA `CUdeviceptr` is a 32-bit offset, not a native host pointer;
- a range allocator owns suballocations;
- translated kernels receive access to one real backing allocation plus the
  pseudo-pointer offset;
- handles and device addresses occupy disjoint namespaces.

For in-process Vulkan, the real backing object can be a storage buffer instead
of a 64-bit HIP pointer. The exact Vulkan binding and allocation size remain
open until `vulkan-requirements.md`; the x86/server indirection does not.

## Host memory and copies

The x86 client implements `cuMemHostAlloc` as zeroed `calloc` and frees it with
`free`; flags are ignored. It is not actually page-locked. See
[`zluda32/src/lib.rs`, lines 602–618](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L602-L618).

Copy behavior in
[`zluda32/src/lib.rs`, lines 620–710](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L620-L710):

- HtoD copies source bytes into the request frame;
- DtoH returns a byte vector and copies it into the caller's pointer;
- DtoD sends only offsets and count;
- memset sends offset/value/count;
- all sizes truncate to u32 wire values.

The server translates u32 offsets to backend pointers. DtoH explicitly
synchronizes the backend stream before returning; see
[`zluda64_server/src/main.rs`, lines 891–908](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L891-L908).

These shortcuts prove that Mirror's Edge does not require true pinned-host or
DtoH-asynchronous behavior in the captured path. pxvk still needs ordered
copies and `cuCtxSynchronize` visibility, but it should use local Vulkan
staging/mapping rather than emulate ZLUDA's serialization.

## Packed launch handling

ZLUDA's x86 launch path exactly matches our traced ABI. It:

1. rejects `kernel_params != NULL` or `extra == NULL`;
2. requires the exact five-element buffer-pointer/buffer-size/end token list;
3. retrieves cached `(size, align)` metadata for the function;
4. uses `Layout::extend` to split the packed buffer into per-parameter byte
   vectors;
5. verifies the computed end equals the caller's packed size;
6. sends launch geometry, stream, shared bytes, handle, and argument vectors.

See
[`zluda32/src/lib.rs`, lines 492–558](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L492-L558).
This independently confirms every packed layout in
`mirrors-edge-ptx-corpus.md`.

The server reconstructs a `kernelParams` array, then appends hidden parameters:

- the actual 64-bit base of the 32-bit virtual memory pool;
- one 32-bit pseudo-pointer for each module global.

It launches the transformed backend function with `extra = NULL`; see
[`zluda64_server/src/main.rs`, lines 916–958](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L916-L958).

The compiler transformation that enables this is explicit in
[`ptx/src/pass/convert_32bit_to_64bit.rs`, lines 1–10](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/convert_32bit_to_64bit.rs#L1-L10)
and
[lines 141–229](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/ptx/src/pass/convert_32bit_to_64bit.rs#L141-L229): it removes
32-bit globals, injects a 64-bit memory-base argument and one u32 argument per
global, records explicit argument layouts, and rewrites pseudo-pointer loads
relative to the hidden base.

Reusable idea: translate 32-bit CUDA addresses as offsets into an explicitly
bound memory object and keep global offsets as module metadata. pxvk does not
need to convert them into host 64-bit pointers or append the same HIP-specific
hidden ABI, but it needs an equivalent shader-visible base/offset model.

## Legacy texture references

The x86 client forwards only address, format, and flags. It treats address mode,
filter mode, mipmap filter/bias/clamp, and anisotropy as successful no-ops; see
[`zluda32/src/lib.rs`, lines 833–931](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs#L833-L931).

The server memoizes texture handles by `(module, name)`, translates a 32-bit
address to the backing allocation, and forwards address/format/flags to the
64-bit runtime; see
[`zluda64_server/src/main.rs`, lines 833–875](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L833-L875) and
[lines 985–1007](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda64_server/src/main.rs#L985-L1007).

This simplification is evidence-backed for Mirror's Edge because every active
texture is linear, point-filtered, non-mipmapped, and clamp-addressed. pxvk can
store the full observed state locally but may implement the invariant
point/clamp/zero-mipmap setters as validation plus no-op state transitions.

## Minimal NVAPI shim pattern

ZLUDA's NVAPI implementation is directly relevant and independent of its
server. `nvapi_QueryInterface` maps known numeric interfaces to local function
pointers; unknown IDs return null. See
[`nvapi/src/lib.rs`, lines 558–560](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/nvapi/src/lib.rs#L558-L560).

The hardware gate fabricates one GPU handle/ID and sets only the topology flags
needed to advertise CUDA and PhysX as capable, available, enabled, and
recommended. It supports both V1 and V2 topology layouts, fake driver strings,
GPU-ID round trips, and private interface `0x36e39e6b`; see
[`nvapi/src/lib.rs`, lines 544–719](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/nvapi/src/lib.rs#L544-L719).

This is a strong candidate for a small C shim in pxvk tooling/runtime. It does
not justify importing ZLUDA's generated 825 KiB NVAPI bindings or implementing
general NVAPI.

## What to reuse

1. **Full export presence with not-supported stubs.** This matches the observed
   broad CUDART query behavior without implementing unused CUDA.
2. **Local context/CLS/heap/callback-buffer state.** All are small host data
   structures and do not need a backend round trip.
3. **Fatbin member selection and module metadata.** Keep PTX, globals,
   initializers, texrefs, entry layouts, and shared-memory requirements in one
   immutable module object.
4. **Packed argument decoding by `(size, align)`.** ZLUDA and our independent
   analysis agree exactly.
5. **u32 virtual device pointers.** Use offsets into a bounded backing memory
   object, reserve zero for null, and maintain allocation ranges.
6. **Disjoint opaque handle IDs.** Avoid exposing Vulkan handles or host
   pointers through the CUDA ABI.
7. **Explicit base/global lowering.** Shader code needs a defined way to turn
   32-bit pseudo-pointers and globals into accesses to bound Vulkan memory.
8. **Narrow NVAPI topology shim.** Advertise one PhysX/CUDA-capable logical GPU
   and nothing more.
9. **Successful no-ops for invariant texture state.** Only where the captured
   values prove the operation has no effect on this workload.

## What not to reuse

1. `zluda64_server.exe`, shared mappings, named events, rkyv messages, opcodes,
   or client/server version checks.
2. HIP objects, HIP streams/events, COMGR/LLVM output assumptions, or ROCm
   device-property translation.
3. Cross-process handle translation and host-payload serialization.
4. Forced stream synchronization as the general meaning of async copies.
5. `calloc` presented as pinned host allocation without checking failure.
6. A fixed 512 MiB allocation size before Vulkan limits and workload needs are
   established.
7. Fake context constants such as `0xDEADBEEF` as externally meaningful values.
8. Broad global serialization through one mutex/request loop.
9. ZLUDA TODO/no-op behavior outside the exact observed invariant values.

## Consequences for pxvk research

The server is not an architectural prerequisite. Removing it simplifies the
observed host path to:

```text
x86 CUDA/NVAPI exports
  -> local context, handles, module metadata, allocations, and texture state
  -> local PTX translation
  -> local Vulkan memory/descriptor/dispatch operations
```

This does not yet choose a language, Vulkan wrapper, descriptor strategy,
shader IR, cache format, or allocator size. Those decisions remain gated by
`zluda-ptx.md`, `vulkan-requirements.md`, validation evidence, and
`research-synthesis.md`.

## Remaining runtime questions

- Which of the 161 queried-only CUDA exports can be absent versus requiring a
  not-supported stub? ZLUDA proves stubs work, not that null pointers work.
- Which minimum NVAPI interface IDs are required specifically by the current
  system `PhysXDevice.dll`?
- Should host allocations be ordinary aligned memory, Vulkan-mapped memory, or
  staged copies for the first slice?
- What Vulkan allocation/descriptor model best preserves u32 pseudo-pointer
  arithmetic and the seven module globals?
- How should default-stream ordering and `cuCtxSynchronize` be represented
  without adopting ZLUDA's forced RPC synchronization?
- Which exact PTX lowering choices from ZLUDA are useful once its HIP-specific
  backend is removed? This is the subject of `zluda-ptx.md`.
