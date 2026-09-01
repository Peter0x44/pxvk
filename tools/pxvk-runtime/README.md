# pxvk CUDA facade

`pxvk-runtime` builds the 32-bit `nvcuda.dll` loaded by PhysX 2.8.x. It is a
finite CUDA Driver API facade for the translated pxvk kernel set, not a general
CUDA implementation.

## Design

The DLL follows the parts of ZLUDA's x86 PhysX path that the captured workload
proves useful:

- broad CUDA export-name presence with not-supported behavior outside the
  observed surface;
- opaque local contexts and handles;
- six local `cuGetExportTable` tables used by CUDART and PhysX;
- 32-bit virtual device pointers;
- packed `cuLaunchKernel` buffers;
- local host allocations and synchronous ordered copies.

The reference is ZLUDA commit
[`8a7931be`](https://github.com/vosen/ZLUDA/tree/8a7931be87d758835776f54ea2e33d30621c74ce):
[`zluda32/src/lib.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda32/src/lib.rs)
defines the x86 export wrappers, local context state, packed launch decoding,
virtual memory calls, and fatbin registration.
[`dark_api/src/lib.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/dark_api/src/lib.rs)
defines the private export-table layouts mirrored here.

Unlike ZLUDA, pxvk has no x86-to-x64 server, shared-memory RPC, HIP objects, or
LLVM/HIP launch ABI. The 32-bit DLL owns a Vulkan 1.2 instance, device, compute
queue, mapped storage buffers, shader pipelines, and CUDA-visible state in the
application process.

```text
PhysX 2.8 CUDART
  -> PE32 nvcuda.dll exports / Dark API tables
  -> context, module, function, allocation, global, texture records
  -> packed launch validation and translated-kernel lookup
  -> pretranslated SPIR-V in .\spirv (or PXVK_SHADER_DIR)
  -> one ordered Vulkan compute queue
```

CUDA pointers are byte offsets into one mapped Vulkan storage buffer. Zero stays
null. Allocation records validate every transfer range; shaders divide the same
32-bit addresses by four when indexing binding 0. Bindings are fixed by the
existing emitter: heap 0, launch parameters 1, module constants 2, legacy
texture metadata 3, and storage-backed shared memory 4.

The first runtime intentionally executes the default stream synchronously.
This preserves issue order and host visibility while avoiding per-call staging
allocations. It is slower than queued batching but compatible with the captured
PhysX path and keeps the initial failure surface small.

NVIDIA's installed `cudart32_65.dll` constructs an absolute System32 path for
`nvcuda.dll`, so normal application-directory DLL search cannot replace it.
`pxvk-launch.exe` creates the target suspended, injects the staged runtime, and
then resumes it. The runtime's loader notification patches only CUDART's
`LoadLibraryW`/`LoadLibraryExW` imports so its `nvcuda.dll` request resolves to
the already selected staged DLL. Other module loads are unchanged.

Generated SPIR-V is deployment data because captured PhysX PTX is proprietary.
`prepare-sample.ps1` translates the three `SampleCloth` entries, builds the DLL,
and stages a runnable directory without modifying the original sample.

## Build

Use a 32-bit MinGW or Visual Studio generator:

```powershell
cmake -S .\tools\pxvk-runtime -B .\tmp\build-pxvk-runtime `
  -G "MinGW Makefiles" `
  -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc `
  -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++
cmake --build .\tmp\build-pxvk-runtime
ctest --test-dir .\tmp\build-pxvk-runtime --output-on-failure
```

The built `nvcuda.dll` requires a 32-bit Vulkan loader and ICD. Set
`PXVK_VULKAN_DEVICE` to a physical-device index and `PXVK_LOG` to a diagnostic
log path when needed.
