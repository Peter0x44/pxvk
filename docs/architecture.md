# pxvk architecture

pxvk replaces the 32-bit NVIDIA CUDA driver (`nvcuda.dll`) inside a PhysX
application. Like ZLUDA it implements the driver API and the private CUDART
export tables and compiles PTX at runtime; unlike ZLUDA it targets Vulkan
compute directly and runs entirely inside the 32-bit process.

```text
application ── PhysX 2.8 ── cudart32_xx.dll ── nvcuda.dll (pxvk)
                                                 │
                    src/runtime   CUDA driver API, private export tables,
                                  contexts, memory, modules, launches
                    src/ptx       fatbin → PTX → AST → CFG → structured SPIR-V
                    src/vk        Vulkan device, heap, batching, pipelines
```

## Compiler (`src/ptx`)

A self-contained library (no Windows, CUDA or Vulkan dependencies).

- `fatbin.cpp` decodes classic fatbin containers (raw or LZ4 PTX payloads).
- `parser.cpp` parses PTX into an AST. Parsing is permissive about which
  instructions appear; checks happen when an entry is compiled, so
  unsupported instructions in unlaunched kernels do not block module
  loading. Syntax the parser does not know (function calls with
  parenthesised arguments, body-level `.param`, dual-destination `setp`)
  still fails the whole module; classic PhysX PTX uses none of it.
- `cfg.cpp` builds basic blocks and structurizes the CFG with the dominator
  tree algorithm from Ramsey's "Beyond Relooper": loop headers become loop
  constructs, merge nodes become the continuation of a block construct.
  Conditionals are SPIR-V selections. A block construct that some branch
  must break out of is emitted as a run-once SPIR-V loop; one whose branches
  are all in tail position (typical if/else joins) is emitted inline, since
  drivers compile plain selections better. Branches that leave several
  constructs at once use a small "pending target" dispatch at each merge.
  Barriers therefore stay in structured, uniform control flow.
- `compiler.cpp` lowers each instruction. Notable semantics:
  - PTX registers are function variables (drivers promote them to SSA).
  - Generic loads/stores are resolved to global/shared/local by register
    provenance (PhysX only forms generic pointers to global memory).
  - FMA contraction mirrors ptxas: a `mul` without rounding modifier is fused
    into every `add`/`sub` (optionally through a `neg`) when all its uses are
    such adds in the mul's own basic block, preferring the first source
    operand; everything else is `NoContraction`. Blocks are counted after
    dead code is removed, so a branch over only dead instructions does not
    split one. This was verified against the NVIDIA JIT with `pxvk-opcheck`
    (including branch variants) and the UberCollision captures, and is what
    makes results bit-exact with native.
  - `div.approx` is `a * rcp(b)`, and `div.full` adds the hardware's
    operand scaling for |b| > 2^126; `min`/`max` use `NMin`/`NMax`; `cvt`
    to integer saturates and maps NaN to 0; `SignedZeroInfNanPreserve` is
    requested so unordered comparisons work on all drivers.
  - `.ftz` uses the Vulkan float-controls execution mode where available
    (Intel). NVIDIA's Vulkan driver supports neither flushing nor preserving
    fp32 denormals, and in practice preserves them, so `.ftz` is emulated by
    flushing results and inputs. A dataflow pass skips input flushes for
    registers that can never hold a denormal (results of other `.ftz`
    instructions, int-to-float conversions, normal immediates, and moves or
    selects of those), which removes most of them.
  - Memory operands keep their constant offset separate: an aligned access
    indexes as `(base >> k) + offset / size`, so drivers share the base
    computation between neighbouring accesses.
  - Warp-synchronous code (volatile shared stores) is bracketed by subgroup
    barriers and the pipeline requests full 32-wide subgroups. Devices that
    cannot guarantee subgroups of at least 32 lanes use warp emulation
    instead (below).

### Fallbacks for limited devices

Two code-generation modes exist only for devices that need them (lavapipe,
and most mobile drivers for the split heap). Desktop GPUs never use them; the
first log line reports both (`heap_windows=`, `warp_emulation=`).

- **Warp emulation** (`CompileOptions::emulate_warps`), when compute may run
  with subgroups narrower than 32 lanes that cannot be forced to 32
  (lavapipe: 8). Every volatile shared store is bracketed by *workgroup*
  barriers instead of subgroup barriers. All warps then advance store by
  store together, and since every load of a phase completes before any
  store of the next, this is exactly warp lockstep. Workgroup barriers must
  be reached by every invocation, so an `if` whose arms contain such stores
  (`if (tid < 32) { warp scan }`) is emitted predicated: both arms run for
  all invocations, each instruction guarded by its arm's condition and the
  barriers unconditional. Arms must be straight-line code, and every other
  warp-synchronous store must be reached by all invocations: not after a
  return, break, continue or multi-level exit that some invocations may
  take, and not in a loop without `bar.sync` (whose trip count may differ
  between invocations; CUDA already requires loops with `bar.sync` to be
  uniform). Kernels that break these rules fail to compile with an error
  rather than risk a hang. Emulated kernels are logged as `warp_emulated`.
- **Split heap** (`CompileOptions::heap_window_shift`), when
  `maxStorageBufferRange` is below 512 MiB. The heap is still one buffer,
  so copies and transfers are unchanged, but shaders see it through an
  array of storage-buffer descriptors ("windows") of the largest power of
  two that fits the range, selected per access by the address's high bits
  with non-uniform descriptor indexing. Vector heap accesses become scalar
  ones. One extra window at the end of the array is bound to a small dummy
  buffer, as are windows past the end of the current heap; window indices
  are clamped to it, so a garbage address never reaches live data. Requires `shaderStorageBufferArrayNonUniformIndexing`;
  up to 32 windows, also bounded by `maxPerStageDescriptorStorageBuffers`
  and `maxMemoryAllocationSize`.

### Shader interface

All kernels share one descriptor-set layout so the runtime needs a single
pipeline layout and descriptor set:

| binding | resource |
|---|---|
| 0, 1, 2 | the CUDA heap as `uint[]`, `uvec2[]`, `uvec4[]` (vector loads) |
| 3 | uniform launch block (dynamic offset): kernel parameters, a table of texture bindings, then the addresses of the module globals the kernel uses |
| 4 | scratch buffer for storage-backed shared memory |

Specialization constants carry the block size (`%ntid`), the shared-memory
size and the heap base address. Module-global addresses are per-launch data,
so a kernel's SPIR-V and pipelines do not depend on which module instance
launched it. Shared-memory sizes are rounded up to powers of two to bound the
number of pipelines when applications vary them per launch (PhysX cloth
does).

Shared memory is Workgroup memory when it fits the device; otherwise (Intel
iGPUs have 32 KiB, PhysX cloth needs 42 KiB) a variant places it in a
per-workgroup slice of the scratch buffer.

Textures are PhysX's `tex1Dfetch` on linear memory: the fetch is a bounds
checked heap load using the binding table in the launch block.

## Vulkan backend (`src/vk`)

- CUDA device memory is one device-local storage buffer. Device pointers are
  byte offsets into it (the first 64 KiB are reserved so null stays invalid).
  A Vulkan buffer cannot be resized, so growing means a new buffer and a
  copy. PhysX allocates one pool per context and later pools reuse the
  freed space, so the heap grows once at startup, to the size needed in
  64 KiB pages (FluidMark: 258 MiB for its 256 MiB pool). Small growth
  steps add at least a quarter so piecemeal allocators do not copy every
  time; if that extra does not fit beside the old buffer during the copy,
  growth retries with exactly what is needed. The log records each growth.
- Every operation (upload, fill, copy, dispatch) is recorded into one command
  batch, serialized with full memory barriers, which reproduces CUDA
  default-stream ordering. The batch is submitted only when the host must see
  results: synchronous device-to-host copies and synchronize calls. PhysX
  needs about one submission per simulation step.
- Host-to-device copies snapshot their source at call time through a staging
  ring (PhysX reuses pinned staging memory relying on prompt DMA).
- Pinned host memory (`cuMemHostAlloc`) is host-visible cached Vulkan memory,
  so asynchronous device-to-host copies are recorded GPU copies with no extra
  memcpy.
- Pipelines are created per (kernel, block size, shared-memory bucket).

## Compilation caches

Driver pipeline compilation dominates startup: the cloth solver alone takes
~300 ms per pipeline on NVIDIA. PhysX also unloads and re-registers all of its
modules on every scene change. Two caches remove this:

- Compiled kernels and their Vulkan shaders are cached for the process, keyed
  by PTX content hash, entry name and shared-memory variant. Re-registered
  modules reuse them, and their pipelines too.
- The `VkPipelineCache` persists in `%LOCALAPPDATA%\pxvk\pipelines-<driver
  cache UUID>.bin` (`PXVK_CACHE_DIR` overrides it, `off` disables it). It is
  written atomically about a second after new pipelines appear, so later runs
  start with no compilation at all.

With a warm cache, SampleCloth starts and switches scenes faster than native
CUDA (0.7 s vs 1.7 s at startup, 107–215 ms vs 146–260 ms per switch).

## Runtime (`src/runtime`)

- `loader.cpp`: CUDART builds an absolute System32 path for `nvcuda.dll`, so
  pxvk is injected by `pxvk-launch.exe` and patches CUDART's
  `LoadLibraryW/ExW` imports to return itself. It also patches the
  `LoadLibraryA/W/ExW` imports of `PhysXDevice.dll` and `PhysXCore.dll` so
  that their `nvapi.dll` loads return `pxvk-nvapi.dll` from the same folder.
  `PhysXDevice` loads NVAPI with `LOAD_LIBRARY_SEARCH_SYSTEM32`, so an
  application-local `nvapi.dll` is not found. The redirect is the default on
  every machine, NVIDIA included, so hardware selection does not depend on
  the installed driver or on whether its GPU is enabled. `PXVK_NVAPI=system`
  keeps the driver's NVAPI. The redirect also does not apply if
  `pxvk-nvapi.dll` is absent.
- `nvapi.cpp` (`pxvk-nvapi.dll`): PhysX picks hardware PhysX only if NVAPI
  reports a PhysX-capable GPU. This exports `nvapi_QueryInterface` with the
  eight interfaces ZLUDA implements (initialize/unload, driver version,
  compute-capable GPU enumeration including the private V2 id `0x36e39e6b`,
  GPU id ↔ handle) and reports one GPU. Other ids return null ("no
  implementation").
- `runtime.cpp`: the driver API plus the six private export tables CUDART
  uses instead of most public entry points (module registration from the
  fatbin wrapper, context creation, context-local storage, ...; GUIDs and
  slots are in the "private export tables" section). The device presents an
  sm_52 persona whose
  attributes mirror a real NVIDIA driver (CUDART validates against them; e.g.
  texture bindings fail if limits read as zero). Allocations follow the native
  driver's alignment (4 KiB, 2 MiB for large blocks): PhysX sub-allocates
  inside a pool and CUDART rejects texture bindings that are not 512-byte
  aligned. New allocations are zeroed, as the native driver's are in
  practice: SampleParticleFluid reallocates its pool on scene changes and
  diverged from native when a reused block kept the previous scene's data.
- Modules: PTX is parsed at registration, globals are allocated and
  initialized immediately, kernels compile on first use (or on the first
  attribute query) and are cached per module.
- `stubs.cpp` (generated by `tools/gen-stubs.py` from
  `unsupported-exports.txt` and the CUDA toolkit headers): unsupported
  exports that log their name and return `CUDA_ERROR_NOT_SUPPORTED`. Like
  the real entry points they are `__stdcall` and declare parameters of the
  real sizes, so they pop exactly the caller's arguments.

## Vulkan requirements

Required: Vulkan 1.1 with a compute queue, `robustBufferAccess` and
768-invocation workgroups. The CUDA heap (PhysX cloth alone allocates a
128 MiB pool, FluidMark a 256 MB one) needs either a `maxStorageBufferRange`
that covers it or non-uniform storage-buffer descriptor indexing for the
split heap. No buffer device address or 8/16-bit storage is used. Float
atomics are optional: where the device has `VK_EXT_shader_atomic_float`'s
`shaderBufferFloat32AtomicAdd` (NVIDIA, lavapipe), global float atomic adds
use it; otherwise, and always in shared memory, they are a compare-and-swap
loop on 32-bit integer atomics. Native shared float atomics would need
`VK_KHR_workgroup_memory_explicit_layout` and were measured not to be worth
it (at most 1 µs per cloth normal launch on the RTX 3070; `docs/testing.md`).

Optional, with fallbacks:

| Feature | Core in | Used for | Without it |
|---|---|---|---|
| float controls: fp32 `DenormFlushToZero` | 1.2 | native `.ftz` | emulated (slower on small launches; NVIDIA today) |
| float controls: `SignedZeroInfNanPreserve` | 1.2 | correct unordered comparisons | NaN comparisons may be wrong on some drivers |
| `subgroupSizeControl` + `computeFullSubgroups` | 1.3 | 32-wide subgroups for warp-synchronous kernels | fine if compute never runs narrower than 32 lanes; otherwise warp emulation |
| `maxStorageBufferRange` ≥ 512 MiB | limit | the whole heap in one descriptor | split heap (needs `shaderStorageBufferArrayNonUniformIndexing`, 1.2) |
| `shaderInt64` | 1.0 feature | PTX using 64-bit integers | such kernels fail to compile (PhysX has not needed it) |
| ≥ 42 KiB shared memory | limit | cloth solver shared memory as Workgroup memory | storage-buffer fallback (slower) |
| `VK_KHR_pipeline_executable_properties` | ext | `PXVK_PIPELINE_STATS` | — |

In practice this is any GPU with a current Windows Vulkan driver: NVIDIA
Kepler and later, AMD GCN and later, Intel Gen9 (Skylake) and later. With
the fallbacks above it also runs on Mesa's lavapipe (software Vulkan), which
behaves like a mobile driver: a 128 MiB storage range and 8-lane subgroups.

## Known limitations

- `div.rn`/`sqrt.rn` are not correctly rounded (Vulkan precision); PhysX
  does not use them.
- `%clock` reads the subgroup clock of `VK_KHR_shader_clock` and is 0 on
  devices without it. Only PhysX 2.7.5 uses it, to seed the random numbers
  of fluid force fields, which differ between runs natively too.
- Float atomic add (`atom`/`red .add.f32`) rounds to nearest and, in
  global memory, flushes subnormals as NVIDIA does, both with the native
  Vulkan float atomic (where available, global memory only) and with the
  compare-and-swap loop (opcheck exact on NVIDIA, Intel and lavapipe, with
  `PXVK_NO_FLOAT_ATOMICS` forcing the loop). In shared memory NVIDIA keeps subnormal inputs and pxvk flushes
  them (NVIDIA's Vulkan driver cannot preserve subnormals). PhysX 2.8.4's
  cloth normal kernels start shared sums at 0 and add only flushed values,
  and opcheck shows that pattern bit-exact; they could differ only if a
  partial sum cancelled to a subnormal and a later addend were near
  10^-38, which would need triangle edges around 10^-15. Accumulation
  order follows thread scheduling, natively too, so these kernels are not
  bit-exact against native in practice.
- On non-NVIDIA GPUs `rcp`/`rsqrt`/`sqrt` differ from NVIDIA hardware by
  1–2 ULP; iterative solvers amplify this into small numerical drift.
- Results are not yet fully 1:1 with native on NVIDIA: every instruction
  variant is bit-exact, but 2 of 90 replayed SampleCloth launches
  (`UberSolver`, scene 5) differ by ≤ 4 ULP in some words. These kernels
  use no atomics; the cause is ptxas leaving some products unfused for
  reasons no structural rule found so far explains (in one x/y/z triple
  of identical `sub`s it fuses x and y but not z, which accounts for about
  a fifth of the difference). In a full sweep only SampleCloth scene 5
  differs from native (under 0.1% of the pixels); every other cloth scene
  and all SampleParticleFluid scenes are pixel-identical. Starting point:
  replay the "ok" cloth captures (launches 6643 and 6646).
- Warp emulation and the split heap are slower than the normal paths (two
  workgroup barriers per warp-synchronous store; scalar heap accesses).
  Only devices that need them pay for that.
- 64-thread `UberSolver` launches (SampleCloth scenes 5 and 9) still pay
  30-80% of their GPU time for `.ftz` emulation on NVIDIA; frame rates are
  above native regardless.
- Streams and events are serialized on one queue; no host-mapped memory,
  arrays, surfaces or graphics interop (unused by the targets). The device
  still reports `CAN_MAP_HOST_MEMORY` like the NVIDIA card it imitates, and
  `cuMemHostAlloc` accepts `DEVICEMAP` without providing a device pointer.
- Events are timestamped when `cuEventRecord` is called, not when the GPU
  reaches them, because work is batched until a synchronization:
  `cuEventElapsedTime` measures host time. Use `PXVK_PROFILE` for GPU times.
- Contexts are simplified to what PhysX does (one context per thread at a
  time): `cuCtxCreate` replaces the current context instead of pushing it,
  destroying a context clears it only from the calling thread's stack, and
  `cuCtxSetCurrent(NULL)` clears rather than pops.
- One process-wide lock covers every driver call, including GPU waits. A
  thread killed by `ExitProcess` while inside a call leaves it held, so a
  CUDA call from DLL detach in that state would hang.
