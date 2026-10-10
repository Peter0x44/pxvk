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
  unsupported code in unlaunched kernels never blocks module loading.
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
    such adds, preferring the first source operand; everything else is
    `NoContraction`. This was verified against the NVIDIA JIT with
    `pxvk-opcheck` and is what makes results bit-exact with native.
  - `div.full`/`div.approx` are `a * rcp(b)` with the hardware's operand
    scaling; `min`/`max` use `NMin`/`NMax`; `cvt` to integer saturates and
    maps NaN to 0; `SignedZeroInfNanPreserve` is requested so unordered
    comparisons work on all drivers.
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
    barriers and the pipeline requests full 32-wide subgroups.

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
  The buffer grows on demand, preserving contents.
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
  application-local `nvapi.dll` is not found. The redirect applies only when
  the system has no `nvapi.dll` (SysWOW64 for this 32-bit process), so with an
  NVIDIA driver installed PhysX keeps using the real NVAPI. It also does not
  apply if `pxvk-nvapi.dll` is absent.
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
  aligned.
- Modules: PTX is parsed at registration, globals are allocated and
  initialized immediately, kernels compile on first use (or on the first
  attribute query) and are cached per module.
- `stubs.cpp` (generated by `tools/gen-stubs.py`): unsupported exports that
  log their name and return `CUDA_ERROR_NOT_SUPPORTED`.

## Vulkan requirements

Required: Vulkan 1.1 with a compute queue, `robustBufferAccess`, a
`maxStorageBufferRange` large enough for the application's CUDA memory (one
buffer holds it all; PhysX cloth alone allocates a 128 MiB pool) and
768-invocation workgroups. No descriptor indexing, buffer device address,
8/16-bit storage or float atomics are used.

Optional, with fallbacks:

| Feature | Core in | Used for | Without it |
|---|---|---|---|
| float controls: fp32 `DenormFlushToZero` | 1.2 | native `.ftz` | emulated (slower on small launches; NVIDIA today) |
| float controls: `SignedZeroInfNanPreserve` | 1.2 | correct unordered comparisons | NaN comparisons may be wrong on some drivers |
| `subgroupSizeControl` + `computeFullSubgroups` | 1.3 | 32-wide subgroups for warp-synchronous kernels | allowed if the device never runs compute narrower than 32 lanes; refused otherwise |
| `shaderInt64` | 1.0 feature | PTX using 64-bit integers | such kernels fail to compile (PhysX has not needed it) |
| ≥ 42 KiB shared memory | limit | cloth solver shared memory as Workgroup memory | storage-buffer fallback (slower) |
| `VK_KHR_pipeline_executable_properties` | ext | `PXVK_PIPELINE_STATS` | — |

In practice this is any GPU with a current Windows Vulkan driver: NVIDIA
Kepler and later, AMD GCN and later, Intel Gen9 (Skylake) and later. Mobile
drivers commonly cap `maxStorageBufferRange` at 128 MiB (see
`docs/roadmap.md`).

## Known limitations

- `div.rn`/`sqrt.rn` are not correctly rounded (Vulkan precision); PhysX
  does not use them.
- On non-NVIDIA GPUs `rcp`/`rsqrt`/`sqrt` differ from NVIDIA hardware by
  1–2 ULP; iterative solvers amplify this into small numerical drift.
- Results are not yet fully 1:1 with native on NVIDIA: every instruction
  variant is bit-exact, but 6 of 202 replayed SampleCloth launches (a few
  `UberCollision`/`UberSolver` launches) differ by ≤ 4 ULP in a handful of
  words, likely another ptxas contraction pattern. Chaotic scenes amplify
  this: SampleCloth scene 7 visibly diverges from native after ~100 frames,
  and SampleParticleFluid scenes 2, 3 and 5 drift slightly (native runs are
  deterministic). The physics stays plausible. Starting points: replay the
  "ok" cloth captures with `--verbose`, and diff native vs pxvk captures of
  fluid scene 2 with `tools/capture-diff.py` (its captured launches replay
  exactly, so the divergence comes from other frames or non-kernel work).
- Warp-synchronous kernels need 32-lane (or wider) subgroups; on devices
  that may run narrower subgroups without size control they fail to load.
- 64-thread `UberSolver` launches (SampleCloth scenes 5 and 9) still pay
  30-80% of their GPU time for `.ftz` emulation on NVIDIA; frame rates are
  above native regardless.
- Streams and events are serialized on one queue; no host-mapped memory,
  arrays, surfaces or graphics interop (unused by the targets).
