# Runtime PTX-to-SPIR-V architecture plan

## Goal

Ship pxvk as a focused classic 32-bit GPU PhysX compatibility runtime that can
consume PTX embedded in an application's classic CUDA fatbins, translate only
launched supported entries to SPIR-V, cache the result, and dispatch through
Vulkan without an application-specific kernel catalog or a required offline
capture/staging pass.

This remains scoped to the finite PTX and CUDA behavior justified by
[`charter.md`](charter.md). Completing this plan does not by itself qualify
Mirror's Edge gameplay; captured-state replay and game-route validation remain
separate acceptance gates.

## Status conventions

- `[ ]` not started
- `[x]` complete and verified
- A phase is complete only when all of its tasks and exit criteria are checked.
- Check a task only in the same change that provides its implementation or
  verification evidence.
- Record material design changes in the relevant task or architecture document;
  do not silently broaden the project scope.

## Contracts to preserve

- [ ] Keep the runtime and CUDA-visible ABI PE32 with 32-bit pointers and
  `size_t`.
- [ ] Keep the qualified classic PTX 4.1 / `sm_52` virtual device persona,
  including CUDA-visible warp size 32 and the 48 KiB shared-memory limit.
- [ ] Keep the observed private CUDART/PhysX export-table layouts exact.
- [ ] Keep one logical CUDA device and ordered synchronous default-stream
  execution until target traces require more.
- [ ] Keep the fixed Vulkan descriptor ABI: heap 0, parameters 1, constants 2,
  texture metadata 3, and storage-backed shared scratch 4.
- [ ] Keep unsupported CUDA APIs, PTX variants, launch dimensions, and fatbin
  forms explicit failures rather than approximations.
- [ ] Keep proprietary captured PTX and generated application shaders out of
  the repository and release package.

## Phase 0 — Re-establish a data-driven AOT baseline

The AOT path establishes module identity, metadata, generic launch, and cache
contracts before runtime compilation is introduced. It remains the deterministic
prewarm and fallback path after JIT exists.

### Classic fatbin decoding

- [ ] Move classic fatbin decoding into a small runtime/compiler component with
  no CUDA or Vulkan dependencies.
- [ ] Validate the wrapper, main header, file headers, offsets, and all checked
  size arithmetic before reading payloads.
- [ ] Enforce an explicit 64 MiB decoded-data safety cap.
- [ ] Support the observed raw PTX payload form.
- [ ] Support the observed raw-LZ4 PTX payload form with bounded decompression.
- [ ] Reject empty PTX, truncated members, ambiguous multiple PTX payloads,
  Zstd payloads, and cubin-only fatbins explicitly.
- [ ] Add focused fixtures for valid raw, valid LZ4, truncated, oversized,
  malformed, multiple-PTX, Zstd, and cubin-only inputs.
- [ ] Compute a stable SHA-256 identity over the exact decoded PTX bytes.

### Versioned deployment metadata

- [ ] Define one in-memory `ModuleMetadata` contract containing PTX identity,
  PTX version/target/address size, every declared routine, module globals and
  constants, and ordered legacy texture references.
- [ ] Define one in-memory `KernelMetadata` contract containing entry identity,
  packed parameter bytes, `.maxntid`, static shared/local storage, constant
  layout, complete ordered texture layout, descriptor ABI version, and
  shared-memory lowering mode.
- [ ] Define and document versioned module-sidecar serialization.
- [ ] Define and document versioned kernel-sidecar serialization.
- [ ] Reject unknown schema versions, address sizes, descriptor ABI versions,
  malformed paths, duplicate identities, and inconsistent resource layouts.
- [ ] Preserve unused texture-reference slots because their positions affect
  every subsequent texture index.
- [ ] Make all sidecar paths relative to the sidecar directory and reject paths
  that escape it.

### Data-driven runtime catalog

- [ ] Load module and kernel sidecars once during runtime initialization.
- [ ] Match a registered module by exact PTX SHA-256 and byte count, never by an
  entry name alone.
- [ ] Validate `cuModuleGetFunction` names against routines declared by that
  exact module.
- [ ] Permit handles for declared but untranslated routines so PhysX capability
  discovery can complete.
- [ ] Return an explicit unsupported result when a declared untranslated
  routine is launched.
- [ ] Derive global names, sizes, offsets, and allocation behavior from module
  metadata.
- [ ] Derive constant staging and ordered texture metadata from kernel/module
  metadata.
- [ ] Remove `kKernelSpecs`, `Profile`, handwritten constant tables, handwritten
  texture tables, `global_size`, and application-specific shader lookup from
  `runtime.cpp`.
- [ ] Delete the production `pxvk_runtime_probe` special case after the smoke
  test uses the normal fatbin and metadata route.

### Generic launch artifacts

- [ ] Emit local size through SPIR-V `LocalSizeId` specialization IDs 0, 1, and
  2.
- [ ] Make `%ntid.x` read the same local-size-X specialization constant used by
  Vulkan pipeline creation.
- [ ] Pass storage-backed shared stride as a dispatch push constant.
- [ ] Keep grid dimensions, local size, and dynamic shared bytes out of kernel
  artifact identity.
- [ ] Keep local X/Y/Z in Vulkan pipeline identity.
- [ ] Keep dynamic shared stride out of Vulkan pipeline identity.
- [ ] Retain X-only launch rejection until Y/Z built-ins and guarded shared-slice
  linearization are implemented together from workload evidence.
- [ ] Validate local dimensions and their product against PTX `.maxntid`, the
  virtual CUDA profile, and Vulkan device limits before pipeline creation.
- [ ] Guard storage-backed scratch size multiplication against overflow and
  device buffer-range limits.

### Deployment profile cleanup

- [ ] Replace the hardcoded `$jobs` array in `prepare-sample.ps1` with a
  versioned SampleCloth deployment profile.
- [ ] Store only selected entry names, output stems, and evidenced shared-memory
  lowering policy in the profile.
- [ ] Do not encode captured grid dimensions, local sizes, or dynamic shared
  byte counts in the deployment profile.
- [ ] Make the staging script discover the exact PTX module containing each
  selected entry and fail on missing or ambiguous matches.
- [ ] Generate module metadata for every registered PTX module, including
  modules with no translated entries.

### Phase 0 exit criteria

- [ ] Runtime smoke registers and launches a real test fatbin through the same
  metadata route used by PhysX.
- [ ] A module-sidecar identity mismatch fails before any module resources or
  function handles are exposed.
- [ ] One generic shader executes at two supported local sizes without shader
  regeneration.
- [ ] One storage-backed shader executes at two dynamic shared-memory sizes
  without shader regeneration or a second pipeline.
- [ ] SampleCloth scenes 1 through 9 run without unsupported-launch, module
  mismatch, Vulkan validation, or backend errors.
- [ ] The focused native/Vulkan differential fixtures remain exact.

## Phase 1 — Extract the reusable PTX compiler

The compiler is an internal static library. The command-line tools remain thin
adapters for development, inspection, and deterministic AOT generation.

### Build and source boundaries

- [ ] Add one `pxvk-ptx` static CMake target containing the parser, normalized
  schema-5 IR, selected-entry lowering, SPIR-V emitter, and metadata helpers.
- [ ] Move the frontend `main()` into a thin CLI translation unit linked to
  `pxvk-ptx`.
- [ ] Move the emitter `main()` into a thin CLI translation unit linked to
  `pxvk-ptx`.
- [ ] Keep compiler code independent of Windows, CUDA facade state, Vulkan
  objects, filesystem cache policy, and application profiles.
- [ ] Link the existing frontend and emitter executables against the library
  without changing their supported command-line behavior.
- [ ] Make the PE32 runtime build link the same static compiler target.
- [ ] Keep nlohmann JSON and SPIR-V headers pinned through one shared CMake
  dependency definition rather than fetching separate copies per executable.

### Compiler facade

- [ ] Expose a metadata inspection operation accepting exact PTX bytes and
  returning `ModuleMetadata`.
- [ ] Expose a selected-entry compile operation accepting exact PTX bytes, an
  exact entry name, and finite lowering options.
- [ ] Return SPIR-V words and `KernelMetadata` together as one `CompiledKernel`
  result.
- [ ] Keep schema-5 JSON internal to the compiler/CLI boundary; runtime callers
  must not serialize and reparse manifests.
- [ ] Report PTX identity, entry name, source line/column, compilation stage,
  and unsupported form in compilation diagnostics.
- [ ] Catch all compiler exceptions at CUDA ABI boundaries so none cross an
  exported `nvcuda.dll` function.
- [ ] Validate only the selected entry and reachable helpers for executable
  support; unsupported unlaunched routines must not block module metadata.
- [ ] Reject module variable initializers until a captured target requires and
  tests initializer upload semantics.

### Phase 1 exit criteria

- [ ] Existing frontend and emitter CTests pass through the library-backed CLIs.
- [ ] Committed probe manifests are byte-identical before and after extraction.
- [ ] Emitted probe SPIR-V is byte-identical before and after extraction.
- [ ] All 17 captured Mirror's Edge modules retain their current metadata and
  parse result.
- [ ] All translated-kernel fixtures retain exact native/Vulkan results.
- [ ] A PE32 smoke executable links and invokes the compiler facade in memory
  without temporary files or subprocesses.


## Phase 2 — Add the persistent compiler cache

### Cache identity and format

- [ ] Define a versioned cache key containing exact PTX SHA-256, exact entry
  name, compiler schema/version, SPIR-V/Vulkan target, descriptor ABI version,
  and shared-memory lowering mode.
- [ ] Hash the complete logical cache key into a filename so entry names never
  become paths.
- [ ] Exclude grid dimensions, specialized local size, and dynamic shared bytes
  from compiled-kernel cache identity.
- [ ] Store enough metadata beside SPIR-V to validate identity, resource layout,
  word count, and schema before loading it.
- [ ] Treat unknown, malformed, truncated, inconsistent, or corrupt cache files
  as misses.
- [ ] Write cache entries atomically through a temporary sibling file followed
  by rename.

### Cache policy

- [ ] Use a writable per-user application-data directory by default.
- [ ] Provide one cache-directory override for deterministic tests and advanced
  deployment.
- [ ] Do not require write access to the application or runtime directory.
- [ ] Resolve kernels in this order: process memory, exact AOT/prewarmed bundle,
  persistent JIT cache, compiler.
- [ ] Ensure a compiler/schema upgrade invalidates older cache entries without
  deleting unrelated versions.
- [ ] Log cache identity, source, hit/miss, compile duration, and rejection
  reason without logging proprietary PTX contents.

### Phase 2 exit criteria

- [ ] A clean cache produces misses followed by atomic cache files.
- [ ] A warm cache produces hits with byte-identical SPIR-V and metadata.
- [ ] Truncated, corrupted, wrong-entry, wrong-module, and wrong-schema cache
  fixtures are rejected and rebuilt.
- [ ] Concurrent processes cannot observe a partially written cache entry.
- [ ] A read-only/unavailable cache directory degrades to in-memory compilation
  with a clear log message rather than preventing execution.

## Phase 3 — Make Vulkan consume compiled kernels

### Backend API

- [ ] Replace `DispatchInfo::shader_path` with an immutable compiled-kernel or
  SPIR-V code handle.
- [ ] Keep PTX parsing, JSON, cache paths, deployment profiles, and application
  names out of `VulkanBackend`.
- [ ] Build shader modules directly from validated in-memory SPIR-V words.
- [ ] Key Vulkan pipelines by compiled-kernel identity plus local X/Y/Z.
- [ ] Specialize local size through IDs 0, 1, and 2 at pipeline creation.
- [ ] Add the storage-shared stride push-constant range to the fixed pipeline
  layout when required by kernel metadata.
- [ ] Pass shared stride per dispatch without rebuilding the pipeline.
- [ ] Reuse or grow parameter, constant, texture, and scratch buffers without
  changing the kernel's resource contract.
- [ ] Keep synchronous queue submission and host visibility semantics unchanged.

### Virtual device consistency

- [ ] Centralize the qualified virtual CUDA device persona in one immutable
  profile.
- [ ] Make `cuDeviceGetAttribute`, `cuDeviceGetProperties`,
  `cuFuncGetAttribute`, launch validation, and backend capability checks derive
  from that profile plus actual Vulkan limits.
- [ ] Do not advertise a capability that the launch path rejects.
- [ ] Do not expose extra Vulkan limits when doing so would change PhysX kernel
  selection away from the qualified `sm_52` behavior.

### Phase 3 exit criteria

- [ ] No production Vulkan dispatch reads SPIR-V from an application-specific
  filename.
- [ ] Repeated launches reuse the expected shader module and pipeline.
- [ ] Changing only dynamic shared bytes reuses the pipeline.
- [ ] Changing local size creates or reuses the corresponding specialized
  pipeline without recompiling PTX.
- [ ] All Vulkan buffer-range, workgroup, grid, scratch-size, and push-constant
  validation failures are deterministic and logged.

## Phase 4 — Add lazy runtime compilation

### Module registration

- [ ] Decode and retain exact bounded PTX bytes during classic fatbin
  registration.
- [ ] Inspect module metadata without compiling all routine bodies.
- [ ] Store PTX identity and immutable module metadata in `ModuleRecord`.
- [ ] Create function, global, and texture handles only from the registered
  module's metadata.
- [ ] Reclaim module compiler/cache references and module-owned globals safely
  on unload.
- [ ] Ensure unsupported unlaunched entries do not add compiler latency or
  prevent initialization.

### First-launch compilation

- [ ] Validate launch parameters, block limits, parameter bytes, and shared
  memory before compilation or Vulkan state mutation.
- [ ] Resolve an existing in-memory, AOT, or disk-cache kernel before compiling.
- [ ] Compile exactly one selected entry on a cache miss.
- [ ] Derive shared-memory lowering from PTX metadata and qualified backend
  capabilities, not application names.
- [ ] Store successful compiled kernels by immutable cache identity.
- [ ] Memoize compilation failure for the process to avoid repeated work and
  log spam.
- [ ] Map compiler failures to stable CUDA errors while preserving complete
  compiler diagnostics in `pxvk-runtime.log`.
- [ ] Keep compilation out of `DllMain` and loader-notification callbacks.
- [ ] Measure first-run compile time and PE32 private memory on SampleCloth
  before adding parser-IR retention or eviction complexity.

### Phase 4 exit criteria

- [ ] SampleCloth starts and simulates with the staged `spirv` directory absent.
- [ ] The first run records deterministic compiler cache misses and successful
  selected-entry compilation.
- [ ] The second run records cache hits and performs no compilation for the same
  entries.
- [ ] JIT and AOT output are byte-identical for the same PTX, entry, compiler
  version, and lowering mode.
- [ ] Querying unsupported routines succeeds only when declared; launching one
  fails explicitly without blocking other entries in the module.
- [ ] No temporary manifest or SPIR-V files are required for an uncached launch.

## Phase 5 — Packaging and release validation

### Runtime package

- [ ] Produce a package containing `nvcuda.dll`, `pxvk-launch.exe`, and only
  evidenced optional runtime components.
- [ ] Statically include the PTX compiler in `nvcuda.dll`; do not ship a compiler
  service or require the frontend/emitter executables.
- [ ] Confirm the package requires no Python, CMake, Git, CUDA toolkit, NVIDIA
  driver, writable application directory, or captured PTX directory.
- [ ] Confirm MinGW and MSVC PE32 builds have documented runtime dependencies.
- [ ] Keep AOT/prewarm tooling available to developers without making it a
  normal-user prerequisite.
- [ ] Document cache location, cache override, logging, device selection,
  supported platform, and explicit unsupported behavior.

### Demo and workload validation

- [ ] Run the complete portable `run-tests.ps1` suite.
- [ ] Run all sixteen Mirror's Edge launched-entry differential fixtures and
  preserve exact native/Vulkan outputs and guard ranges.
- [ ] Run the ordered deformable and fluid fixtures through the runtime JIT
  route, not only standalone Vulkan runners.
- [ ] Run SampleCloth scenes 1 through 9 from a clean package with no staged
  SPIR-V.
- [ ] Check scene switching, self-collision, pressure, attachment, remap, pipe,
  and tearing paths for unsupported launches and backend errors.
- [ ] Run SampleParticleFluid from the same package and JIT/cache path.
- [ ] Validate first-run cache misses and second-run cache hits on every
  qualified x86 Vulkan vendor available.
- [ ] Complete the captured-state replays identified in
  [`research/kernel-status.md`](research/kernel-status.md) before claiming
  in-game PhysX correctness.
- [ ] Complete the named Mirror's Edge routes and deferred NVAPI integration
  gate before claiming Mirror's Edge compatibility.

### Release exit criteria

- [ ] A clean machine can launch a supported PhysX demo using only the packaged
  runtime and an installed x86 Vulkan loader/driver.
- [ ] No application-specific kernel name, launch shape, resource table, or
  shader filename remains in production runtime source.
- [ ] No proprietary PTX or generated application SPIR-V is included in source
  control or release archives.
- [ ] Unsupported modules and entries fail explicitly with actionable logs.
- [ ] Supported cold-cache and warm-cache runs produce equivalent simulation
  results.
- [ ] Release notes state the exact qualified applications, workloads, devices,
  and remaining validation gaps without implying general CUDA support.

## Deferred work

These items are not part of this plan unless new target evidence changes the
project charter:

- General CUDA or CUDART compatibility.
- 64-bit host/runtime support.
- General streams, events, asynchronous overlap, or multiple contexts.
- CUDA arrays, surfaces, texture objects, or graphics interop.
- Cubin/SASS execution or modern PTX/fatbin formats.
- Multi-GPU and peer access.
- General Y/Z launch support without matching PTX built-ins and shared-memory
  linearization.
- Speculative PTX instruction families not present in an evidenced target.
- A helper process, compiler service, plugin system, or stable public
  compiler ABI.
