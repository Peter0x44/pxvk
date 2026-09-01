# Validation plan for the Mirror's Edge target

## Status

Validation is split into independently useful gates. No game run can substitute
for a failed API, PTX, or workload-output gate, and no screenshot can establish
numerical correctness.

The project has a committed native NVIDIA semantic oracle for f32,
`bar.sync 0`, invocation-private local memory, and signed-coordinate `u32`/`f32`
linear textures. Repeatable native RTX fixtures for all eleven translated fluid
kernels and all three captured deformable kernels pass bit-for-bit through
Vulkan on Intel and RTX, including guard ranges. An ordered three-kernel
deformable step also passes with one shared heap and explicit compute barriers.
It is trace-shaped but synthetic: zero shapes and zero solver iterations.
The focused fluid fixtures also cover only named synthetic branches, not broad
input distributions. Captured in-game state remains a later integration gate.

This document defines pass/fail evidence. It does not select the permanent test
framework, runtime language, shader compiler, or architecture.

## Validation principles

1. **Native NVIDIA is the behavioral oracle.** Compare the same x86 API call,
   PTX, launch, and input against driver 610.88 on the RTX 3070 reference system.
2. **Exact by default.** Compare integer values and floating-point bit patterns
   exactly. A tolerance is permitted only for one named approximate operation
   after its native and Vulkan difference is measured and documented.
3. **Test the narrow contract.** Cover the 35 active public CUDA exports, active
   Dark API slots, 16 launched entries, and 120 launched opcode variants. Do not
   add generic CUDA tests as a proxy for target evidence.
4. **Separate contract from hardening.** Native-observed results are conformance
   checks. Bounds checks, allocation failures, stale-handle rejection, and
   malformed inputs may be safety tests, but must not be presented as observed
   NVIDIA error compatibility.
5. **Reproduce before preserving.** Record a native result at least three times
   and require byte-identical normalized output before making it a fixture.
6. **Pin every input.** A result report names the executable, driver, GPU,
   Vulkan device/driver, PTX/fatbin, test program, and fixture hashes.
7. **Do not bless a mismatch.** Classify it as capture instability, API state,
   parser, lowering, synchronization, floating point, texture, or runtime
   integration before changing expected output.
8. **Visual success is integration evidence only.** The game may look plausible
   while a simulation buffer is numerically wrong.

## Existing validation assets

| Asset | Current evidence | Role in the plan |
|---|---|---|
| `tools/cuda-oracle/probe.ptx` | Native PTX microkernels for f32, barrier, aligned local memory, module constants, and `u32`/`f32` texture behavior | Run unchanged through native CUDA and the Vulkan translator |
| `tools/reference/expected/reference-workstation.json` key `cuda.semantic_oracle` | Committed exact NVIDIA result bits | Primary standalone semantic fixture |
| `tools/cuda-oracle/main.c` | PE32/i386 Driver API launcher | Native reference runner and shape for a backend runner |
| `tools/nvcuda-trace/tests/launch_params_test.c` | Standard/reversed packed launch tokens and malformed/unreadable inputs | Preserve `pxvk_parse_launch_extra` behavior; extend only when the launch ABI changes |
| `tools/nvcuda-trace/tests/proxy_smoke.c` | `LoadLibrary`, export lookup, `cuInit`, and enumeration | Shape for an out-of-process pxvk facade smoke test |
| `tools/nvcuda-trace/tests/injected_smoke.c` | Startup-injected `nvcuda.dll` discovery and calls | Shape for launcher/injection validation |
| `tools/vulkan-probe/main.c` | x86 loader, device features, and quantitative limits | Input to startup-gate tests, not a compute-correctness test |
| `tools/vulkan-probe/memory_smoke.cpp` plus generated SPIR-V | Real 32 MiB buffers; f32/FMA/conversion, workgroup barrier, invocation-private local storage, module constants, and integer/float texture output exactly match CUDA on Intel and RTX | Completed reference-system standalone Vulkan semantic gate; preserve for deferred AMD run |
| `tools/ptx-frontend/main.cpp` | Strict parser validates all 17 modules, 4,174 blocks, 5,935 edges, all label/register references, routine storage-symbol layouts, and finite result/use-type contracts for all 147 variants; all 16 launched layouts match independent analysis | Completed whole-corpus grammar/metadata/CFG/dataflow/type gate; selected kernels emit SPIR-V |
| 17 local captured PTX modules | Complete corpus accepted by both ZLUDA and the project frontend; hashes remain local/cataloged | Local parse/normalization coverage without committing proprietary PTX |
| Three native game traces | Fluid, deformable, and confirmed broken-glass execution | Route and workload-presence oracle |

The original CUDA oracle report SHA-256 is
`38fc7ef4be8863cc55b0a5408cbeab5e20f6f81de852a9b7818339897cc88011`.
It is stored with the native PhysX pipeline and all reference Vulkan reports in
the single reference bundle. Its payload contains 14 named f32 results,
64-element barrier and local-memory results, two module-constant payloads, and
12 signed-coordinate rows for each integer and float texture.

## Validation layers

| Layer | Runs without GPU | Needs Vulkan GPU | Needs native NVIDIA | Needs proprietary game data |
|---|---:|---:|---:|---:|
| Host utilities and state invariants | yes | no | no | no |
| CUDA x86 facade and Dark API contract | mostly | initialization path | differential only | local corpus cases only |
| PTX parser and static lowering | yes | no | differential only | local whole-corpus pass |
| Standalone SPIR-V execution | no | yes | expected fixture already captured | no |
| Real-kernel replay | no | yes | fixture capture | yes, kept local |
| Mirror's Edge smoke routes | no | yes | comparison runs | yes |

A change runs the lowest layer that can observe it. Milestone claims require all
applicable higher layers; routine parser edits do not require launching the
game.

## CUDA facade and runtime contract checks

### PE32 ABI and symbol surface

A 32-bit client must:

- load the candidate `nvcuda.dll` by the same mechanism used for the game;
- resolve every one of the 196 names queried by the captured CUDART until the
  deferred NULL-versus-stub probe narrows that list;
- call every active export using its captured `__stdcall` signature;
- verify repeated calls return with an intact x86 stack;
- verify queried-only stubs are never reached on the synthetic active path and
  return a non-success result if deliberately invoked.

This is symbol and ABI coverage, not implementation of the 161 unused CUDA
features.

### Initialization and state

One synthetic client should reproduce the observed lifecycle:

```text
cuInit and device enumeration
  -> six cuGetExportTable requests
  -> dark ctx_create_v2_bypass(flags=20, device=0)
  -> current-context queries
  -> CLS get/get/put/get
  -> module registration and lookup
  -> allocation, copies, launch, synchronize
  -> frees, cuCtxDetach, dark heap release
```

Checks:

- ordinal 0 and current-context handles remain stable;
- the first two CLS `get` calls return `CUDA_ERROR_INVALID_HANDLE`;
- `put` followed by `get` returns the stored pointer and `CUDA_SUCCESS`;
- modules, functions, globals, and texture references remain valid through
  context teardown;
- normal teardown releases pools and does not require module unload, streams,
  events, arrays, or graphics interop.

Do not create native-result assertions for public invalid arguments or
asynchronous failures: no such result was captured.

### Device properties

Test only values that affect caller behavior or backend feasibility:

- one device at ordinal 0;
- 32-bit `CUdeviceptr` and saturated x86 total-memory behavior;
- warp size 32;
- launch dimension/shared-memory values sufficient for captured launches;
- the private attribute `0x20080402` behavior once the facade implements it.

Clock, PCI, cache, surface, and unused texture limits are trace fixtures for
investigation, not mandatory RTX constants. A later conservative-value probe
must determine which can differ without changing PhysX behavior.

### Memory and ordering

A synthetic test allocates one 32 MiB host pool and one 32 MiB device pool, then
checks:

1. returned device pointers and interior arithmetic remain 32-bit offsets;
2. `cuMemGetAddressRange_v2` on several interior addresses returns the same pool
   base and full size;
3. `cuMemsetD8_v2` covers exact boundary ranges;
4. HtoD, DtoD, and DtoH operations preserve bytes at sizes 8, 256, 1024, and
   5120, covering captured endpoints and the fixed DtoD size;
5. operations issued to `stream = NULL` become visible in issue order after
   `cuCtxSynchronize`;
6. copies do not affect guard ranges outside the requested interval.

The guard checks are implementation safety invariants. Native behavior for an
invalid address is not asserted.

### Modules, globals, textures, and launches

Committed synthetic modules cover handle/state behavior. The local corpus pass
additionally requires:

- all 17 fatbins register through Dark CUDART slot 1;
- all 70 functions, 7 globals, and 33 texture references resolve by exact name;
- `cuModuleGetGlobal_v2(..., bytes=NULL)` succeeds for observed globals;
- texture bind, unbind, format, flags, point filtering, zero mip fields, zero
  anisotropy, and dimension-0 clamp setters preserve the observed state;
- `cuTexRefSetAddress_v2(..., byte_offset=NULL)` is accepted;
- all 16 packed parameter layouts preserve their exact bytes and padding;
- `kernel_parameters=NULL`, packed `extra`, `stream=NULL`, Y/Z dimensions 1,
  block X values 64/128/192/256, and dynamic shared sizes 0/260/4096/7072 are
  accepted;
- the implementation consumes or copies the packed bytes before caller reuse.

The existing `pxvk_parse_launch_extra` tests remain the focused check for token
order and malformed host memory. End-to-end launch tests additionally prove the
parsed bytes reach shader-visible parameters.

## PTX frontend and static-lowering checks

### Whole-corpus local gate

For each of the 17 cataloged PTX hashes:

1. parse the module without diagnostics;
2. reproduce its manifest of PTX version, target, address size, entry names,
   globals, texture references, parameter layouts, and shared/local sizes;
3. select and lower each runtime-resolved function;
4. compile all 16 launched entries to valid SPIR-V for the selected Vulkan
   target;
5. run `spirv-val` for every emitted module;
6. reject an unsupported instruction or directive with the module hash, entry,
   source location, and exact form.

The project-owned oracle completes this sequence through direct SPIR-V emission
and exact Intel/RTX execution. The captured `KernelFluidCalcHashPacked`,
`KernelReorderPosVelAndFindCellStartPacked`, `KernelFluidDensity`, and
`KernelFluidForce` entries compile, validate, and execute as a one-particle
pipeline. PTX-derived Vulkan matches calc/reorder state, both density outputs,
and the force vector on Intel and RTX. Density and force use trace-observed
texture lengths, constant-copy lengths, and block sizes with synthetic exact
constant values.

`PxgCudaFluidCollisionConstraintsBatch` is the eighth captured entry to compile
and execute. Its 725 instructions emit a deterministic 59,442-word module at
local size 128 with 1,840 bytes of static shared memory. A focused one-particle
no-contact record produces exact position, velocity, unit-weight, and twelve
cleared-output results with intact guards on native RTX and translated
Intel/RTX.

`PxgCudaFluidCollisionDynamicResponseBatch` is the ninth captured entry to
compile and execute. Its 263 instructions emit a deterministic 14,671-word
module at local size 128 with 304 bytes of static shared memory. A focused
active-constraint record consumes one constraints-layout normal, accumulated
position, count, body velocity, particle velocity, and ID. Native RTX and
translated Intel/RTX match the cleared accumulators, averaged position,
resolved velocity, response vector, unit weight, copied ID, cleared flags, and
all guards exactly.

`PxgCudaFluidCollisionTrianglesBatch` is the tenth captured entry to compile and
execute. Its 1,022 instructions emit a deterministic 68,441-word module at
local size 64 with 4,912 bytes of static shared memory. The selected module
requires core `shaderInt64`; 64-bit mask arithmetic lowers natively while
`clz.b64` uses validated 32-bit high/low halves. A focused one-particle
zero-triangle work unit preserves position, old position, and state words with
intact guards on native RTX and translated Intel/RTX. It covers the empty-mesh
branch, not a non-empty triangle contact.

`PxgCudaFluidCollisionStaticResponseBatch` is the eleventh captured entry to
compile and execute. Its 330 instructions emit a deterministic 22,632-word
module at local size 128 with 304 bytes of static shared memory. A focused
active-contact record exercises contact normalization, position averaging,
restitution response, output compaction, and particle/source pairing. Native
RTX and translated Intel/RTX match all output bits and guards. Alternate
contact flags, tangential damping, distance clamping, and plane projection
remain uncovered.

`PxgCudaFluidCollisionBoxShapesBatch` is the twelfth captured entry to compile
and execute. Its 1,288 instructions emit a deterministic 61,420-word module at
local size 64 with 6,752 bytes of static shared memory. A trace-shaped
two-work-unit, one-particle, zero-box fixture crosses five barriers and verifies
shape-start, parameter, and particle loads plus the empty-shape exit on native
RTX and translated Intel/RTX. No box-contact geometry is covered.

`PxgCudaFluidCollisionCapsuleShapesBatch` is the thirteenth captured entry to
compile and execute. Its 1,312 instructions emit a deterministic 103,800-word
module at local size 64 with 2,816 bytes of static shared memory. A
trace-shaped two-work-unit, one-particle, zero-capsule fixture crosses five
barriers and verifies shape-start, parameter, and particle loads plus the
empty-shape exit on native RTX and translated Intel/RTX. No capsule-contact
geometry is covered.

`PxgCudaFluidCollisionConvexShapesBatch` is the fourteenth captured entry to
compile and execute. Its 1,433 instructions emit a deterministic 75,510-word
module at local size 64 with 4,225 bytes of static shared memory. A minimum
trace-observed five-workgroup, two-work-unit, one-particle, zero-convex fixture
crosses three barriers on its in-range path and verifies work-unit, shape-start,
parameter, and particle loads plus both out-of-range and empty-shape exits on
native RTX and translated Intel/RTX. No convex-contact
geometry is covered.

`KernelFluidSimpleParticle` is the fifteenth captured entry to compile and
execute. Its 171 instructions emit a deterministic 6,817-word module at local
size 256 with 10,756 bytes of static shared memory. A focused one-particle
fixture stages one 160-byte fluid-parameter record, two start indices, and one
fluid size through shared memory across one barrier, runs the start-index
binary search, and integrates velocity with damping and a skipped clamp.
Native RTX and translated Intel/RTX match the integrated velocity vector and
all guards exactly.

`PxgCudaDeformableUpdate`, `KernelDeformableUberCollision`, and
`KernelDeformableUberSolver<512>` are the fifth through seventh captured entries
to compile and execute. Liveness-based reuse of non-interfering PTX register
slots produces deterministic 9,792-, 175,407-, and 80,354-word SPIR-V modules;
all pass Vulkan 1.2 validation. Native RTX and translated Intel/RTX match exact
output bits and guard ranges for all three focused fixtures. Collision uses the
captured local size 64 and 4,097-byte static shared layout; the solver uses
local size 64 and the captured 7,072-byte dynamic shared allocation.
The last untranslated launched entry, `radixSortBlocks<32,0,false>`, remains
parse/type/CFG validated rather than lowered.

The same three modules also execute as one ordered step. Native CUDA queues
update, collision, and solver on the default stream without intermediate host
synchronization. Vulkan dispatches the same shaders against one heap with
compute-write to compute-read/write barriers between them. Both Intel and RTX
match the native update motion, collision state, solver delta, and 64-byte
record exactly.

The raw PTX remains local. A committed normalized manifest should contain only
hashes and non-proprietary structural expectations already documented in
`mirrors-edge-ptx-corpus.md`.

### Committed synthetic PTX cases

Synthetic cases should defend observable lowering boundaries rather than one
shallow test per spelling:

| Area | Required observations |
|---|---|
| Parameters and control flow | byte-array/scalar alignment, predicates, predicated branches, weak helper calls, return paths |
| Integer arithmetic | 16/32/64-bit wrapping, signed/unsigned compare, `mul.wide.s16`, `mad.lo`, `clz.b64`, shifts at relevant boundaries |
| Addressing and memory | `u8/u16/u32/u64/f32`, scalar/v2/v4, 32-bit offset arithmetic, global/constant/local/shared spaces, volatile shared `u32` |
| Launch built-ins | `%tid.x`, `%ntid.x`, `%ctaid.x`, `%nctaid.x` at observed block sizes and multi-block grids |
| Shared memory | static alignment, dynamic base alignment, and static-plus-dynamic sizes used by radix, reorder, and deformable solver entries |
| Local memory | aligned routine symbols and 64 independent per-invocation store/load results |
| Floating point | every field in the committed CUDA oracle, including fused versus separate operations and NaN operand behavior |
| Barrier | all 64 workgroup invocations observe neighbor writes after `bar.sync 0` |
| Texture | signed coordinates, in-range four-component bit preservation, and negative/high zero results for both `u32` and `f32` |

`tools/ptx-frontend/cfg_probe.ptx` covers packed parameter loads, launch
built-ins, predicated exit, a back-edge loop, global address calculation, and
storage through the conservative loop/switch CFG lowering.
`barrier_cfg_probe.ptx` covers divergent shared-memory setup, predicate
conjunction, structured reconvergence, and one dynamically uniform workgroup
barrier. The project oracle additionally covers a 16-byte-aligned `.local`
symbol with independent state for all 64 invocations. Selected-entry texture
fixtures cover one-component `u32`, four-component bit-preserving `f32`, and
signed OOB zero on Intel and RTX. CTest emits both CFG manifests, the direct
oracle, and three selected oracle SPIR-V modules; SDK validation and device
execution remain explicit compiler verification commands.

Parser-only cases may use dead code to cover the 27 whole-corpus variants absent
from launched entries. They do not become execution requirements unless a
selected launched function reaches them.

## Standalone Vulkan differential

The existing CUDA oracle is the first backend acceptance test. Run the same PTX
intent through the candidate translator and compare normalized JSON field by
field.

This differential now passes on the reference system. Every committed math,
barrier, local-memory, module-constant, and integer/float-texture field matches
native CUDA bit-for-bit on Intel and RTX. The only observed mismatch was Intel
splitting GLSL.std.450 `Fma`; adding `NoContraction` to the FMA result made both
devices return native `0xa8800000`. AMD execution remains a separately tracked
deferred gate.

### Exact fields

These remain bit-exact unless new native evidence changes the contract:

- division results captured by the oracle;
- every FTZ zero result;
- fused FMA and separate multiply/add distinction;
- RTZ and RTE conversion results;
- min/max NaN result for the captured operand order;
- all barrier values;
- all integer texture values and OOB zeros.

### Approximate fields

`rcp.approx`, `sqrt.approx`, and `rsqrt.approx` start as exact comparisons to the
RTX fixture. If Vulkan differs:

1. rerun native and Vulkan at least three times;
2. record both bit patterns and ULP distance;
3. check whether a helper can reproduce native bits;
4. if exact reproduction is disproportionate, define a separate tolerance for
   that operation and input class;
5. prove the tolerance does not hide FTZ, NaN, infinity, or sign errors.

There is no shared blanket epsilon and no comparison of formatted decimal text.

### Device matrix

Run standalone Vulkan differentials on every available target device:

- RTX 3070 Vulkan isolates CUDA-versus-Vulkan semantic differences on the same
  physical architecture;
- Intel UHD Vulkan proves the implementation does not depend on NVIDIA Vulkan
  behavior;
- AMD Radeon 8060S Vulkan is the first external cross-vendor target. Its PE32
  hardware gate passes; the standalone SPIR-V/oracle run is explicitly deferred
  until that machine is available again.

Each result records the physical-device name, vendor/device IDs, Vulkan API,
driver version, enabled features, shader hash, and pxvk commit.

## Real PhysX kernel fixtures

### Capture method

Extend the native tracer only when the backend can run standalone microkernels.
For one selected launch, capture immediately before and after native execution:

- raw fatbin/PTX hash and exact entry name;
- grid, block, dynamic shared size, and packed parameter bytes;
- the 32 MiB device-pool base normalized to offset zero;
- the complete pre-launch device-pool bytes;
- post-launch changed intervals and their exact bytes;
- module global/constant bytes reachable by that entry;
- active texture-reference metadata and bound ranges;
- native driver/GPU identity and all artifact hashes.

The interceptor synchronizes before reading pre-state and after each replayed
launch. Those waits are capture instrumentation, not runtime/performance
requirements.

Capturing the full pre-launch pool avoids guessing which pointer-shaped fields
are inputs. Storing only normalized changed intervals for expected output avoids
including unrelated post-launch bytes in the comparison. Guard intervals around
all changes must remain unchanged in replay.

At the intercepted launch, restore the frozen pool and module state and execute
the same native launch three times in the same context, recording each
post-state. Require identical changed intervals, then restore one accepted
post-state before allowing the game to continue. This holds device addresses
and texture handles constant. If in-process restoration is unsafe, use a
focused native host launcher and explicitly record any address relocation; do
not compensate with a broad output tolerance.

Game/runtime payloads are proprietary and stay under `tmp/evidence/` with a
manifest and SHA-256 values. Do not commit raw heaps, fatbins, or PTX.

### First two fixtures

Use low-complexity launched entries to isolate integration before validating a
whole simulation step:

1. **Fluid:** `KernelFluidCalcHashPacked` — 52-byte packed parameters, block X
   64, grid X 1–5, no dynamic shared memory.
2. **Deformable:** `PxgCudaDeformableUpdate` — 12-byte packed parameters, block
   X 64, grid X 28, no dynamic shared memory in the captured subway route.

Both initial fixtures pass. The deformable case uses one synthetic instance
with exact acceleration, time-step, position, motion, and state-flag bits.
Three native RTX captures are byte-identical; translated Intel and RTX match
all outputs and preserve every surrounding guard word.

The next deformable fixture covers `KernelDeformableUberSolver<512>` with one
synthetic particle and zero constraint iterations. This isolates the captured
7,072-byte dynamic-shared layout, five static barrier sites, 16-bit register
semantics, module-constant addressing, and the final 64-byte solver record
without claiming a complete constraint solve. Native RTX and PTX-derived
Intel/RTX match the exact record, particle delta, and guards.

The third focused fixture covers `KernelDeformableUberCollision` with one
particle and no shapes. It exercises particle initialization, subword memory
accesses, five workgroup barriers, module constants, and response-phase control
without inventing shape data not present in the trace evidence. Native RTX and
PTX-derived Intel/RTX match the copied particle vectors, cleared collision
state, exact scalar words, and 17 guarded heap ranges.

The ordered fixture then connects those focused states instead of replaying
three isolated heaps. Collision reads update's position/motion ranges, and
solver reads and updates collision's position output. Native CUDA uses one
default-stream submission chain and one final synchronization; Vulkan uses one
command buffer with two explicit inter-dispatch barriers. All 25 live ranges
remain guarded, and Intel/RTX match native output bit-for-bit. This proves the
dispatch and shared-heap contract for the trace-shaped path, not collision
geometry or iterative constraint response.

The focused and ordered synthetic fixtures do not prove captured in-game
deformable state or the complete fluid pipeline. Those remain separate
acceptance gates.

### Comparison policy

Replay begins from the exact normalized pre-launch pool and module state.
Compare:

- every expected changed byte;
- every guard interval;
- completion/error state;
- no unexpected changed interval.

Bitwise output is required initially. A mismatch inside a float-bearing range
must be traced to a named PTX operation before any structured field or ULP
comparison is introduced.

The ordered three-kernel deformable step now passes and validates default-stream
ordering, Vulkan dispatch barriers, static/dynamic shared launch shapes, and
inter-kernel state flow. A captured-state deformable replay and the complete
nine-kernel fluid spine remain required before claiming full workload-family
correctness.

## Mirror's Edge integration smoke

### Fixed environment

Every smoke manifest records:

- `MirrorsEdge.exe` SHA-256
  `58de3df21f40e9953e00be92f7749097832bad02e5b657d8485cdb61f360a3dd`;
- system PhysX component hashes from `mirrors-edge-native-trace.md`;
- disabled private `PhysXDevice.dll`/`PhysXLocal` state;
- `PhysXEnhanced=True` and `bDisablePhysXHardwareSupport=False`;
- pxvk binary/commit and Vulkan device/driver;
- renderer wrapper presence, display mode, and resolution;
- save/checkpoint and exact user actions;
- process exit status and backend diagnostic log hash.

The reference display setup is 1280x720 native windowed mode. A graphics wrapper
is operational tooling and must be recorded, never silently treated as a pxvk
dependency.

### Route A: tutorial fluid

1. Start from the captured tutorial save.
2. Continue through the swing using the same ordinary play route.
3. Require CUDA initialization, 17 module registrations, and at least one
   complete nine-kernel fluid spine.
4. Require clean synchronization and normal exit.

Exact launch totals are not a pass criterion because route duration and frame
rate vary. Presence and ordering of the nine-kernel spine are.

### Route B: subway deformable

After an ordinary level has loaded, use:

```ini
F3 = start Subway_p?LoadCheckpoint=Subway_Station?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE
```

Require all three captured deformable entries:

- `PxgCudaDeformableUpdate`;
- `KernelDeformableUberCollision`;
- `KernelDeformableUberSolver<512>` with 7,072 bytes dynamic shared memory.

The native run contained 76 steps, but exact count is diagnostic rather than a
portable pass threshold. No cloth object was visually identified, so this route
must not claim visual cloth equivalence.

### Route C: server-room glass

After an ordinary level has loaded, use:

```ini
F4 = start Scraper_p?LoadCheckpoint=Server_room?Game=TdGame.TdSpStoryGame?AllowCPSaving=FALSE
```

Trigger both player and police gunfire against the known glass. Require:

- visibly intact glass before the trigger and visibly broken glass afterward;
- `KernelFluidSimpleParticle` and the captured collision families to launch;
- no non-success active CUDA result, device loss, hang, or crash;
- normal level operation after destruction.

This is the only current visual effect assertion. It proves scene integration,
not attribution of every particle launch to glass; the enhanced-PhysX-off
control remains a separate attribution probe.

### Smoke failure classification

| Symptom | First evidence to inspect |
|---|---|
| Hardware PhysX warning or no CUDA init | system/private PhysX selection, NVAPI gate, export presence |
| Init succeeds, no modules | Dark API table IDs/slots and x86 ABI |
| Modules resolve, first launch fails | packed parameters, PTX selection, Vulkan startup gate |
| Hang during a scene | default-stream ordering, workgroup barrier, fence completion |
| Scene runs but effect is absent/wrong | real-kernel/sequence fixture before visual tuning |
| Device loss/validation error | Vulkan object lifetime, bounds, synchronization log |
| Native and Vulkan both vary | fixture/route nondeterminism, not expected-output update |

## Artifact and report policy

### Commit

Commit only redistributable inputs and normalized expectations:

- synthetic test sources and PTX;
- exact JSON outputs from project-owned probes;
- structural manifests without proprietary code/data;
- comparison code;
- concise result metadata.

### Keep local and hash

- Mirror's Edge binaries and saves;
- PhysX/CUDART binaries not owned by the project;
- raw fatbins and PTX;
- full device-pool snapshots;
- raw game traces, logs, videos, and screenshots.

Each local fixture directory contains a manifest, per-file SHA-256 list, capture
command, environment, and route. Normalization must be deterministic and its
tool/version recorded.

### Result record

A pass record includes:

```text
pxvk commit
runner and expected-fixture hashes
host process architecture
GPU, vendor/device ID, driver, Vulkan API
shader/module hash
enabled Vulkan features relevant to the test
command/route
comparison result and any named tolerance
raw log/result hash
```

A test is not reproducible if its expected fixture cannot be identified without
a developer's memory.

## Acceptance gates

| Claim | Required evidence |
|---|---|
| CUDA facade initializes | PE32 export/ABI smoke, Dark lifecycle, startup gate, clean teardown |
| PTX grammar/dataflow covers the target | all 17 local modules parse; all label/register/type references resolve; all 16 launched layouts match independent analysis |
| PTX semantic emission covers the target | the project-owned oracle emits matching SPIR-V, then all 16 launched entries emit valid SPIR-V |
| Vulkan semantics are viable on the reference system | committed CUDA-oracle differential passes on RTX and Intel Vulkan, with every mismatch classified |
| Cross-vendor AMD execution is validated | the same generated SPIR-V/oracle output passes on the AMD Radeon 8060S |
| First translated PhysX kernel works | one stable real-kernel fixture passes bitwise with guards |
| Fluid/deformable family works | selected single-kernel fixture plus complete-step sequence fixture passes |
| Mirror's Edge integration works | applicable real-workload fixtures plus all three game routes pass |
| Mirror's Edge target is validated | facade, corpus, standalone, real-kernel, sequence, and game gates pass; remaining exclusions are explicit |

Performance is not an acceptance gate until correctness passes without tracing.
Later timing must use an untraced build and compare simulation stability/frame
impact over a fixed route; trace timings are not evidence.

## First validation implementation slice

The smallest useful slice after research synthesis is:

1. reuse the CUDA oracle inputs as a backend-independent expected schema;
2. create a Vulkan runner for one integer/global-memory microkernel;
3. add f32, barrier, aligned local-memory, and integer/float-texture cases;
4. run on RTX and Intel Vulkan; preserve the identical command/output for the
   deferred AMD run;
5. classify differences before capturing proprietary real-kernel fixtures.

This slice proves the compiler/runtime test loop while keeping game data,
NVAPI, queried-only exports, and full facade integration out of the critical
path.

## Deferred validation evidence
- AMD Radeon 8060S memory-smoke and standalone-oracle execution are deferred
  until the machine is available again. This blocks AMD execution claims, not
  continued runner development or reference-system acceptance.

- Float `tex.1d.v4.f32.s32` native component/OOB behavior is required before
  accepting its lowering.
- Real fluid and deformable input/output captures are required before accepting
  the first corresponding translated kernel.
- Queried-only NULL-versus-stub behavior is required before shrinking the
  conservative symbol surface.
- A minimal non-NVIDIA NVAPI interface set is required before game integration
  on a non-NVIDIA renderer.
- Host thread IDs are required before using observed single-thread behavior to
  relax locking; thread-safe state remains the default.
- The enhanced-PhysX-off server-room control is required only for attributing
  individual particle launches, not for the broken-glass smoke assertion.
