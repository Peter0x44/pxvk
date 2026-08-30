# Research index

Architecture work begins only after this sequence is complete:

| Order | Document | Required result |
|---:|---|---|
| 0 | [`zluda-tracing.md`](zluda-tracing.md) | **Complete.** Pinned ZLUDA's x86 NVIDIA injection path and the exact Mirror's Edge capture procedure. |
| 1 | [`mirrors-edge-native-trace.md`](mirrors-edge-native-trace.md) | **Fluid, deformable, and visibly broken-glass workloads captured; control pending.** Preserve and identify controlled native traces; document routes, environments, CUDA initialization, chronological API behavior, module loads, launches, transfers, errors, and shutdown. |
| 2 | [`mirrors-edge-ptx-corpus.md`](mirrors-edge-ptx-corpus.md) | **Complete for captured fluid, deformable, and glass workloads.** Seventeen unique PTX modules, 70 entries, 16 launched kernels, packed parameter layouts, and the exact launched instruction/memory/texture/barrier surface are cataloged. |
| 3 | [`cuda-driver-contract.md`](cuda-driver-contract.md) | **Complete for captured workloads.** The 196 queried exports, 35 executed exports, six Dark API tables, x86 ABI, lifecycle, native device responses, memory/copy ordering, texture binding, packed launches, NVAPI gate, and remaining probes are documented. |
| 4 | [`zluda-runtime.md`](zluda-runtime.md) | **Complete for the observed runtime surface.** ZLUDA's x86 facade, shared-memory/event server boundary, public/Dark APIs, virtual pointers and handles, module metadata, packed launches, texture simplifications, NVAPI gate, reusable ideas, and deliberately excluded HIP/server machinery are documented. |
| 5 | [`zluda-ptx.md`](zluda-ptx.md) | **Complete for the captured corpus.** Parser coverage, normalization order, dead-code removal, 32-bit offset conversion, module metadata, arithmetic/FP behavior, memory spaces, predication, barriers, textures, backend helpers, reusable frontend ideas, and HIP/LLVM-specific exclusions are documented. |
| 6 | [`native-oracle-probes.md`](native-oracle-probes.md) | **Core PTX semantic probes complete; workload-output/runtime probes remain.** A MinGW x86 tool and committed RTX 3070 fixture define native FTZ, FMA, approximate math, conversion, `bar.sync`, and legacy linear-texture bounds behavior. |
| 7 | [`vulkan-requirements.md`](vulkan-requirements.md) | **Complete for the captured corpus and observed Intel, NVIDIA, and AMD x86 devices.** Vulkan 1.2, integer/float-control features, quantitative limits, storage/workgroup mappings, dispatch barriers, software-FTZ risk, texture bounds, and the non-requirement for subgroup size 32 are documented without selecting a permanent wrapper or runtime architecture. |
| 8 | [`validation-plan.md`](validation-plan.md) | **Complete for the captured target.** Layered x86 facade, CUDA state, PTX corpus, standalone Vulkan differential, real-kernel replay, fluid/deformable sequence, and three-route game checks now have explicit artifacts, pass criteria, comparison policy, and acceptance gates. |
| 9 | [`research-synthesis.md`](research-synthesis.md) | **Complete; standalone Vulkan oracle and project-owned PTX parser passed, opening the gate to whole-corpus frontend coverage.** The minimum C++17/Volk in-process design, finite PTX-to-SPIR-V path, memory/command model, cross-vendor device selection, exclusions, blockers, risks, and subsequent independently testable slices are fixed. |

## Evidence rules

- Pin and hash raw traces, binaries, modules, tools, and probe outputs.
- Cite trace records, probe results, or concrete source files and functions for every compatibility requirement.
- Put only observed workload behavior in the required scope.
- Record uncertainty explicitly; use `native-oracle-probes.md` rather than guessing.
- Do not choose architecture, dependencies, or general CUDA coverage before `research-synthesis.md`.

## Implementation start gate

The native traces now establish the workload families. Before writing the
permanent pxvk runtime, complete enough research to answer:

- **Module contract:** which fatbins/PTX modules and entry points are unique,
  which kernels actually launch, and which globals and texture references each
  uses.
- **PTX contract:** the exact directives, instructions, types, address spaces,
  barriers, warp operations, texture operations, and shared-memory behavior in
  the launched corpus.
- **Launch ABI:** how each captured packed buffer maps to the corresponding PTX
  `.param` declarations, including alignment and pointer fields.
- **Driver ABI:** the finite exported and Dark API surface, x86 calling
  conventions, handle lifetimes, state transitions, queried device values,
  synchronization, and error results.
- **Backend facts:** which Vulkan features and limits can implement that
  contract while preserving CUDA-visible warp size 32.
- **Oracle:** at least one deterministic correctness signal for fluid and
  deformable execution, not only visual success. Prefer focused native probes
  or selected output-buffer hashes over full replay capture.
- **Scope gate:** `research-synthesis.md` names the first independently testable
  slice and confirms that unresolved items do not block it.

The synthesis gate now permits the standalone cross-vendor Vulkan oracle
runner. The permanent game runtime remains behind standalone semantic results,
deterministic real-kernel fixtures, and the later CUDA-facade/NVAPI gates.

## Deferred probe register

Open probes are tracked rather than dropped:

| Probe | Status | Required before |
|---|---|---|
| Deterministic fluid/deformable kernel outputs | Deferred to `validation-plan.md`; standalone PTX semantics already have an oracle | Accepting the first real translated PhysX kernel |
| Float legacy-texture fetch bits/OOB | Deferred until a candidate texture lowering exists | Accepting texture lowering |
| Queried-only CUDA symbol NULL versus stub behavior | Non-blocking; ZLUDA proves non-null not-supported stubs work | Removing conservative export stubs |
| Minimal non-NVIDIA NVAPI interface set | Deferred to game-integration validation | Running the original PhysX runtime without NVIDIA |
| CUDA host-call thread IDs | Deferred runtime-hardening evidence | Claiming observed single-thread assumptions or optimizing locking |
| PhysX-off server-room control | Deferred attribution check | Claiming individual particle launches represent only glass |
| Vulkan differential against committed CUDA oracle | Complete on Intel/RTX: every f32, conversion, barrier, and integer-texture field matches bit-for-bit | Accepting reference-system standalone PTX semantic lowering |
| AMD generated-SPIR-V execution | Hardware preflight passes; full oracle deferred until the AMD machine is available | Claiming cross-vendor execution correctness on AMD |

Each probe blocks only the named acceptance point. None blocks the next local
slice: finite PTX frontend development on the Intel/RTX workstation.
