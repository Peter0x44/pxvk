# Research index

Architecture work begins only after this sequence is complete:

| Order | Document | Required result |
|---:|---|---|
| 0 | [`zluda-tracing.md`](zluda-tracing.md) | **Complete.** Pinned ZLUDA's x86 NVIDIA injection path and the exact Mirror's Edge capture procedure. |
| 1 | [`mirrors-edge-native-trace.md`](mirrors-edge-native-trace.md) | **Fluid, deformable, and visibly broken-glass workloads captured; control pending.** Preserve and identify controlled native traces; document routes, environments, CUDA initialization, chronological API behavior, module loads, launches, transfers, errors, and shutdown. |
| 2 | [`mirrors-edge-ptx-corpus.md`](mirrors-edge-ptx-corpus.md) | **Complete for captured fluid, deformable, and glass workloads.** Seventeen unique PTX modules, 70 entries, 16 launched kernels, packed parameter layouts, and the exact launched instruction/memory/texture/barrier surface are cataloged. |
| 3 | [`cuda-driver-contract.md`](cuda-driver-contract.md) | **Complete for captured workloads.** The 196 queried exports, 35 executed exports, six Dark API tables, x86 ABI, lifecycle, native device responses, memory/copy ordering, texture binding, packed launches, NVAPI gate, and remaining probes are documented. |
| 4 | [`zluda-runtime.md`](zluda-runtime.md) | For each observed CUDA API requirement, cite ZLUDA's concrete implementation path, state model, 32/64-bit boundary, behavior, limitations, and reusable ideas. Ignore unobserved CUDA functionality. |
| 5 | [`zluda-ptx.md`](zluda-ptx.md) | For each construct in the captured corpus, cite ZLUDA's parser, lowering, and execution paths; record assumptions and unsupported cases. Ignore PTX absent from the corpus. |
| 6 | [`native-oracle-probes.md`](native-oracle-probes.md) | Resolve only questions left open by the trace and source study with minimal NVIDIA experiments. Record each question, probe, environment, result, and resulting contract decision. |
| 7 | [`vulkan-requirements.md`](vulkan-requirements.md) | Map the settled CUDA/PTX contract to required Vulkan capabilities and limits. Separate hard workload requirements from candidate implementation techniques; do not select a wrapper or detailed architecture. |
| 8 | [`validation-plan.md`](validation-plan.md) | Define observable correctness checks against the NVIDIA oracle: focused API/PTX tests, captured corpus checks, and a reproducible Mirror's Edge smoke route. |
| 9 | [`research-synthesis.md`](research-synthesis.md) | Consolidate the minimum implementation scope, explicit exclusions, evidence, remaining blockers, risks, and the recommended first independently testable implementation slice. This is the architecture gate. |

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

A disposable experiment may start after the PTX and driver contracts expose a
coherent thin path. Language, dependency, and permanent architecture choices
remain deferred until the synthesis gate.
