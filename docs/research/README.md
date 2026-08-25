# Research index

Architecture work begins only after this sequence is complete:

| Order | Document | Required result |
|---:|---|---|
| 0 | [`zluda-tracing.md`](zluda-tracing.md) | **Complete.** Pinned ZLUDA's x86 NVIDIA injection path and the exact Mirror's Edge capture procedure. |
| 1 | [`mirrors-edge-native-trace.md`](mirrors-edge-native-trace.md) | Preserve and identify a controlled native trace; document the route, environment, CUDA initialization, chronological API behavior, module loads, launches, transfers, errors, and shutdown. |
| 2 | [`mirrors-edge-ptx-corpus.md`](mirrors-edge-ptx-corpus.md) | Deduplicate every captured module and map modules, functions, and launches. Inventory PTX versions, instructions, address spaces, barriers, warp operations, textures, shared memory, parameters, and launch shapes. |
| 3 | [`cuda-driver-contract.md`](cuda-driver-contract.md) | Turn the observed API surface into a finite compatibility contract: x86 ABI, state transitions, ordering, memory behavior, device attributes, legacy launch setup, required outputs, and observed error behavior. |
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
