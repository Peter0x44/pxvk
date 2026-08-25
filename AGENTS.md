# Agent guidance

- Read `docs/charter.md` before making significant changes.
- Target classic 32-bit GPU PhysX, not general CUDA. Mirror's Edge is the first compatibility target.
- Prefer evidence from traces and real PhysX/PTX workloads over speculative generality.
- Do not implement features solely because CUDA supports them.
- Keep tasks small and independently testable.
- Cite concrete source files and functions in research findings where applicable.
- Do not silently broaden project scope.
- Do not make major architectural decisions before the research phase is complete.
