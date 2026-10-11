# Agent guidance

- Read `docs/charter.md` and `docs/architecture.md` before significant
  changes; `docs/development.md` covers building and tools, and
  `docs/roadmap.md` lists open work. The README is for players: keep it
  non-technical.
- Target GPU PhysX, not general CUDA. The charter orders the work in
  stages; the current one is the rest of 32-bit PhysX 2.x (every 2.7/2.8
  runtime), with Mirror's Edge's native comparison still open from the
  first. Do not silently move to a later stage.
- Prefer evidence from traces, captures and real PhysX/PTX workloads over
  speculative generality. Cite concrete source files and functions.
- Keep tasks small and independently testable.
- Before committing compiler or runtime changes, run the regression check
  (`docs/testing.md`) on every available GPU; NVIDIA results must stay
  bit-exact against native.
- PhysX PTX, captures, samples and game files are proprietary: keep them in
  `tmp/` or `test-output/` (git-ignored), never in the repository.
- Keep the docs current when behaviour changes.
