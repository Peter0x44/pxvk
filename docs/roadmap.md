# Roadmap

Open work, roughly in priority order. Known numerical and performance
limitations of the current design are listed at the end of
`docs/architecture.md`.

## 1. Run without an NVIDIA driver (NVAPI shim)

The PhysX runtime checks for hardware PhysX through NVAPI before touching
CUDA, so on AMD/Intel-only machines games most likely fall back to CPU
PhysX and never reach pxvk. Every test so far ran with the real NVIDIA driver
installed. Needed:

- a minimal 32-bit `nvapi.dll` exposing `nvapi_QueryInterface` for the dozen
  interfaces PhysX calls (listed in `docs/mirrors-edge.md`), reporting one
  GPU, injected by `pxvk-launch` like `nvcuda.dll`;
- a local reproduction first: disabling the RTX 3070 in Device Manager on
  the development laptop leaves the Intel GPU driving the display and should
  show the CPU fallback, then confirm the shim fixes it.

## 2. AMD validation

Never run on AMD. An AMD laptop with a Radeon 8060S is available; an earlier
capability probe on it reported Vulkan 1.4, a 4 GiB storage-buffer range,
native fp32 flush-to-zero, subgroup sizes 32–64 with size control, and 32 KiB
of shared memory (so the cloth solver will use the storage-backed shared
memory variant, as on Intel). Plan: build there, run `tools/check.py` with
the copied PTX corpus and captures (`docs/testing.md`, "Testing on another
machine"), then the SDK samples and Mirror's Edge once item 1 works.

## 3. Mirror's Edge validation

pxvk runs the game without errors, but nothing has been compared with native
yet. Capture launches from the glass and subway scenes with
`tools/nvcuda-trace`, replay them, and compare the scenes side by side.
Confirm which `Escape_p` checkpoint starts at the RPA glass lobby.

## 4. Easier installation

`pxvk-launch` must start the game, which store launchers (GOG Galaxy, Steam,
EA) do not do. Options: a proxy DLL placed in the game folder that loads pxvk
before CUDART runs, or patching the game's PhysX runtime to load pxvk's
`nvcuda.dll`. Also document or automate the PhysX runtime file renames.

## 5. Exactness on NVIDIA

Six of 202 replayed SampleCloth launches differ from native by ≤ 4 ULP,
which makes SampleCloth scene 7 diverge visibly after ~100 frames;
SampleParticleFluid scenes 2, 3 and 5 also drift slightly. Players will not
notice, but exact results keep differential testing simple. Starting points
are in the known limitations of `docs/architecture.md`.

## 6. Performance

Frame rates match or beat native on the RTX 3070. Remaining: the cloth
`UberSolver` kernel is ~19% slower per launch than native in scene 7, and
64-thread launches pay 30–80% for `.ftz` emulation on NVIDIA. NVIDIA's Vulkan
driver does not expose compiled machine code, which limits further analysis.

## Later: Winlator on Qualcomm Adreno

Running Windows games through Winlator (Wine + Box64) on Android would use
the Adreno Vulkan driver (Qualcomm's or Mesa Turnip). Known obstacles:

- Most Android drivers, including Adreno ones, report a 128 MiB
  `maxStorageBufferRange`. pxvk keeps all CUDA memory in one buffer, and
  PhysX cloth alone allocates a 128 MiB pool. The heap would have to be split
  across several buffers selected by address bits (needs non-uniform
  descriptor indexing), or use buffer device addresses (needs 64-bit integer
  support in shaders).
- Adreno subgroups are 64 or 128 lanes; the warp-synchronous guard accepts
  that, but it has not been verified.
- 32 KiB shared memory (storage fallback), and unknown float-controls support
  (FTZ emulation and possible NaN-comparison issues).
- The x86 runtime itself runs under emulation, so first-use compilation is
  slow; the persistent pipeline cache helps afterwards.

A first step that needs no phone: simulate Adreno limits on the development
laptop with the Khronos Vulkan Profiles layer and run the replay tests.

## Later: more games

Alice: Madness Returns and Mafia II Classic. Trace them natively first: new
PTX modules or API usage would show up there.
