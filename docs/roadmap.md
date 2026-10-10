# Roadmap

Open work, roughly in priority order. Known numerical and performance
limitations of the current design are listed at the end of
`docs/architecture.md`.

## 1. AMD bit-level validation

Done on the Radeon 8060S (AMD proprietary driver 26.8.1, no NVIDIA driver),
using `pxvk-nvapi.dll` (`src/runtime/nvapi.cpp`) for PhysX's NVAPI check.
Without it PhysX called `cuInit` and then chose CPU PhysX.

- `tools/check.py`: all 70 kernels compile and pass `spirv-val`.
- SampleCloth (9 scenes) and SampleParticleFluid run with no errors, and two
  runs are pixel-identical. The cloth solver uses storage-backed shared
  memory (32 KiB device limit). Frame rates are in the README.
- Mirror's Edge (GOG) selects hardware PhysX and runs the fluid kernels at the
  game's 62 fps cap with no errors.

Remaining: copy the native capture sets to the AMD machine and replay them
(`docs/testing.md`, "Testing on another machine"). Compared with CPU PhysX,
SampleCloth scenes 6 (tearing) and 9 (remap mesh) end differently on pxvk:
part of the rug stays on the bar in scene 6, and the sphere ends crumpled in
scene 9. GPU and CPU cloth differ in general, so this needs a native NVIDIA
comparison before it counts as a bug.

## 2. NVAPI coverage

`pxvk-nvapi.dll` implements the eight NVAPI interfaces that ZLUDA implements.
The native Mirror's Edge trace also calls `NvAPI_EnumLogicalGPUs`,
`NvAPI_EnumPhysicalGPUs`, `NvAPI_Stereo_IsEnabled` and two private ids
(`docs/mirrors-edge.md`). These return "no implementation", which has been
enough for the samples and Mirror's Edge. Add them only if another game needs
them.

## 3. Mirror's Edge validation

pxvk runs the game without errors, but nothing has been compared with native
yet. Capture launches from the glass and subway scenes with
`tools/nvcuda-trace`, replay them, and compare the scenes side by side.
Confirm which `Escape_p` checkpoint starts at the RPA glass lobby.

## 4. Easier installation

`pxvk-launch` must start the game, which store launchers (GOG Galaxy, Steam,
EA) do not do. Options: a proxy DLL placed in the game folder that loads pxvk
before CUDART runs, or patching the game's PhysX runtime to load pxvk's
`nvcuda.dll`. The PhysX runtime file renames are documented in
`docs/install.md`. They could also be automated.

## 5. Exactness on NVIDIA

Six of 90 replayed SampleCloth launches differ from native by ≤ 4 ULP,
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
