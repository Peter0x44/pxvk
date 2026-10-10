# Roadmap

Open work in priority order. Known numerical and performance limitations of
the current design are listed at the end of `docs/architecture.md`.

## Correctness

### 1. Rare nondeterminism in SampleCloth scene 2 (not reproduced)

Seen once: one 9-scene pxvk sweep on the RTX 3070 differed from native by 22%
of the pixels at frame 599 (scene 2), while other runs of the same build
matched native exactly. That run came during hours of heavy GPU load that
ended, minutes later, in a hard GPU/display lock of the laptop.

It did not reproduce in about 50 further runs of scenes 1–2 (alone, two
instances at once, and with an empty pipeline cache), nor in 640 replays of
the native scene 2 captures on NVIDIA (all bit-exact) and 240 on Intel (the
same result every time). The sample takes its time step from the harness's
virtual clock and waits for PhysX every frame, PhysX issues CUDA calls from
one thread per scene, and no PhysX kernel uses atomics. Treat it as a
hardware/driver glitch unless it recurs; then compare `PXVK_API_LOG` logs
(which record the calling thread) and pxvk captures of a diverging and a
matching run with `tools/capture-diff.py`.

### 2. Mirror's Edge against native

pxvk runs the game without errors, but nothing has been compared with native.
Capture launches from the glass (server room) and cloth (subway) scenes with
`tools/nvcuda-trace`, replay them, and compare the scenes side by side.
Confirm which `Escape_p` checkpoint starts at the RPA glass lobby
(`docs/mirrors-edge.md`).

### 3. AMD bit-level validation

Done on the Radeon 8060S (AMD driver 26.8.1, no NVIDIA driver), using
`pxvk-nvapi.dll` for PhysX's NVAPI check: all 70 kernels compile and pass
`spirv-val`; SampleCloth and SampleParticleFluid run with no errors and
repeated runs are pixel-identical (the cloth solver uses storage-backed shared
memory, 32 KiB device limit); Mirror's Edge selects hardware PhysX and holds
its 62 fps cap.

Remaining: copy the native capture sets to the AMD machine and replay them
(`docs/testing.md`, "Testing on another machine"). Compared with CPU PhysX,
SampleCloth scenes 6 (tearing) and 9 (remap mesh) end differently on pxvk:
part of the rug stays on the bar in scene 6, and the sphere ends crumpled in
scene 9. GPU and CPU cloth differ in general, so compare with native NVIDIA
before treating it as a bug.

## Usability

### 4. Starting games from store launchers

`pxvk-launch` must start the game, which GOG Galaxy, Steam and EA do not do.
Options: a proxy DLL in the game folder (e.g. `dinput8.dll` or `version.dll`,
forwarding to the system DLL) that loads pxvk before CUDART runs, or patching
the game's PhysX runtime to load pxvk's `nvcuda.dll`. Also automate the
Mirror's Edge PhysX runtime renames documented in `docs/install.md`.

### 5. Tester feedback

A tester with an NVIDIA GPU received an early build (before
`pxvk-nvapi.dll` and CUDA 4.x fatbin support). Send a current build and
`docs/install.md`, and fold the results back in.

## Nice to have

### 6. Exactness on NVIDIA

Six of 90 replayed SampleCloth launches differ from native by ≤ 4 ULP,
which makes SampleCloth scene 7 diverge visibly after ~100 frames;
SampleParticleFluid scenes 2, 3 and 5 also drift slightly. Players will not
notice, but exact results keep differential testing simple. Starting points
are in the known limitations of `docs/architecture.md`.

### 7. Performance

Frame rates match or beat native on the RTX 3070. Remaining: the cloth
`UberSolver` kernel is ~19% slower per launch than native in scene 7, and
64-thread launches pay 30–80% for `.ftz` emulation on NVIDIA. NVIDIA's Vulkan
driver does not expose compiled machine code, which limits further analysis.

### 8. NVAPI coverage

`pxvk-nvapi.dll` implements the eight NVAPI interfaces that ZLUDA implements.
The native Mirror's Edge trace also calls `NvAPI_EnumLogicalGPUs`,
`NvAPI_EnumPhysicalGPUs`, `NvAPI_Stereo_IsEnabled` and two private ids
(`docs/mirrors-edge.md`). These return "no implementation", which has been
enough for the samples, Mirror's Edge and FluidMark. Add them only if another
application needs them.

## Later

### 9. Winlator on Qualcomm Adreno

Running Windows games through Winlator (Wine + Box64) on Android would use
the Adreno Vulkan driver (Qualcomm's or Mesa Turnip). Known obstacles:

- Most Android drivers, including Adreno ones, report a 128 MiB
  `maxStorageBufferRange`. The split heap (`docs/architecture.md`) handles
  this when the driver supports non-uniform storage-buffer indexing (Turnip
  does; Qualcomm's driver is unverified). Tested on lavapipe, which has the
  same limit.
- Adreno subgroups are 64 or 128 lanes, which contain whole warps, so warp
  emulation should not be needed; not verified on hardware.
- 32 KiB shared memory (storage fallback), and unknown float-controls support
  (FTZ emulation and possible NaN-comparison issues).
- The x86 runtime itself runs under emulation, so first-use compilation is
  slow; the persistent pipeline cache helps afterwards.

lavapipe already reproduces the storage-range limit on a PC
(`docs/testing.md`); the Khronos Vulkan Profiles layer could simulate the
remaining Adreno limits before testing on a phone.

### 10. More games

Alice: Madness Returns and Mafia II Classic. Trace them natively first: new
PTX modules or API usage would show up there.
