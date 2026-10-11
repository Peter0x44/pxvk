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

Done for the charter's first stage: a tester ran the game on pxvk and on
native GPU PhysX on an NVIDIA card (October 2026). The effects looked the
same and frame rates matched, about 72 FPS on pxvk against 70 natively in a
GPU-bound path-traced scene with cloth, where most of the cost is rendering.
The game's kernels are the 2.8.0 runtime's, which replay bit-exact.

Still useful later: native launch captures from the glass (server room) and
cloth (subway) scenes with `tools/nvcuda-trace`, for regression replays of
the game itself. Confirm which `Escape_p` checkpoint starts at the RPA glass
lobby (`docs/mirrors-edge.md`).

### 3. AMD bit-level validation

Done on the Radeon 8060S (AMD driver 26.8.1, no NVIDIA driver), using
`pxvk-nvapi.dll` for PhysX's NVAPI check: all 70 kernels compile and pass
`spirv-val`; SampleCloth and SampleParticleFluid run with no errors and
repeated runs are pixel-identical (the cloth solver uses storage-backed shared
memory, 32 KiB device limit); Mirror's Edge selects hardware PhysX and holds
its 62 fps cap.

The native RTX 3070 captures were then replayed on the Radeon (driver
26.9.1): SampleCloth, SampleParticleFluid and FluidMark give exactly the
Intel UHD results, and `check.py` passes. All 7 tolerated BAD launches are
the iterative cloth `UberSolver`, drifting through the approximation units.
Sweeps compared with native frames end every cloth and fluid scene the
same way as native, including cloth scenes 6 and 9, which differ from CPU
PhysX on native NVIDIA too. Details in `docs/testing.md`.

`pxvk-replay --mismatches` and `wrong_int=` then checked every wrong word:
the two integer-looking ones trace, through probes, to a single particle
flag decided by a float threshold test that the approximation units flip
(`docs/testing.md`).

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

Two of 90 replayed SampleCloth launches (`UberSolver`, scene 5) differ from
native by ≤ 4 ULP, and SampleCloth scene 5 ends with under 0.1% of its
pixels different; every other SampleCloth and SampleParticleFluid scene is
pixel-identical to native. The cause is ptxas leaving some products unfused
without a structural reason found so far (`docs/architecture.md`).

### 7. Performance

Frame rates match or beat native on the RTX 3070. Remaining: the cloth
`UberSolver` kernel is ~19% slower per launch than native in scene 7, and
64-thread launches pay 30–80% for `.ftz` emulation on NVIDIA. NVIDIA's Vulkan
driver does not expose compiled machine code, which limits further analysis.

On Intel UHD, PhysX 2.8.4's cloth normal kernel spends up to three quarters
of its time in the shared-memory float atomic add loop (180 µs for a
1024-vertex cloth against a 43 µs floor; `docs/testing.md`). Intel has no
native shared float atomic add. Neighbouring triangles, which sit in the
same subgroup, add to the same vertices, so the loop's retries are likely
the cost; combining equal addresses within a subgroup before the atomic
might help. Worth measuring once a 2.8.4 cloth game shows how many such
launches a frame has.

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
PTX modules or API usage would show up there. `docs/compatibility.md` lists
the 32-bit GPU PhysX games and runtimes; every 2.7.x/2.8.x runtime now
compiles, so what is left is tracing games that use them.

### 11. Smaller launch captures

Every capture stores sparse before and after images of the whole heap: about
23 MB per FluidMark launch (1.9 GB for 82), and replay needs several copies
of the image in memory (it is linked large-address-aware for that). A kernel
changes a small part of the heap, so storing the after image as a diff,
chaining consecutive captures (the next launch's before image is this one's
after image plus host copies), compressing, and comparing only written pages
in replay would cut this to a small fraction.
