# ZLUDA NVIDIA tracing for 32-bit PhysX

## Conclusion

Use the **32-bit** ZLUDA launcher in NVIDIA-trace mode from Mirror's Edge's `Binaries` working directory:

```bat
cd /d "C:\GOG Games\Mirror's Edge\Binaries"
"C:\Tools\ZLUDA-8a7931b\32\zluda.exe" --nvidia-trace -- MirrorsEdge.exe
```

This injects ZLUDA's x86 CUDA and NVAPI trace shims while forwarding calls to the real NVIDIA driver. It does not replace the game's files or run CUDA through ZLUDA's compute implementation.

Research is pinned to ZLUDA commit [`8a7931be87d758835776f54ea2e33d30621c74ce`](https://github.com/vosen/ZLUDA/tree/8a7931be87d758835776f54ea2e33d30621c74ce).

## Required package

Use the Windows asset from [ZLUDA v7-preview.9](https://github.com/vosen/ZLUDA/releases/tag/v7-preview.9):

- Asset: `zluda-windows-8a7931b.zip`
- SHA-256 published by GitHub: `0e537e5bd0b0d22c1cc954bb8e870451645d490d820f65812f05da83b03cac43`
- Required files are under `zluda\32\`: `zluda.exe`, `zluda_redirect.dll`, `trace\nvcuda.dll`, and `trace\nvapi.dll`.

The stable v6 Windows archive was inspected and does not contain the `32` subtree. Do not use it for this target. The current workflow explicitly builds `i686-pc-windows-msvc` and merges it into `zluda\32`; see [`.github/workflows/push_master.yml`, lines 80–94](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/.github/workflows/push_master.yml#L80-L94).

Extract ZLUDA outside the game directory. Do not copy trace DLLs into the game or Windows system directories.

## How the NVIDIA trace path works

1. The x86 launcher parses `--nvidia-trace`, creates the target suspended with Microsoft Detours, injects the trace DLLs plus `zluda_redirect.dll`, copies their paths as payloads, then resumes and waits for the target. See `zluda_inject::main_impl` and `InjectionConfig::copy_payloads_to_process` in [`zluda_inject/src/bin.rs`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_inject/src/bin.rs#L39-L74) and [lines 287–311](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_inject/src/bin.rs#L287-L311).
2. For a 32-bit build, the injection set is only `nvcuda.dll` and `nvapi.dll`. Their real-library environment variables are `ZLUDA_CUDA_LIB` and `ZLUDA_NVAPI_LIB`; see [`zluda_windows/src/lib.rs`, lines 41–58](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_windows/src/lib.rs#L41-L58) and [lines 210–223](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_windows/src/lib.rs#L210-L223).
3. `InjectionConfig::nvidia_trace` points those variables at the native NVIDIA system DLLs, injects the corresponding DLLs from `32\trace`, and sets `ZLUDA_LOG_DIR` to `%TEMP%\zluda`; see [`zluda_inject/src/bin.rs`, lines 183–266](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_inject/src/bin.rs#L183-L266). The source names `C:\Windows\System32`; under the x86 launcher and target, WOW64 filesystem redirection resolves this to the 32-bit driver in `SysWOW64`.
4. `zluda_redirect.dll` detours DLL loading by name and propagates the injection into child processes. Its two operating cases are documented in `DetourDetachGuard`: hook `LoadLibrary*` before CUDA loads, or override CUDA functions if CUDA is already loaded. See [`zluda_redirect/src/lib.rs`, lines 748–845](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_redirect/src/lib.rs#L748-L845).
5. The CUDA trace shim forwards through the real driver and records calls, arguments, nested calls, and return values. It captures modules passed through `cuModuleLoad*`, `cuModuleLoadFatBinary`, and related entry points; see the post-call hooks in [`zluda_trace/src/lib.rs`, lines 1372–1489](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_trace/src/lib.rs#L1372-L1489).
6. `StateTracker` splits loaded code containers into PTX, ELF, and other recognized module files. `DumpWriter` names them `module_####...<kind>`; see [`zluda_trace/src/trace.rs`, lines 70–165](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_trace/src/trace.rs#L70-L165) and [lines 318–379](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_trace/src/trace.rs#L318-L379).

The upstream 32-bit PhysX guide explicitly supports `--nvidia-trace` and lists Mirror's Edge as tested; see [`docs/src/physx32.md`](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/docs/src/physx32.md#L39-L72).

## Local target inventory

The installed target is suitable for an x86 trace:

| File or component | Evidence |
|---|---|
| `MirrorsEdge.exe` | PE32/i386; SHA-256 `58de3df21f40e9953e00be92f7749097832bad02e5b657d8485cdb61f360a3dd` |
| `PhysXDevice.dll` | PE32/i386; contains `nvcuda.dll`, `cuInit`, CUDA device-query entry points, and `cuCtxPushCurrent`; SHA-256 `7daa1ccbbaa71150bdb1e367bf0496de63862ee6f46eb7b2947b788e03d143b9` |
| `PhysXLoader.dll` | PE32/i386; refers to the registered `PhysXCore Path`; SHA-256 `2f8bc8adaabe6deec84cf374b96bbb47823de3d9a870bd8ffa41e4a939023e27` |
| `PhysXCore.bak` | PE32/i386; embedded PDB path identifies `EpicPhysX\Epic\2.8.0_FC7`; SHA-256 `d10126c3cfc9232f34ce94e708dee6462ce493685100e6bbd900c02c2993bae9` |
| NVIDIA reference | GeForce RTX 3070 Laptop GPU, driver 610.88 |
| Native x86 driver | `C:\Windows\SysWOW64\nvcuda.dll` and `nvapi.dll` are present |
| PhysX runtime | Registry selects GPU and points to `C:\Program Files (x86)\NVIDIA Corporation\PhysX\Engine`; registered 2.8.0, 2.8.1, and 2.8.3 engines are present |

The game directory currently has `PhysXCore.bak`, not `PhysXCore.dll`. Do not rename it or copy a core DLL locally for the trace: that would change which PhysX implementation is being observed.

## Capture procedure

### Prepare once

1. Download and verify the pinned preview archive above.
2. Extract the inner `zluda` directory to a fixed path such as `C:\Tools\ZLUDA-8a7931b`.
3. Launch Mirror's Edge normally, enable PhysX support, and exit cleanly. Do not change the PhysX setting during a traced run; upstream documents reinitialization failures when changing it in game.
4. Define one reproducible route before capture: load the same checkpoint, exercise the same visible PhysX scene for a fixed interval, then exit normally. Record the checkpoint, route, and duration with the trace.
5. Note the existing directories under `%TEMP%\zluda`. The trace creates a new run directory and appends `_1`, `_2`, and so on rather than overwriting an existing run.

### Run

Use **Command Prompt**, not a shortcut, and make the game binaries directory the working directory:

```bat
cd /d "C:\GOG Games\Mirror's Edge\Binaries"
"C:\Tools\ZLUDA-8a7931b\32\zluda.exe" --nvidia-trace -- MirrorsEdge.exe
```

Passing the bare `MirrorsEdge.exe` name avoids a launcher edge case: `Arguments::command_line_zero_terminated` concatenates the parsed executable and arguments without adding Windows quoting. See [`zluda_inject/src/args.rs`, lines 20–30](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_inject/src/args.rs#L20-L30).

During the run:

1. Confirm PhysX remains enabled without toggling it.
2. Load the predefined checkpoint.
3. Follow the route for the predefined interval.
4. Exit through the game menu so buffered logs are flushed and the launcher returns the game's exit code.

### Collect and validate

The expected directory is `%TEMP%\zluda\MirrorsEdge.exe`; later runs use `%TEMP%\zluda\MirrorsEdge.exe_1`, and so on. The executable name and suffix behavior comes from `Settings::create_trace_directory` in [`zluda_trace/src/lib.rs`, lines 1348–1368](https://github.com/vosen/ZLUDA/blob/8a7931be87d758835776f54ea2e33d30621c74ce/zluda_trace/src/lib.rs#L1348-L1368).

A useful run must contain:

- `log.txt`, with CUDA calls and return statuses;
- at least one module-load call if the exercised scene reached GPU PhysX;
- extracted `module_*.ptx`, `module_*.elf`, or another recognized module kind when the loaded container includes it.

Before analysis:

1. Copy the entire new run directory without renaming individual files.
2. Record the ZLUDA commit and archive digest, NVIDIA driver version, GPU model, game executable hash, checkpoint/route, duration, and whether PhysX was visibly active.
3. Hash every captured file.
4. Preserve the raw trace as immutable evidence; analyze a copy.

`log.txt` alone proves that interception occurred, but not that the selected scene exercised GPU kernels. PTX/ELF files are emitted only when a module-loading path is reached and recognized. The trace logs pointer-valued API arguments; it is not a snapshot of device memory or the contents of every kernel argument.

## Failure triage

- **No `MirrorsEdge.exe*` output directory:** wrong launcher bitness, injection failed before the first traced call, or the target never loaded CUDA. Confirm the command uses `32\zluda.exe` and the preview contains `32\trace\nvcuda.dll`.
- **Directory and `log.txt`, but only device queries:** interception works; PhysX may be disabled or the route did not reach GPU work.
- **Module-load calls but no module files:** inspect adjacent trace errors. The loader may have supplied an unsupported container or extraction may have failed.
- **Immediate native-driver load failure:** confirm both x86 NVIDIA DLLs exist in `C:\Windows\SysWOW64` and that no manually copied `nvcuda.dll` shadows them in the game directory.
- **Crash after changing PhysX settings:** restore the setting during an ordinary run, exit, and start a fresh traced process without toggling it.
- **Game cannot resolve relative assets:** ensure the command was launched after `cd /d` into the `Binaries` directory.

## Scope boundary

This research establishes how to collect evidence. It does not select a Vulkan architecture, infer required CUDA coverage, or implement tracing, CUDA, PTX, or game compatibility code.

## Next task

Collect one controlled native NVIDIA trace using this procedure and document the observed CUDA behavior in `mirrors-edge-native-trace.md`.
