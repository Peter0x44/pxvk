# Testing pxvk

pxvk is tested differentially against the real NVIDIA driver at three
levels. Everything derived from PhysX binaries (PTX, captures, samples) is
proprietary and stays local under `tmp/` or `test-output/`.

## 1. Instruction semantics: `pxvk-opcheck`

Generates a small PTX kernel for each instruction variant PhysX uses (and the
FMA-contraction patterns ptxas applies), runs it through the 32-bit NVIDIA
driver and through pxvk with random and special-value inputs, and compares
results bit for bit.

```powershell
build\x86\pxvk-opcheck.exe --device nvidia
build\x86\pxvk-opcheck.exe --device intel --filter f32
```

## 2. Real launches: capture and replay

`tools/nvcuda-trace` is a native proxy driver. With `PXVK_CAPTURE_FRAMES` (or
`PXVK_CAPTURE_LAUNCHES`) it snapshots all device memory around selected
launches and records parameters, texture bindings and module globals.
`pxvk-replay` re-executes each capture on Vulkan at the original addresses
and grades the result (EXACT, ok = within 4 ULP / 1e-3 relative, BAD, FAIL).

```powershell
cmake -S tools\nvcuda-trace -B build\nvcuda-trace-mingw -G Ninja -DCMAKE_C_COMPILER=i686-w64-mingw32-gcc
cmake --build build\nvcuda-trace-mingw
$env:PXVK_CAPTURE_FRAMES = "150,450,750"
python tools\bench.py sweep captures --mode capture
build\x86\pxvk-replay.exe --device nvidia test-output\bench\captures\trace\SampleCloth.exe_<pid>
```

Replay options: `--verbose` (show wrong words), `--time N` (GPU timing),
`--storage-shared` (force storage-backed shared memory), `--ptx FILE`
(override the module PTX, e.g. with hand-inserted probes storing to a
`.global pxvk_debug[]` symbol, which replay maps to a scratch area and
prints).

The pxvk runtime writes captures in the same format (`PXVK_CAPTURE_DIR` +
`PXVK_CAPTURE_FRAMES`), and `tools/capture-diff.py` compares native and
pxvk captures of the same frames with pointer translation. That finds the
first launch whose inputs or outputs diverge in a full application run.

`PXVK_API_LOG=1` makes both the tracer and the runtime log every copy,
synchronization, texture binding and launch per frame for sequence diffs.

## 3. Applications: `tools/bench.py`

`pxvk-harness.dll` makes GLUT samples deterministic (virtual 20 ms frames),
scripts key presses, captures frames and logs frame times; native runs are
bit-for-bit reproducible.

```powershell
python tools\bench.py sweep native --mode native
python tools\bench.py sweep pxvk --mode pxvk --device nvidia
python tools\bench.py compare test-output\bench\native test-output\bench\pxvk
```

Native runs force OpenGL onto the NVIDIA GPU (`SHIM_MCCOMPAT`), as loading
pxvk's Vulkan device does on Optimus laptops, so images are comparable.

## Regression check

```powershell
python tools\check.py --ptx tmp\extracted-ptx --captures <capture dirs> --device nvidia intel
```

Builds, compiles and validates every kernel of the PTX corpus, runs opcheck
and replays the captures. On NVIDIA exact semantics are required; on other
vendors approximation-unit differences are tolerated.
