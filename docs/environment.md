# Environment variables

This is every environment variable read by pxvk and its tools. Set them in
the shell that starts `pxvk-launch.exe` (PowerShell: `$env:NAME = "value"`,
`cmd.exe`: `set NAME=value`). The launched application inherits them.

Conventions:

- **Switches** are on for any value except empty, `0`, `off`, `false` and
  `no` (case-insensitive). `PXVK_SYNC=1` and `PXVK_SYNC=yes` enable it, and
  `PXVK_SYNC=0` disables it.
- **Folders** named by a runtime or tracer variable are created if missing,
  including parent folders. For `PXVK_LOG` this is the folder containing the
  log file.
- The runtime ignores values longer than 1023 characters.

## Runtime (`nvcuda.dll`): for users and testers

| Variable | Values | Default | Effect |
|---|---|---|---|
| `PXVK_LOG` | file path | no log | Writes the log. The file is **replaced** on each run. The first line names the device and its capabilities and ends with `nvapi=` (see `PXVK_NVAPI`). The log then records context creation, module loads, kernel compiles and first launches, errors, unsupported calls, and a `stats` line about every 5 s. |
| `PXVK_DEVICE` | device-name fragment (case-insensitive, e.g. `radeon`, `nvidia`, `intel`) or index (`0`, `1`, ...) | the first discrete GPU, otherwise device 0 | Selects the Vulkan device. If nothing matches, CUDA initialization fails and PhysX uses the CPU. The log then shows `init failed: ...; available devices: "0: ..." "1: ..."`, listing the names and indices to use. |
| `PXVK_NVAPI` | `system` (case-insensitive) | pxvk's NVAPI | `system` makes PhysX's hardware-PhysX check use the NVIDIA driver's `nvapi.dll` instead of `pxvk-nvapi.dll`. Only useful on NVIDIA machines. Without an NVIDIA driver it leads to CPU PhysX. Other values keep the default. The log's first line shows the result: `nvapi=pxvk`, `nvapi=system`, or `nvapi=missing` (no `pxvk-nvapi.dll` next to `nvcuda.dll`, so PhysX uses the driver's NVAPI or none). |
| `PXVK_CACHE_DIR` | folder path, or `off` (case-insensitive) | `%LOCALAPPDATA%\pxvk` | Where the Vulkan pipeline cache (`pipelines-<driver cache UUID>.bin`) is kept, so later runs skip compilation. `off` disables the cache. |

## Runtime: debugging and profiling

The log output of these needs `PXVK_LOG`.

| Variable | Values | Effect |
|---|---|---|
| `PXVK_API_LOG` | switch | Also logs every memory copy, synchronization, texture binding and launch, tagged with the harness frame number (`f=`) and the calling thread (`t=`). The log grows quickly. |
| `PXVK_PROFILE` | switch | Measures GPU time per kernel with timestamp queries. Every 500 launches (as well as every 5 s) it logs a `stats` line plus a `profile` line per kernel with launch count and total/average time. |
| `PXVK_SYNC` | switch | Waits for the GPU after every operation instead of batching. Very slow. Use it to tell a synchronization bug from a computation bug. |
| `PXVK_DUMP_SPIRV` | folder path | Writes every compiled kernel as `<module hash>_<kernel name hash>_<variant>.spv`. Does not need `PXVK_LOG`. |
| `PXVK_NO_FLOAT_ATOMICS` | switch | Compiles global float atomic adds as a compare-and-swap loop even where the device has native float atomics (`VK_EXT_shader_atomic_float`), for comparing the two. The log's `init` line shows `float_atomics=`. |
| `PXVK_PIPELINE_STATS` | switch | Prints the driver's statistics for each pipeline (register count, binary size, ...) to **stderr**, if the driver supports `VK_KHR_pipeline_executable_properties`. GUI applications have no visible stderr, so use it with `pxvk-replay` or redirect the output. Does not need `PXVK_LOG`. |

## Launch capture (differential testing)

See [`testing.md`](testing.md). pxvk and the native tracer
(`tools/nvcuda-trace`) write captures in the same format.

| Variable | Read by | Values | Effect |
|---|---|---|---|
| `PXVK_CAPTURE_DIR` | pxvk | folder path | Destination for pxvk's captures. Capturing requires both this and `PXVK_CAPTURE_FRAMES`. |
| `PXVK_CAPTURE_FRAMES` | pxvk, tracer | comma-separated frame numbers, e.g. `150,450,750` | Snapshots device memory around every launch in these frames, as numbered by the harness (`PXVK_FRAME`). Only works under the harness. |
| `PXVK_CAPTURE_LAUNCHES` | tracer only | launch ordinals and ranges, e.g. `100-120,500` | Snapshots these launches by ordinal. Use it for applications without the harness, such as games. pxvk itself selects only by frame. |
| `PXVK_TRACE_DIR` | tracer | folder path | Tracer output folder. Defaults to `%TEMP%\pxvk-nvcuda-trace`. `pxvk-trace.exe --output DIR` sets it. |
| `PXVK_TRACE_TIMING` | tracer | switch | Logs native per-kernel GPU times, the counterpart of `PXVK_PROFILE`. |
| `PXVK_API_LOG` | tracer | switch | Per-call log in the trace folder with the same `api f=<frame> t=<thread>` prefix as pxvk's, for sequence comparison. Individual messages are close but not identical (the tracer logs what native CUDA was asked, pxvk what it did). |

## Benchmark harness (`pxvk-harness.dll`)

`tools/bench.py` sets these. Set them by hand only when running
`pxvk-run.exe` directly. The harness is a test tool for the GLUT SDK samples,
not part of what testers use.

| Variable | Values | Effect |
|---|---|---|
| `PXVK_HARNESS_OUT` | folder path | Output folder for `frames.csv`, `summary.txt`, `harness.log`, `watchdog.log` and frame captures. Default: the current folder. Only the last folder level is created. |
| `PXVK_HARNESS_FRAMES` | number | Exits the application after this many frames. `0` or unset runs until closed. |
| `PXVK_HARNESS_DT_US` | microseconds | Virtual time step per frame (bench.py uses `20000`), which makes runs deterministic. `0` or unset uses real time. |
| `PXVK_HARNESS_KEYS` | `frame:keys,...`, e.g. `300:2,600:3` | Presses the given keys at the given frames (bench.py's scene sweeps). |
| `PXVK_HARNESS_CAPTURE` | comma-separated frame numbers | Saves the framebuffer as `frame_NNNNN.bmp` at these frames. |
| `PXVK_HARNESS_VSYNC` | switch | Keeps vsync on. By default the harness turns it off to measure throughput. |
| `PXVK_FRAME` | (set by the harness) | The current frame number, which the harness publishes for capture selection and API-log tagging. Do not set it yourself. |

`bench.py` also sets `SHIM_MCCOMPAT=0x800000001` for native runs. This is a
Windows/NVIDIA Optimus setting, not a pxvk variable. It forces OpenGL onto
the NVIDIA GPU, so native and pxvk images come from the same GPU.
