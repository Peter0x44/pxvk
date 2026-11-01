#!/usr/bin/env python3
"""Frame-time benchmark runs of a Direct3D 9 game, native against pxvk.

Starts the game through pxvk-launch with pxvk-framelog.dll injected, which
records every IDirect3DDevice9::Present (tools/framelog), closes the game
after a fixed time, and reports frame statistics for the stretch between
loading stalls, such as a benchmark fly-through. Needs no administrator
rights, unlike PresentMon.

  game-bench.py run NAME --mode native|pxvk --exe GAME.exe [--seconds S]
                [--device NAME] [-- game arguments...]
  game-bench.py report NAME...

Output goes to test-output/game-bench/NAME.frames (and NAME.pxvk.log for
pxvk runs). Example, Borderlands 2's PhysX benchmark (85.8 s fly-through
from about 12 s; the game returns to its menu at about 98 s):

  game-bench.py run native-1 --mode native --seconds 110 \\
      --exe "C:/Program Files (x86)/Steam/steamapps/common/Borderlands 2/Binaries/Win32/Borderlands2.exe" \\
      -- benchmark
  game-bench.py report native-1 pxvk-1

Turn off the game's frame-rate cap first (Borderlands 2: bSmoothFrameRate
in WillowEngine.ini), or every setup measures the cap.
"""
import argparse
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "test-output" / "game-bench"


def running(image):
    result = subprocess.run(["tasklist", "/FI", f"IMAGENAME eq {image}", "/FO", "CSV", "/NH"],
                            capture_output=True, text=True)
    return image.lower() in result.stdout.lower()


def run(args):
    build = Path(args.build).resolve()
    exe = Path(args.exe).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if not k.startswith("PXVK_")}
    env["PXVK_FRAMELOG_OUT"] = str(OUT / f"{args.name}.frames")
    if args.mode == "pxvk":
        env["PXVK_FRAMELOG_LOAD"] = str(build / "nvcuda.dll")
        env["PXVK_LOG"] = str(OUT / f"{args.name}.pxvk.log")
        if args.device:
            env["PXVK_DEVICE"] = args.device
    if running(exe.name):
        sys.exit(f"{exe.name} is already running")
    start = time.monotonic()
    # pxvk-launch stays until the game exits, so it is not waited for here.
    launcher = subprocess.Popen([str(build / "pxvk-launch.exe"), "--inject-dll", str(build / "pxvk-framelog.dll"),
                                 str(exe), *args.game_args], env=env, cwd=exe.parent)
    time.sleep(5)
    if launcher.poll() not in (None, 0) and not running(exe.name):
        sys.exit(f"pxvk-launch failed with exit code {launcher.returncode}")
    while running(exe.name) and time.monotonic() - start < args.seconds:
        time.sleep(2)
    if running(exe.name):
        subprocess.run(["taskkill", "/F", "/IM", exe.name], capture_output=True)
    launcher.wait(timeout=30)
    print(f"{args.name}: {time.monotonic() - start:.0f} s, {OUT / (args.name + '.frames')}")


def window(times, load_ms):
    """The longest stretch between frames of at least load_ms (loads)."""
    loads = [i for i in range(1, len(times)) if (times[i] - times[i - 1]) / 1000 >= load_ms]
    bounds = [0] + loads + [len(times)]
    first, last = max(zip(bounds, bounds[1:]), key=lambda r: times[r[1] - 1] - times[r[0]])
    return times[first:last], [(round((times[i] - times[0]) / 1e6, 1), round((times[i] - times[i - 1]) / 1000))
                               for i in loads]


def report(args):
    print(f"{'run':20}{'seconds':>8}{'frames':>8}{'avg fps':>9}{'median ms':>10}{'p95 ms':>8}"
          f"{'p99 ms':>8}{'1% low fps':>11}  loads (s, ms)")
    for name in args.names:
        path = Path(name) if Path(name).exists() else OUT / f"{name}.frames"
        times = [int(line) for line in path.read_text().split()]
        frames, loads = window(times, args.load_ms)
        ft = sorted((frames[i] - frames[i - 1]) / 1000 for i in range(1, len(frames)))
        seconds = (frames[-1] - frames[0]) / 1e6
        slowest = ft[int(len(ft) * 0.99):]
        print(f"{path.stem:20}{seconds:8.1f}{len(ft):8d}{len(ft) / seconds:9.1f}{statistics.median(ft):10.2f}"
              f"{ft[int(len(ft) * 0.95)]:8.2f}{ft[int(len(ft) * 0.99)]:8.2f}"
              f"{1000 / statistics.mean(slowest):11.1f}  {loads}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run")
    p.add_argument("name")
    p.add_argument("--mode", choices=["native", "pxvk"], required=True)
    p.add_argument("--exe", required=True)
    p.add_argument("--seconds", type=float, default=110)
    p.add_argument("--device", help="PXVK_DEVICE for pxvk runs")
    p.add_argument("--build", default=str(ROOT / "build" / "x86"))
    p.add_argument("game_args", nargs="*")
    p = sub.add_parser("report")
    p.add_argument("names", nargs="+", help="run names or .frames files")
    p.add_argument("--load-ms", type=float, default=400, help="frames this long count as loads")
    args = parser.parse_args()
    run(args) if args.command == "run" else report(args)


if __name__ == "__main__":
    main()
