#!/usr/bin/env python3
"""Deterministic benchmark/comparison driver for GLUT PhysX samples.

Runs a sample under the pxvk harness (tools/harness): virtual 20 ms frame
time, scripted keyboard input, framebuffer captures and frame-time logging.
The same run can execute on native NVIDIA CUDA, on pxvk, or on native CUDA
with the launch-capture tracer (tools/nvcuda-trace).

  bench.py run NAME --mode native|pxvk|capture [--frames N] [--keys 300:2,...]
               [--capture 100,299] [--exe SampleCloth.exe]
  bench.py sweep NAME --mode native|pxvk          all 9 SampleCloth scenes
  bench.py compare DIR_A DIR_B [frame ...]         pixel differences

Common options: --stage DIR (sample directory), --build DIR (pxvk build with
nvcuda.dll and the harness), --device NAME (pxvk Vulkan device), --timeout S.
Output goes to test-output/bench/NAME. Native runs force OpenGL onto the
NVIDIA GPU (as loading pxvk's Vulkan device does) so images are comparable.
"""
import argparse
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCENE_FRAMES = 300
SCENE_KEYS = ",".join(f"{SCENE_FRAMES * i}:{i + 1}" for i in range(1, 9))
SCENE_CAPTURES = ",".join(str(SCENE_FRAMES * (i + 1) - 1) for i in range(9))


def run(args, name, frames, keys="", captures="", extra_env=None):
    out = ROOT / "test-output" / "bench" / name
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*"):
        if old.is_file():
            old.unlink()
    build = Path(args.build)
    env = dict(os.environ)
    env.update({
        "PXVK_HARNESS_OUT": str(out),
        "PXVK_HARNESS_FRAMES": str(frames),
        "PXVK_HARNESS_DT_US": str(args.dt_us),
        "PXVK_HARNESS_KEYS": keys,
        "PXVK_HARNESS_CAPTURE": captures,
    })
    inject = []
    if args.mode == "pxvk":
        env["PXVK_LOG"] = str(out / "pxvk.log")
        if args.device:
            env["PXVK_DEVICE"] = args.device
        inject += ["--inject", str(build / "nvcuda.dll")]
    else:
        env.setdefault("SHIM_MCCOMPAT", "0x800000001")
        if args.mode == "capture":
            env["PXVK_TRACE_DIR"] = str(out / "trace")
            inject += ["--inject", str(Path(args.tracer))]
    if extra_env:
        env.update(extra_env)
    inject += ["--inject", str(build / "pxvk-harness.dll")]
    command = [str(build / "pxvk-run.exe"), "--timeout", str(args.timeout), *inject, "--",
               str(Path(args.stage) / args.exe)]
    result = subprocess.run(command, env=env)
    summary = {}
    summary_path = out / "summary.txt"
    if summary_path.exists():
        for line in summary_path.read_text().splitlines():
            key, _, value = line.partition("=")
            summary[key] = value
    summary["exit"] = result.returncode
    return out, summary


def scene_fps(out):
    rows = [line.split(",") for line in (out / "frames.csv").read_text().splitlines()[1:] if line]
    ms = [float(r[1]) for r in rows]
    fps = []
    for scene in range(9):
        segment = ms[scene * SCENE_FRAMES + SCENE_FRAMES // 2:(scene + 1) * SCENE_FRAMES]
        fps.append(round(1000 * len(segment) / sum(segment)) if segment else 0)
    return fps


def load_bmp(path):
    data = path.read_bytes()
    offset = struct.unpack_from("<I", data, 10)[0]
    width, height = struct.unpack_from("<ii", data, 18)
    return width, height, data[offset:]


def pixel_diff(a, b):
    wa, ha, pa = load_bmp(a)
    wb, hb, pb = load_bmp(b)
    if (wa, ha) != (wb, hb):
        return None
    total = len(pa) // 3
    differing = sum(1 for i in range(0, len(pa), 3) if pa[i:i + 3] != pb[i:i + 3])
    return 100.0 * differing / total


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "sweep"):
        p = sub.add_parser(name)
        p.add_argument("name")
        p.add_argument("--mode", choices=["native", "pxvk", "capture"], required=True)
        p.add_argument("--frames", type=int, default=400)
        p.add_argument("--keys", default="")
        p.add_argument("--capture", default="")
        p.add_argument("--exe", default="SampleCloth.exe")
        p.add_argument("--stage", default=str(ROOT / "tmp" / "pxvk-samplecloth-stage"))
        p.add_argument("--build", default=str(ROOT / "build" / "x86"))
        p.add_argument("--tracer", default=str(ROOT / "build" / "nvcuda-trace-mingw" / "nvcuda.dll"))
        p.add_argument("--device", default="")
        p.add_argument("--dt-us", type=int, default=20000)
        p.add_argument("--timeout", type=int, default=900)
    p = sub.add_parser("compare")
    p.add_argument("a")
    p.add_argument("b")
    p.add_argument("frames", nargs="*", type=int)
    args = parser.parse_args()

    if args.command == "run":
        out, summary = run(args, args.name, args.frames, args.keys, args.capture)
        print(" ".join(f"{k}={v}" for k, v in summary.items()))
    elif args.command == "sweep":
        out, summary = run(args, args.name, SCENE_FRAMES * 9, SCENE_KEYS, SCENE_CAPTURES)
        print(f"exit={summary['exit']} fps " + " ".join(f"s{i + 1}:{f}" for i, f in enumerate(scene_fps(out))))
    else:
        a, b = Path(args.a), Path(args.b)
        frames = args.frames or [int(p.stem.split("_")[1]) for p in sorted(a.glob("frame_*.bmp"))]
        failures = 0
        for frame in frames:
            name = f"frame_{frame:05d}.bmp"
            if not (a / name).exists() or not (b / name).exists():
                print(f"{name}: missing")
                failures += 1
                continue
            diff = pixel_diff(a / name, b / name)
            print(f"{name}: {'size mismatch' if diff is None else f'{diff:.3f}% pixels differ'}")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
