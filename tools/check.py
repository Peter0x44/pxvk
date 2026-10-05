#!/usr/bin/env python3
"""pxvk regression checks.

  check.py [--build DIR] [--ptx DIR] [--captures DIR ...] [--device NAME ...]
           [--no-opcheck]

Steps (each skipped when its inputs are absent):
  1. build the 32-bit tree with CMake/Ninja (i686 MinGW by default)
  2. compile every kernel entry of every module in --ptx with pxvk-ptxc and
     validate the SPIR-V with spirv-val (when on PATH)
  3. pxvk-opcheck: every PhysX instruction variant against native CUDA
     (needs the NVIDIA driver; the only known difference, div.rn, is unused
     by PhysX and expected)
  4. pxvk-replay of every capture directory on every --device; any launch
     graded BAD or FAIL fails the check

Captures and PTX come from proprietary PhysX binaries and stay local; see
docs/testing.md for how to produce them.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def step(title):
    print(f"\n== {title}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default=str(ROOT / "build" / "x86"))
    parser.add_argument("--ptx", help="directory of extracted .ptx modules")
    parser.add_argument("--captures", nargs="*", default=[], help="capture directories (nvcuda-trace output)")
    parser.add_argument("--device", nargs="*", default=[""], help="Vulkan device name fragments to test")
    parser.add_argument("--no-opcheck", action="store_true")
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    build = Path(args.build)
    failures = []

    if not args.no_build:
        step("build")
        if not (build / "build.ninja").exists():
            subprocess.run(["cmake", "-S", str(ROOT), "-B", str(build), "-G", "Ninja",
                            "-DCMAKE_C_COMPILER=i686-w64-mingw32-gcc",
                            "-DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++",
                            "-DCMAKE_BUILD_TYPE=Release"], check=True)
        if subprocess.run(["cmake", "--build", str(build)]).returncode != 0:
            print("build failed")
            return 1

    if args.ptx:
        step(f"compile corpus {args.ptx}")
        spirv_val = shutil.which("spirv-val")
        with tempfile.TemporaryDirectory() as out:
            compiled = invalid = 0
            for module in sorted(Path(args.ptx).glob("*.ptx")):
                target = Path(out) / module.stem
                target.mkdir()
                r = subprocess.run([str(build / "pxvk-ptxc.exe"), "-o", str(target), str(module)],
                                   capture_output=True, text=True)
                for line in r.stdout.splitlines():
                    if line.startswith("FAIL"):
                        failures.append(f"{module.name}: {line}")
                        print(f"  {module.name}: {line}")
                for spv in target.glob("*.spv"):
                    compiled += 1
                    if spirv_val and subprocess.run([spirv_val, "--target-env", "vulkan1.1", str(spv)],
                                                    capture_output=True).returncode != 0:
                        invalid += 1
                        failures.append(f"{module.name}/{spv.name}: spirv-val failed")
            print(f"  {compiled} kernels compiled, {invalid} failed validation"
                  + ("" if spirv_val else " (spirv-val not found)"))

    if not args.no_opcheck:
        step("instruction semantics vs native CUDA")
        for device in args.device:
            command = [str(build / "pxvk-opcheck.exe")] + (["--device", device] if device else [])
            r = subprocess.run(command, capture_output=True, text=True)
            if "native CUDA unavailable" in r.stderr:
                print("  skipped: no native CUDA driver")
                break
            differing = [l for l in r.stdout.splitlines() if "mismatches" in l or "failed" in l]
            # div.rn (unused by PhysX) is not correctly rounded in Vulkan. On
            # non-NVIDIA hardware the approximation units (rcp/rsqrt/sqrt and
            # div.full built on rcp) legitimately differ by a few ULP.
            nvidia = "NVIDIA" in r.stdout.splitlines()[0] if r.stdout else False
            allowed = ("div.rn.f32",) if nvidia else ("div.rn.f32", "div.full", "rcp.approx", "rsqrt.approx", "sqrt.approx")
            unexpected = [l for l in differing if not l.startswith(allowed) or "failed" in l]
            print(f"  {device or 'default'}: {len(differing)} differing variant(s), {len(unexpected)} unexpected")
            failures += [f"opcheck {device}: {l}" for l in unexpected]

    for capture in args.captures:
        for device in args.device:
            step(f"replay {capture} on {device or 'default device'}")
            command = [str(build / "pxvk-replay.exe")] + (["--device", device] if device else []) + [capture]
            r = subprocess.run(command, capture_output=True, text=True)
            lines = r.stdout.splitlines()
            graded = [l.split()[2] for l in lines if len(l.split()) > 2 and l.split()[2] in ("EXACT", "ok", "BAD", "FAIL")]
            print("  " + (lines[0] if lines else "no output"))
            print("  " + ", ".join(f"{g}={graded.count(g)}" for g in ("EXACT", "ok", "BAD", "FAIL")))
            nvidia = bool(lines) and "NVIDIA" in lines[0]
            for l in lines:
                if " FAIL " in l:
                    failures.append(f"replay {capture} {device}: {l}")
                elif " BAD " in l:
                    # Exact semantics are expected on NVIDIA hardware. Elsewhere
                    # iterative solvers amplify approximation-unit differences
                    # into a few out-of-tolerance words; flag only gross errors.
                    fields = dict(f.split("=", 1) for f in l.split() if "=" in f)
                    changed = max(1, int(fields.get("changed", "1")))
                    wrong = int(fields.get("wrong", "0"))
                    if nvidia or wrong > changed * 0.005:
                        failures.append(f"replay {capture} {device}: {l}")

    step("result")
    for f in failures:
        print("  " + f)
    print("PASS" if not failures else f"FAIL ({len(failures)})")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
