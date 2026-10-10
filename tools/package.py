#!/usr/bin/env python3
"""Build the tester package: the runtime files plus the tester documentation.

  package.py [--build build/x86] [--out dist] [--name pxvk-<commit>]

Creates OUT/NAME/ and OUT/NAME.zip containing:

  pxvk-launch.exe, nvcuda.dll, pxvk-nvapi.dll   from the build directory
  README.md          docs/install.md (the tester guide)
  environment.md     docs/environment.md
  mirrors-edge.md    docs/mirrors-edge.md
  run-mirrors-edge.cmd  starts the GOG version of Mirror's Edge with a log
  LICENSE-MIT, LICENSE-APACHE

Links between the packaged documents are kept (install.md becomes
README.md); links to documents that are not packaged are reduced to their
text, so the package stands on its own. The build must contain only
Windows system DLL imports (checked here). Used by CI
(.github/workflows/build.yml) and usable locally.
"""
import argparse
import re
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BINARIES = ["pxvk-launch.exe", "nvcuda.dll", "pxvk-nvapi.dll"]
DOCUMENTS = {"install.md": "README.md", "environment.md": "environment.md", "mirrors-edge.md": "mirrors-edge.md"}
# DLLs present on every supported Windows installation.
SYSTEM_DLLS = {"kernel32.dll", "user32.dll", "shell32.dll", "advapi32.dll", "gdi32.dll", "msvcrt.dll",
               "ntdll.dll", "ole32.dll"}
LAUNCHER = (
    "@echo off\r\n"
    "rem Starts Mirror's Edge (GOG) on pxvk with a log next to this file.\r\n"
    "rem Edit GAME if the game is installed elsewhere. See README.md for setup.\r\n"
    "setlocal\r\n"
    "set \"GAME=C:\\GOG Games\\Mirror's Edge\\Binaries\\MirrorsEdge.exe\"\r\n"
    "set \"PXVK_LOG=%~dp0pxvk.log\"\r\n"
    "\"%~dp0pxvk-launch.exe\" \"%GAME%\"\r\n"
)


def imported_dlls(path):
    """Names of the DLLs a PE32 image imports (no external tools needed)."""
    data = path.read_bytes()
    pe = struct.unpack_from("<I", data, 0x3C)[0]
    if data[pe:pe + 4] != b"PE\0\0" or struct.unpack_from("<H", data, pe + 24)[0] != 0x10B:
        raise SystemExit(f"{path.name}: not a 32-bit PE image")
    sections = struct.unpack_from("<H", data, pe + 6)[0]
    optional = pe + 24
    import_rva = struct.unpack_from("<I", data, optional + 104)[0]
    table = optional + struct.unpack_from("<H", data, pe + 20)[0]

    def offset(rva):
        for i in range(sections):
            virtual_size, virtual_address, raw_size, raw_offset = struct.unpack_from("<IIII", data, table + 40 * i + 8)
            if virtual_address <= rva < virtual_address + max(virtual_size, raw_size):
                return raw_offset + rva - virtual_address
        raise SystemExit(f"{path.name}: bad RVA {rva:#x}")

    names = []
    descriptor = offset(import_rva) if import_rva else None
    while descriptor is not None:
        name_rva = struct.unpack_from("<I", data, descriptor + 12)[0]
        if name_rva == 0:
            break
        start = offset(name_rva)
        names.append(data[start:data.index(b"\0", start)].decode("ascii"))
        descriptor += 20
    return names


def package_markdown(text):
    """Keep links between packaged documents; reduce other relative links to their text."""
    def replace(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(("http://", "https://", "#")):
            return match.group(0)
        document = target.split("#", 1)[0]
        if document in DOCUMENTS:
            return f"[{label.replace(document, DOCUMENTS[document])}]({DOCUMENTS[document]}{target[len(document):]})"
        return label
    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", replace, text)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--build", default=str(ROOT / "build" / "x86"))
    parser.add_argument("--out", default=str(ROOT / "dist"))
    parser.add_argument("--name", help="package name (default: pxvk-<short commit>)")
    args = parser.parse_args()

    name = args.name
    if not name:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        name = "pxvk-" + (commit.stdout.strip() if commit.returncode == 0 else "local")
    build, out = Path(args.build), Path(args.out)
    target = out / name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for binary in BINARIES:
        source = build / binary
        if not source.exists():
            raise SystemExit(f"missing {source}; build first")
        foreign = [dll for dll in imported_dlls(source) if dll.lower() not in SYSTEM_DLLS]
        if foreign:
            raise SystemExit(f"{binary} imports non-system DLLs: {', '.join(foreign)} (link statically)")
        shutil.copy2(source, target / binary)
    for document, packaged in DOCUMENTS.items():
        text = (ROOT / "docs" / document).read_text(encoding="utf-8")
        (target / packaged).write_text(package_markdown(text), encoding="utf-8", newline="\n")
    (target / "run-mirrors-edge.cmd").write_bytes(LAUNCHER.encode("ascii"))
    for license_file in ("LICENSE-MIT", "LICENSE-APACHE"):
        shutil.copy2(ROOT / license_file, target / license_file)

    archive = out / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(target.iterdir()):
            zf.write(path, f"{name}/{path.name}")
    print(f"{target}\n{archive}")


if __name__ == "__main__":
    sys.exit(main())
