#!/usr/bin/env python3
"""Compare launch captures of the same frames from native CUDA and pxvk.

Native and pxvk place allocations at different device addresses, so regions
are matched (allocations of 1 MiB or more by size, in address order among
equal sizes; module globals by name) and 32-bit words that hold device
pointers, in memory and in the launch parameters, are translated before
comparing.

usage: capture-diff.py <native-capture-dir> <pxvk-capture-dir> [--limit N]
"""
import struct
import sys
from pathlib import Path


def read_meta(path):
    meta = {"globals": {}, "allocations": [], "textures": {}}
    for line in path.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if key == "global":
            meta["globals"][parts[1]] = (int(parts[2], 16), int(parts[3]))
        elif key == "allocation":
            meta["allocations"].append((int(parts[1], 16), int(parts[2])))
        elif key == "texref":
            meta["textures"][parts[1]] = (int(parts[2], 16), int(parts[3]))
        elif key in ("kernel", "frame", "launch", "params", "shared"):
            meta[key] = parts[1] if len(parts) > 1 else ""
    return meta


def read_image(path):
    data = path.read_bytes()
    memory = {}
    offset = 0
    while offset + 8 <= len(data):
        address, size = struct.unpack_from("<II", data, offset)
        memory[address] = data[offset + 8: offset + 8 + size]
        offset += 8 + size
    return memory


def flatten(memory, base, size):
    out = bytearray(size)
    for address, chunk in memory.items():
        start = max(address, base)
        end = min(address + len(chunk), base + size)
        if start < end:
            out[start - base:end - base] = chunk[start - address:end - address]
    return bytes(out)


def build_mapping(native, pxvk):
    """Pairs of (native_base, pxvk_base, size) regions."""
    pairs = []

    def big(meta):
        # pxvk lists module globals among its allocations; leave them to the
        # by-name matching below.
        globals_ = {base for base, _ in meta["globals"].values()}
        by_size = {}
        for base, size in sorted(meta["allocations"]):
            if size >= 1 << 20 and base not in globals_:
                by_size.setdefault(size, []).append(base)
        return by_size

    big_native, big_pxvk = big(native), big(pxvk)
    for size in sorted(set(big_native) | set(big_pxvk)):
        n, p = big_native.get(size, []), big_pxvk.get(size, [])
        if len(n) != len(p):
            print(f"    warning: {len(n)} native and {len(p)} pxvk allocations of {size} bytes; "
                  f"{abs(len(n) - len(p))} left unmatched")
        pairs += [(nb, pb, size) for nb, pb in zip(n, p)]
    for name, (base, size) in native["globals"].items():
        if name in pxvk["globals"]:
            pairs.append((base, pxvk["globals"][name][0], size))
    return pairs


def translate(word, pairs):
    for n, p, size in pairs:
        if n <= word < n + size:
            return p + (word - n)
    return None


def params_match(native_params, pxvk_params, pairs):
    """Launch parameters (hex) are equal once device pointers are translated."""
    a, b = bytes.fromhex(native_params), bytes.fromhex(pxvk_params)
    if len(a) != len(b):
        return False
    for i in range(0, len(a) - 3, 4):
        wa, wb = struct.unpack_from("<I", a, i)[0], struct.unpack_from("<I", b, i)[0]
        if wa != wb and translate(wa, pairs) != wb:
            return False
    return a[len(a) & ~3:] == b[len(b) & ~3:]


def compare(native_image, pxvk_image, pairs, limit):
    diffs = []
    total = 0
    for n_base, p_base, size in pairs:
        a = flatten(native_image, n_base, size)
        b = flatten(pxvk_image, p_base, size)
        if a == b:
            continue
        for i in range(0, size - 3, 4):
            wa = struct.unpack_from("<I", a, i)[0]
            wb = struct.unpack_from("<I", b, i)[0]
            if wa == wb:
                continue
            if translate(wa, pairs) == wb:
                continue
            total += 1
            if len(diffs) < limit:
                diffs.append((n_base, i, wa, wb))
    return total, diffs


def main():
    native_dir, pxvk_dir = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else 8
    for meta_path in sorted(native_dir.glob("capture_*.txt")):
        other = pxvk_dir / meta_path.name
        if not other.exists():
            continue
        native, pxvk = read_meta(meta_path), read_meta(other)
        pairs = build_mapping(native, pxvk)
        for phase in ("before", "after"):
            n_img = read_image(meta_path.with_suffix("." + phase))
            p_img = read_image(other.with_suffix("." + phase))
            total, diffs = compare(n_img, p_img, pairs, limit)
            print(f"{meta_path.stem} {native.get('kernel', '?')[:40]:40} {phase:6} differing_words={total}")
            for base, offset, wa, wb in diffs:
                fa = struct.unpack("<f", struct.pack("<I", wa))[0]
                fb = struct.unpack("<f", struct.pack("<I", wb))[0]
                print(f"    region {base:08x}+{offset:#x}: native={wa:08x} ({fa:.6g}) pxvk={wb:08x} ({fb:.6g})")
        if not params_match(native.get("params", ""), pxvk.get("params", ""), pairs):
            print(f"    params differ: native={native.get('params')} pxvk={pxvk.get('params')}")


if __name__ == "__main__":
    main()
