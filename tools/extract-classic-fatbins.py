#!/usr/bin/env python3
"""Extract and deduplicate PTX from captured classic CUDA fatbins."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

FATBIN_MAGIC = 0xBA55ED50
FILE_HEADER_VERSION = 0x101
FILE_KIND_PTX = 1
FLAG_LZ4 = 0x2000
FLAG_ZSTD = 0x8000
MAX_BYTES = 64 * 1024 * 1024
FATBIN_HEADER = struct.Struct("<IHHQ")
FILE_HEADER = struct.Struct("<HH9I3Q")


def decompress_lz4_block(source: bytes, expected_size: int) -> bytes:
    output = bytearray()
    offset = 0
    while offset < len(source):
        token = source[offset]
        offset += 1
        literal_length = token >> 4
        if literal_length == 15:
            extension = 255
            while extension == 255:
                if offset >= len(source):
                    raise ValueError("truncated LZ4 literal length")
                extension = source[offset]
                offset += 1
                literal_length += extension
        literal_end = offset + literal_length
        if literal_end > len(source) or len(output) + literal_length > expected_size:
            raise ValueError("invalid LZ4 literal length")
        output.extend(source[offset:literal_end])
        offset = literal_end
        if offset == len(source):
            break
        if offset + 2 > len(source):
            raise ValueError("truncated LZ4 match offset")
        match_offset = source[offset] | (source[offset + 1] << 8)
        offset += 2
        if match_offset == 0 or match_offset > len(output):
            raise ValueError("invalid LZ4 match offset")
        match_length = token & 0x0F
        if match_length == 15:
            extension = 255
            while extension == 255:
                if offset >= len(source):
                    raise ValueError("truncated LZ4 match length")
                extension = source[offset]
                offset += 1
                match_length += extension
        match_length += 4
        if len(output) + match_length > expected_size:
            raise ValueError("invalid LZ4 match length")
        for _ in range(match_length):
            output.append(output[-match_offset])
    if len(output) != expected_size:
        raise ValueError(
            f"LZ4 output size mismatch: expected {expected_size}, got {len(output)}"
        )
    return bytes(output)


def extract_ptx(path: Path) -> list[bytes]:
    data = path.read_bytes()
    if len(data) < FATBIN_HEADER.size:
        raise ValueError("truncated fatbin header")
    magic, version, header_size, files_size = FATBIN_HEADER.unpack_from(data)
    total_size = header_size + files_size
    if (
        magic != FATBIN_MAGIC
        or version != 1
        or header_size < FATBIN_HEADER.size
        or total_size > len(data)
        or total_size > MAX_BYTES
    ):
        raise ValueError("invalid classic fatbin header")

    modules: list[bytes] = []
    offset = header_size
    while offset + FILE_HEADER.size <= total_size:
        (
            kind,
            file_version,
            file_header_size,
            payload_size,
            _unknown0,
            compressed_size,
            _unknown1,
            _unknown2,
            _sm_version,
            _bit_width,
            _unknown3,
            flags,
            _unknown5,
            uncompressed_size,
        ) = FILE_HEADER.unpack_from(data, offset)
        stored_size = max(payload_size, compressed_size)
        payload_offset = offset + file_header_size
        payload_end = payload_offset + stored_size
        if (
            file_version != FILE_HEADER_VERSION
            or file_header_size < FILE_HEADER.size
            or payload_end > total_size
        ):
            raise ValueError("invalid classic fatbin file header")
        if kind == FILE_KIND_PTX:
            if flags & FLAG_ZSTD:
                raise ValueError("Zstd-compressed PTX is not supported")
            if flags & FLAG_LZ4:
                if uncompressed_size > MAX_BYTES or compressed_size > stored_size:
                    raise ValueError("invalid compressed PTX size")
                ptx = decompress_lz4_block(
                    data[payload_offset : payload_offset + compressed_size],
                    uncompressed_size,
                )
            else:
                ptx = data[payload_offset : payload_offset + payload_size].rstrip(b"\0")
            if not ptx:
                raise ValueError("empty PTX payload")
            modules.append(ptx)
        offset = payload_end
    if offset != total_size:
        raise ValueError("trailing partial fatbin file header")
    return modules


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract unique PTX modules from nvcuda-trace classic fatbin captures."
    )
    parser.add_argument("--input", required=True, type=Path, help="capture directory")
    parser.add_argument("--output", required=True, type=Path, help="PTX output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    inputs = sorted(args.input.resolve().glob("fatbin_*.bin"))
    if not inputs:
        raise SystemExit(f"no fatbin_*.bin files found under {args.input}")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory is not empty: {output}")
    modules_by_hash: dict[str, bytes] = {}
    captures_by_hash: dict[str, list[str]] = {}
    for path in inputs:
        try:
            modules = extract_ptx(path)
        except ValueError as error:
            raise SystemExit(f"{path}: {error}") from error
        for ptx in modules:
            digest = hashlib.sha256(ptx).hexdigest()
            modules_by_hash.setdefault(digest, ptx)
            captures_by_hash.setdefault(digest, []).append(path.name)

    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, (digest, ptx) in enumerate(modules_by_hash.items(), start=1):
        name = f"fatbin_{index:04d}.ptx"
        (output / name).write_bytes(ptx)
        records.append(
            {
                "file": name,
                "sha256": digest,
                "bytes": len(ptx),
                "captures": captures_by_hash[digest],
            }
        )

    manifest = {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "capture_count": len(inputs),
        "module_count": len(records),
        "modules": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Extracted {len(records)} unique PTX modules from {len(inputs)} captures")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
