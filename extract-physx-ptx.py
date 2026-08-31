#!/usr/bin/env python3
"""Extract PTX records from classic NVIDIA CUDA fatbins embedded in a binary."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

FATBIN_MAGIC = b"\x50\xed\x55\xba"
FATBIN_VERSION = 1
FILE_HEADER_VERSION = 0x101
FILE_KIND_PTX = 1
FLAG_COMPRESSED_LZ4 = 0x2000
FLAG_COMPRESSED_ZSTD = 0x8000
ENTRY_RE = re.compile(rb"\.entry\s+([A-Za-z0-9_$]+)")


@dataclass(frozen=True)
class Module:
    index: int
    fatbin_offset: str
    file: str
    sha256: str
    bytes: int
    sm_version: int
    bit_width: int
    entrypoints: list[str]


def decompress_lz4_block(source: bytes, expected_size: int) -> bytes:
    """Decode the raw LZ4 block form used by the classic CUDA fatbins."""
    output = bytearray(expected_size)
    source_offset = 0
    output_offset = 0

    while source_offset < len(source):
        token = source[source_offset]
        source_offset += 1

        literal_length = token >> 4
        if literal_length == 15:
            while True:
                if source_offset >= len(source):
                    raise ValueError("truncated LZ4 literal length")
                extension = source[source_offset]
                source_offset += 1
                literal_length += extension
                if extension != 255:
                    break

        literal_end = source_offset + literal_length
        output_end = output_offset + literal_length
        if literal_end > len(source) or output_end > len(output):
            raise ValueError("invalid LZ4 literal range")
        output[output_offset:output_end] = source[source_offset:literal_end]
        source_offset = literal_end
        output_offset = output_end

        if source_offset == len(source):
            break
        if source_offset + 2 > len(source):
            raise ValueError("truncated LZ4 match offset")

        match_offset = struct.unpack_from("<H", source, source_offset)[0]
        source_offset += 2
        if match_offset == 0 or match_offset > output_offset:
            raise ValueError(f"invalid LZ4 match offset {match_offset}")

        match_length = token & 0xF
        if match_length == 15:
            while True:
                if source_offset >= len(source):
                    raise ValueError("truncated LZ4 match length")
                extension = source[source_offset]
                source_offset += 1
                match_length += extension
                if extension != 255:
                    break
        match_length += 4
        if output_offset + match_length > len(output):
            raise ValueError("invalid LZ4 match range")

        match_source = output_offset - match_offset
        for index in range(match_length):
            output[output_offset + index] = output[match_source + index]
        output_offset += match_length

    if output_offset != expected_size:
        raise ValueError(
            f"LZ4 output was {output_offset} bytes; expected {expected_size}"
        )
    return bytes(output)


def extract_modules(binary: bytes) -> tuple[list[Module], dict[str, bytes]]:
    modules: list[Module] = []
    payloads: dict[str, bytes] = {}
    fatbin_index = 0
    search_offset = 0

    while (offset := binary.find(FATBIN_MAGIC, search_offset)) >= 0:
        search_offset = offset + 1
        if offset + 16 > len(binary):
            break

        _, version, header_size, files_size = struct.unpack_from("<IHHQ", binary, offset)
        fatbin_end = offset + header_size + files_size
        if (
            version != FATBIN_VERSION
            or header_size < 16
            or fatbin_end > len(binary)
        ):
            continue

        fatbin_index += 1
        file_offset = offset + header_size
        while file_offset + 64 <= fatbin_end:
            (
                kind,
                file_version,
                file_header_size,
                payload_size,
                _unknown0,
                compressed_size,
                _unknown1,
                _unknown2,
                sm_version,
                bit_width,
                _unknown3,
                flags,
                _unknown5,
                uncompressed_size,
            ) = struct.unpack_from("<HHIIIIIIIIIQQQ", binary, file_offset)
            stored_size = max(payload_size, compressed_size)
            record_end = file_offset + file_header_size + stored_size
            if (
                file_version != FILE_HEADER_VERSION
                or file_header_size < 64
                or record_end > fatbin_end
            ):
                raise ValueError(f"malformed fatbin record at 0x{file_offset:08x}")

            if kind == FILE_KIND_PTX:
                payload_offset = file_offset + file_header_size
                if flags & FLAG_COMPRESSED_ZSTD:
                    raise ValueError(
                        "Zstd-compressed PTX is outside the classic PhysX "
                        f"extractor scope at 0x{file_offset:08x}"
                    )
                if flags & FLAG_COMPRESSED_LZ4:
                    compressed = binary[
                        payload_offset : payload_offset + compressed_size
                    ]
                    ptx = decompress_lz4_block(compressed, uncompressed_size)
                else:
                    ptx = binary[payload_offset : payload_offset + payload_size].rstrip(
                        b"\0"
                    )

                name = f"fatbin_{fatbin_index:04d}.ptx"
                if name in payloads:
                    raise ValueError(f"multiple PTX records in fatbin {fatbin_index}")
                payloads[name] = ptx
                modules.append(
                    Module(
                        index=fatbin_index,
                        fatbin_offset=f"0x{offset:08x}",
                        file=name,
                        sha256=hashlib.sha256(ptx).hexdigest(),
                        bytes=len(ptx),
                        sm_version=sm_version,
                        bit_width=bit_width,
                        entrypoints=[
                            match.group(1).decode("ascii")
                            for match in ENTRY_RE.finditer(ptx)
                        ],
                    )
                )

            file_offset = record_end

        search_offset = fatbin_end

    return modules, payloads


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract classic CUDA PTX fatbin records from a PE executable or DLL."
        )
    )
    parser.add_argument("binary", type=Path, help="input executable or DLL")
    parser.add_argument("output_dir", type=Path, help="local output directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary_path = args.binary.resolve()
    output_dir = args.output_dir.resolve()
    if not binary_path.is_file():
        raise SystemExit(f"input binary not found: {binary_path}")

    binary = binary_path.read_bytes()
    try:
        modules, payloads = extract_modules(binary)
    except ValueError as error:
        raise SystemExit(f"invalid embedded fatbin: {error}") from error
    if not modules:
        raise SystemExit(f"no embedded CUDA fatbin PTX records found in {binary_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_module in output_dir.glob("fatbin_*.ptx"):
        if old_module.name not in payloads:
            old_module.unlink()
    for name, ptx in payloads.items():
        (output_dir / name).write_bytes(ptx)

    manifest = {
        "schema_version": 1,
        "source_file": binary_path.name,
        "source_sha256": hashlib.sha256(binary).hexdigest(),
        "source_bytes": len(binary),
        "modules": [asdict(module) for module in modules],
    }
    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8", newline="\n") as manifest_file:
        manifest_file.write(json.dumps(manifest, indent=2) + "\n")

    print(f"Extracted {len(modules)} PTX modules from {binary_path}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
