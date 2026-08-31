#!/usr/bin/env python3
"""Assemble the generated CUDA and Vulkan oracle reports into one artifact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

VULKAN_REPORTS = {
    "memory_smoke": "memory-smoke.json",
    "semantic_oracle_static": "oracle.json",
    "semantic_oracle_ptx": "generated-oracle.json",
    "constant_memory": "constant-kernel.json",
    "texture_u32": "texture-kernel.json",
    "texture_f32": "texture-float-kernel.json",
    "fluid_calc_hash": "fluid-calc-hash.json",
    "fluid_reorder": "fluid-reorder.json",
    "fluid_density": "fluid-density.json",
    "fluid_force": "fluid-force.json",
}


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise SystemExit(f"JSON input not found: {path}") from error
    except json.JSONDecodeError as error:
        raise SystemExit(f"invalid JSON in {path}: {error}") from error


def load_vulkan_reports(results_dir: Path) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    missing: list[str] = []
    for key, name in VULKAN_REPORTS.items():
        path = results_dir / name
        if not path.is_file():
            missing.append(name)
            continue
        reports[key] = load_json(path)
    if missing:
        raise SystemExit(
            "Vulkan result directory is incomplete; missing: " + ", ".join(missing)
        )
    return reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build one deterministic reference JSON from generated reports."
    )
    parser.add_argument(
        "--vulkan-results", required=True, type=Path, help="run-tests.ps1 results"
    )
    parser.add_argument(
        "--cuda-oracle", type=Path, help="generated tools/cuda-oracle result"
    )
    parser.add_argument(
        "--cuda-physx", type=Path, help="generated cuda-physx-fixture result"
    )
    parser.add_argument(
        "--base",
        type=Path,
        help="existing bundle whose native CUDA reports and profile are retained",
    )
    parser.add_argument(
        "--profile",
        default="reference-workstation",
        help="stable environment/profile identifier for a new bundle",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = load_json(args.base.resolve()) if args.base else None
    if base is not None and base.get("schema_version") != 1:
        raise SystemExit("base reference bundle has an unsupported schema_version")

    cuda_oracle = (
        load_json(args.cuda_oracle.resolve())
        if args.cuda_oracle
        else base.get("cuda", {}).get("semantic_oracle") if base else None
    )
    cuda_physx = (
        load_json(args.cuda_physx.resolve())
        if args.cuda_physx
        else base.get("cuda", {}).get("physx_fluid_pipeline") if base else None
    )
    if cuda_oracle is None or cuda_physx is None:
        raise SystemExit(
            "a new bundle requires --cuda-oracle and --cuda-physx; "
            "otherwise supply --base"
        )

    profile = base.get("profile", args.profile) if base else args.profile
    bundle = {
        "schema_version": 1,
        "profile": profile,
        "cuda": {
            "semantic_oracle": cuda_oracle,
            "physx_fluid_pipeline": cuda_physx,
        },
        "vulkan": load_vulkan_reports(args.vulkan_results.resolve()),
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        output_file.write(json.dumps(bundle, indent=2) + "\n")
    print(f"Wrote reference bundle: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
