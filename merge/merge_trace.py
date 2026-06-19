#!/usr/bin/env python3
"""Merge Trigger*<TYPE>.yaml trace files in a date directory into single files.

This script merges trace channel files found in the specified directory:
- F  -> `Trigger*_F.yaml`  -> `Trigger_yyyymmdd_F_merged.yaml` (or `_RUN{run}_F_merged.yaml`)
- X  -> `Trigger*_X.yaml`  -> `Trigger_yyyymmdd_X_merged.yaml`
- Y  -> `Trigger*_Y.yaml`  -> `Trigger_yyyymmdd_Y_merged.yaml`
- Z  -> `Trigger*_Z.yaml`  -> `Trigger_yyyymmdd_Z_merged.yaml`
- XY -> `Trigger*_XY.yaml` -> `Trigger_yyyymmdd_XY_merged.yaml`

Usage examples:
  python -m merge.merge_trace 2025/10/28
  python -m merge.merge_trace 2025/10/28 -o /tmp/output_dir
  python -m merge.merge_trace 2025/10/28 --channel XY --run-number 10192

If `-o/--output` is omitted, output files are created inside the input directory.
If `-o` is provided it will be treated as an output directory (because multiple output files are produced).
If `--channel` is specified, only that channel is merged instead of all five.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

from merge.common import (
    build_traceability_header,
    find_files,
    resolve_input_dir,
    write_text_with_header,
)


TYPE_PATTERNS = {
    "F": "Trigger*_F.yaml",
    "X": "Trigger*_X.yaml",
    "Y": "Trigger*_Y.yaml",
    "Z": "Trigger*_Z.yaml",
    "XY": "Trigger*_XY.yaml",
}


def build_result_pattern(merge_type: str, run_number: Optional[int]) -> str:
    """Build coarse glob pattern for one result type and optional RUN number."""
    if run_number is None:
        return TYPE_PATTERNS[merge_type]
    return f"Trigger*_RUN{run_number}_*_{merge_type}.yaml"


def _filter_files_by_run_number(files: list[Path], run_number: Optional[int]) -> list[Path]:
    """Filter files by exact RUN segment to avoid RUN10/RUN100 collisions."""
    if run_number is None:
        return files

    run_pattern = re.compile(rf"RUN{run_number}(?!\d)")
    return [path for path in files if run_pattern.search(path.name)]


def merge_files_for_pattern(
    dirpath: Path,
    pattern: str,
    outpath: Path,
    run_number: Optional[int] = None,
) -> Tuple[int, str]:
    """Concatenate files matching one pattern into one output file."""
    code, message, files = find_files(dirpath, pattern)
    if code != 0:
        return code, message

    files = _filter_files_by_run_number(files, run_number)
    if run_number is not None and not files:
        return (
            1,
            f"No files matching RUN{run_number} in {dirpath} after boundary filtering",
        )

    header_text = build_traceability_header(
        outpath.name,
        files,
        include_time=False,
    )
    chunks = [file_path.read_text(encoding="utf-8") for file_path in files]
    write_text_with_header(outpath, header_text, chunks, ensure_newline_between_chunks=True)

    return 0, f"Wrote {len(files)} files -> {outpath} (run_number={run_number})"


def merge_all_types(
    dirpath: Path,
    ymd: str,
    outdir: Path,
    run_number: Optional[int] = None,
    channel: Optional[str] = None,
) -> int:
    """Concatenate all configured result types.

    When channel is provided, only that trace channel is merged.
    """
    any_error = 0

    if run_number is None:
        print(
            "Deprecated usage: --run-number is not provided; "
            "falling back to full-day per-type merge",
            file=sys.stderr,
        )

    merge_types = [channel] if channel else list(TYPE_PATTERNS)
    for merge_type in merge_types:
        if merge_type not in TYPE_PATTERNS:
            print(f"Unknown channel: {merge_type}", file=sys.stderr)
            return 1

        pattern = build_result_pattern(merge_type, run_number)
        if run_number is None:
            outpath = outdir / f"Trigger_{ymd}_{merge_type}_merged.yaml"
        else:
            outpath = outdir / f"Trigger_{ymd}_RUN{run_number}_{merge_type}_merged.yaml"

        code, msg = merge_files_for_pattern(
            dirpath,
            pattern,
            outpath,
            run_number=run_number,
        )
        if code != 0:
            print(msg, file=sys.stderr)
            any_error = max(any_error, code)
        else:
            print(msg)

    return any_error


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Merge Trigger*<TYPE>.yaml trace files in a date directory"
    )
    parser.add_argument("dir", help="Directory path like yyyy/mm/dd or full path")
    parser.add_argument(
        "-o",
        "--output",
        default="../Reco_Dir",
        help=(
            "Output directory. Because multiple type-files are produced, "
            "this should be a directory. If omitted, outputs are created in the input directory."
        ),
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help=(
            "Optional RUN number for file filtering and output naming. "
            "When omitted, script keeps legacy per-type full-day merge behavior."
        ),
    )
    parser.add_argument(
        "--channel",
        choices=["F", "X", "Y", "Z", "XY"],
        default=None,
        help="Trace channel to merge (default: all five channels)",
    )

    args = parser.parse_args(argv)
    outdir = Path(args.output)
    input_dir, ymd = resolve_input_dir(args.dir, outdir)

    return merge_all_types(input_dir, ymd, outdir, run_number=args.run_number, channel=args.channel)


if __name__ == "__main__":
    raise SystemExit(main())
