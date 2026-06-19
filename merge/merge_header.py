#!/usr/bin/env python3
"""Merge Trigger*.yaml header files in a date directory into one merged file.

This script merges only header-type YAML files (those without trace suffixes
like _X, _Y, _Z, _XY, _matched, _fingerprint, _PWM, _SWM).

Usage examples:
  python -m merge.merge_header 2025/10/28
  python -m merge.merge_header 2025/10/28 -o /tmp/output_dir
  python -m merge.merge_header 2025/10/28 --run-number 10192

If `-o/--output` is omitted, output files are created inside the input directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

from merge.common import (
    build_traceability_header,
    find_files,
    resolve_input_dir,
    write_text_with_header,
)


PATTERN = "Trigger*.yaml"
TRACE_SUFFIX_PATTERN = re.compile(r"_(F|X|Y|Z|XY|matched|fingerprint|PWM|SWM)\.yaml$")


def build_trigger_pattern(run_number: Optional[int]) -> str:
    """Build coarse glob pattern for Trigger header YAML files.

    When run_number is provided, use a narrower glob and then rely on
    regex filtering for strict numeric boundaries.
    """
    if run_number is None:
        return PATTERN
    return f"Trigger_*_RUN{run_number}_*.yaml"


def _filter_files_by_run_number(files: List[Path], run_number: Optional[int]) -> List[Path]:
    """Filter files by exact RUN segment to avoid RUN10/RUN100 collisions."""
    if run_number is None:
        return files

    run_pattern = re.compile(rf"RUN{run_number}(?!\d)")
    return [path for path in files if run_pattern.search(path.name)]


def _filter_out_trace_suffix_files(files: List[Path]) -> List[Path]:
    """Exclude trace-type suffix files from header merge inputs."""
    return [
        path
        for path in files
        if TRACE_SUFFIX_PATTERN.search(path.name) is None
    ]


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

    files = _filter_out_trace_suffix_files(files)
    if not files:
        return (
            1,
            f"No header-compatible files in {dirpath} after trace-suffix filtering",
        )

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


def merge_trigger_files(
    dirpath: Path,
    ymd: str,
    outdir: Path,
    run_number: Optional[int] = None,
) -> int:
    """Merge Trigger*.yaml header files in one date directory into one output file."""
    if run_number is None:
        print(
            "Deprecated usage: --run-number is not provided; "
            "falling back to full-day Trigger*.yaml merge",
            file=sys.stderr,
        )
        outpath = outdir / f"Trigger_{ymd}_merged.yaml"
    else:
        outpath = outdir / f"Trigger_{ymd}_RUN{run_number}_merged.yaml"

    pattern = build_trigger_pattern(run_number)
    code, msg = merge_files_for_pattern(
        dirpath,
        pattern,
        outpath,
        run_number=run_number,
    )
    if code != 0:
        print(msg, file=sys.stderr)
        return code
    print(msg)
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entry point."""
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Merge Trigger*.yaml header files in a date directory"
    )
    parser.add_argument("dir", help="Directory path like yyyy/mm/dd or full path")
    parser.add_argument(
        "-o",
        "--output",
        default="../Reco_Dir",
        help="Output directory for merged files (default: ../Reco_Dir)",
    )
    parser.add_argument(
        "--run-number",
        type=int,
        default=None,
        help=(
            "Optional RUN number for file filtering and output naming. "
            "When omitted, script keeps legacy full-day merge behavior."
        ),
    )

    args = parser.parse_args(argv)
    outdir = Path(args.output)
    input_dir, ymd = resolve_input_dir(args.dir, outdir)

    return merge_trigger_files(input_dir, ymd, outdir, run_number=args.run_number)


if __name__ == "__main__":
    raise SystemExit(main())
