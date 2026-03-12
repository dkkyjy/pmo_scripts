#!/usr/bin/env python3
"""Merge Trigger*<TYPE>.yaml files in a date directory into single files.

This script by default merges three types found in the specified directory:
- matched -> files matching `Trigger*matched.yaml` -> `Trigger_yyyymmdd_matched.yaml`
- PWM     -> files matching `Trigger*PWM.yaml`     -> `Trigger_yyyymmdd_PWM.yaml`
- SWM     -> files matching `Trigger*SWM.yaml`     -> `Trigger_yyyymmdd_SWM.yaml`

Usage examples:
  python -m merge.merge_results 2025/10/28
  python -m merge.merge_results 2025/10/28 -o /tmp/output_dir

If `-o/--output` is omitted, output files are created inside the input directory.
If `-o` is provided it will be treated as an output directory (because multiple output files are produced).
"""

from __future__ import annotations

import argparse
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
    "matched": "Trigger*matched.yaml",
    "PWM": "Trigger*PWM.yaml",
    "SWM": "Trigger*SWM.yaml",
}


def merge_files_for_pattern(
    dirpath: Path,
    pattern: str,
    outpath: Path,
) -> Tuple[int, str]:
    """Concatenate files matching one pattern into one output file."""
    code, message, files = find_files(dirpath, pattern)
    if code != 0:
        return code, message

    header_text = build_traceability_header(
        outpath.name,
        files,
        include_time=False,
    )
    chunks = [file_path.read_text(encoding="utf-8") for file_path in files]
    write_text_with_header(outpath, header_text, chunks, ensure_newline_between_chunks=True)

    return 0, f"Wrote {len(files)} files -> {outpath}"


def merge_all_types(dirpath: Path, ymd: str, outdir: Path) -> int:
    """Concatenate all configured result types."""
    any_error = 0

    for merge_type, pattern in TYPE_PATTERNS.items():
        outpath = outdir / f"Trigger_{ymd}_{merge_type}.yaml"
        code, msg = merge_files_for_pattern(dirpath, pattern, outpath)
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
        description="Merge Trigger*<TYPE>.yaml files in a date directory"
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

    args = parser.parse_args(argv)
    outdir = Path(args.output)
    input_dir, ymd = resolve_input_dir(args.dir, outdir)

    return merge_all_types(input_dir, ymd, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
