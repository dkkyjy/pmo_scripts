#!/usr/bin/env python3
"""Merge Trigger*<TYPE>.yaml files in a date directory into single files.

This script by default merges three types found in the specified directory:
- matched -> files matching `Trigger*matched.yaml` -> `Trigger_yyyymmdd_matched.yaml`
- PWM     -> files matching `Trigger*PWM.yaml`     -> `Trigger_yyyymmdd_PWM.yaml`
- SWM     -> files matching `Trigger*SWM.yaml`     -> `Trigger_yyyymmdd_SWM.yaml`

Usage examples:
  python merge.py 2025/10/28
  python merge.py /path/to/2025/10/28 -o /tmp/output_dir

If `-o/--output` is omitted, output files are created inside the input directory.
If `-o` is provided it will be treated as an output directory (because multiple output files are produced).
"""

from __future__ import annotations

import argparse
import datetime
import sys
from pathlib import Path
from typing import Optional, Tuple


TYPE_PATTERNS = {
    "matched": "Trigger*matched.yaml",
    "PWM": "Trigger*PWM.yaml",
    "SWM": "Trigger*SWM.yaml",
}



def merge_files_for_pattern(dirpath: Path, pattern: str, outdir: Path, outpath: Path) -> Tuple[int, str]:
    if not dirpath.exists() or not dirpath.is_dir():
        return 2, f"Directory not found: {dirpath}"
    # glob in dirpath (ensure deterministic ordering)
    files = sorted(dirpath.glob(pattern))
    if not files:
        return 1, f"No files matching '{pattern}' in {dirpath}"

    outpath.parent.mkdir(parents=True, exist_ok=True)
    # Write a small comment header listing the source files and timestamp for traceability
    import datetime as _dt

    header_lines = [
        f"# Merged: {outpath.name}",
        "# Source files:",
    ]
    for f in files:
        header_lines.append(f"#   - {str(f)}")
    header_text = "\n".join(header_lines) + "\n\n"

    with outpath.open("w", encoding="utf-8") as wf:
        wf.write(header_text)
        for f in files:
            with f.open("r", encoding="utf-8") as rf:
                content = rf.read()
                wf.write(content)
                if not content.endswith("\n"):
                    wf.write("\n")

    return 0, f"Wrote {len(files)} files -> {outpath}"


def merge_all_types(dirpath: Path, ymd: str, outdir: Path) -> int:
    any_error = 0

    for t, pattern in TYPE_PATTERNS.items():
        outpath = outdir / f"Trigger_{ymd}_{t}.yaml"
        code, msg = merge_files_for_pattern(dirpath, pattern, outdir, outpath)
        if code != 0:
            print(msg, file=sys.stderr)
            any_error = max(any_error, code)
        else:
            print(msg)

    return any_error


def main(argv: Optional[list[str]] = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description="Merge Trigger*<TYPE>.yaml files in a date directory")
    parser.add_argument("dir", help="Directory path like yyyy/mm/dd or full path")
    parser.add_argument(
        "-o",
        "--output",
        default='../Reco_Dir',
        help=(
            "Output directory. Because multiple type-files are produced, "
            "this should be a directory. If omitted, outputs are created in the input directory."
        ),
    )

    args = parser.parse_args(argv)
    outdir = Path(args.output)
    dirp = Path(args.output) / Path(args.dir)
    date = datetime.datetime.strptime(args.dir, "%Y/%m/%d")
    ymd = date.strftime("%Y%m%d")

    return merge_all_types(dirp, ymd, outdir)


if __name__ == "__main__":
    raise SystemExit(main())
