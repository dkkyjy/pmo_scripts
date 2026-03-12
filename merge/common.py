"""Shared infrastructure helpers for merge tools."""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


def parse_date_dir(date_dir: str) -> Tuple[Path, str]:
    """Parse yyyy/mm/dd directory string and return path + yyyymmdd."""
    date_obj = datetime.datetime.strptime(date_dir, "%Y/%m/%d")
    return Path(date_dir), date_obj.strftime("%Y%m%d")


def resolve_input_dir(date_dir: str, output_dir: Path) -> Tuple[Path, str]:
    """Resolve input directory from date string and output directory."""
    date_path, ymd = parse_date_dir(date_dir)
    return output_dir / date_path, ymd


def find_files(dirpath: Path, pattern: str) -> Tuple[int, str, List[Path]]:
    """Find sorted files for pattern with common error handling."""
    if not dirpath.exists() or not dirpath.is_dir():
        return 2, f"Directory not found: {dirpath}", []

    files = sorted(dirpath.glob(pattern))
    if not files:
        return 1, f"No files matching '{pattern}' in {dirpath}", []

    return 0, "", files


def build_traceability_header(
    out_name: str,
    files: Iterable[Path],
    include_time: bool,
) -> str:
    """Build comment header listing output identity and source files."""
    header_lines = [f"# Merged: {out_name}"]
    if include_time:
        header_lines.append(
            f"# Time: {datetime.datetime.now().isoformat(timespec='seconds')}"
        )
    header_lines.append("# Source files:")
    for file_path in files:
        header_lines.append(f"#   - {file_path}")
    return "\n".join(header_lines) + "\n\n"


def write_text_with_header(
    outpath: Path,
    header_text: str,
    chunks: Iterable[str],
    ensure_newline_between_chunks: bool = True,
) -> None:
    """Write one output file with a common comment header and text chunks."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with outpath.open("w", encoding="utf-8") as file_obj:
        file_obj.write(header_text)
        for chunk in chunks:
            file_obj.write(chunk)
            if ensure_newline_between_chunks and not chunk.endswith("\n"):
                file_obj.write("\n")
