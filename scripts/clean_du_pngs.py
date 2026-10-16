#!/usr/bin/env python3
"""删除 Reconstruction 子目录中文件名包含 DU 的 png 图片。

默认 dry-run，只打印统计信息；加 --run 才真正删除。

用法:
    # 预览（推荐先用）
    python scripts/clean_du_pngs.py

    # 指定目录
    python scripts/clean_du_pngs.py --reco-dir ./Reconstruction

    # 真正执行删除
    python scripts/clean_du_pngs.py --run
"""

import argparse
import os
import sys
from datetime import datetime


def human_size(size_bytes):
    """字节数转可读格式。"""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


def _du_png_files(reco_dir):
    """生成器：遍历 reco_dir 的所有子目录，yield 包含 DU 的 png 完整路径。"""
    for entry in os.scandir(reco_dir):
        if not entry.is_dir():
            continue
        # 子目录名格式: Trigger_xxx_event_N
        for fname in os.listdir(entry.path):
            if fname.endswith(".png") and "DU" in fname:
                yield os.path.join(entry.path, fname)


def clean(reco_dir, dry_run=True, verbose=False):
    """扫描并（可选）删除 DU png 文件。"""

    _ts0 = datetime.now()
    deleted_count = 0
    total_size = 0
    scanned_populated = 0  # 有文件的子目录数
    total_files = 0        # 子目录下文件总数（包括非 png）

    for d in os.scandir(reco_dir):
        if not d.is_dir():
            continue
        sub_files = os.listdir(d.path)
        if not sub_files:
            continue
        scanned_populated += 1
        total_files += len(sub_files)

        for fname in sub_files:
            if not (fname.endswith(".png") and "_DU_" in fname):
                continue
            fp = os.path.join(d.path, fname)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            total_size += st.st_size
            deleted_count += 1
            if not dry_run:
                os.remove(fp)
                if verbose:
                    print(f"[del] {fp}")
            elif verbose:
                print(f"[dry-run] {fp}")

    elapsed = (datetime.now() - _ts0).total_seconds()

    print(f"\n{'DRY-RUN' if dry_run else 'REAL'}  扫描完成，耗时 {elapsed:.1f}s")
    print(f"  扫描子目录 (有文件): {scanned_populated}")
    print(f"  子目录下文件总数:    {total_files}")
    print(f"  匹配的 DU png:       {deleted_count}")
    print(f"  释放空间:            {human_size(total_size)} ({total_size} bytes)")
    if dry_run:
        print(f"\n  加 --run 参数真正执行删除。")


def main():
    parser = argparse.ArgumentParser(
        description="删除 Reconstruction 子目录中文件名包含 DU 的 png 图片"
    )
    parser.add_argument(
        "--reco-dir", default="./Reconstruction",
        help="Reconstruction 根目录 (默认: ./Reconstruction)",
    )
    parser.add_argument("--run", action="store_true", help="真正删除（默认 dry-run）")
    parser.add_argument("--verbose", action="store_true", help="逐文件打印详情（大量输出，慎用）")
    args = parser.parse_args()

    reco_dir = os.path.abspath(args.reco_dir)
    if not os.path.isdir(reco_dir):
        print(f"错误: 目录不存在 -- {reco_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"目标目录: {reco_dir}")
    print(f"模式:     {'真正删除' if args.run else 'DRY-RUN (预览)'}")
    print()

    clean(reco_dir, dry_run=not args.run, verbose=args.verbose)


if __name__ == "__main__":
    main()
