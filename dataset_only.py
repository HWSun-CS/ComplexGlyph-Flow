"""Convenience entry point to only generate dataset PNGs (no Demons interpolation).

Defaults are identical to running:
    python run.py --skip-demons --force-dataset

Extra options allow overriding input/output/metadata paths.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def override_dataset_paths(
    *,
    font_dir: Path | None,
    out_dir: Path | None,
    metadata_dir: Path | None,
    train_file: Path | None,
    test_file: Path | None,
) -> None:
    """Patch generate_dataset module paths so it writes/reads where we want."""
    import scripts.generate_dataset as ds  # noqa: WPS433

    if font_dir is not None:
        ds.FONT_DIR = font_dir
    if out_dir is not None:
        ds.OUT_DIR = out_dir
    if metadata_dir is not None:
        ds.METADATA_DIR = metadata_dir
        ds.MISSING_CSV = metadata_dir / "missing_chars.csv"
        ds.FONT_STATUS_JSON = metadata_dir / "font_status.json"
    if train_file is not None:
        ds.TRAIN_CHAR_FILE = train_file
    if test_file is not None:
        ds.TEST_CHAR_FILE = test_file


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate dataset PNGs only (no Demons).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--fonts", nargs="+", help="Only process specified fonts.")
    parser.add_argument("--force", action="store_true", help="Force regenerate PNGs.")
    parser.add_argument("--dry-run", action="store_true", help="List work without writing files.")
    parser.add_argument(
        "--font-dir",
        type=Path,
        help="Custom font directory (default: repo_root/font).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        help="Custom output directory for generated PNGs (default: repo_root/out).",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        help="Custom metadata directory for status/missing files (default: repo_root/metadata).",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        help="Custom train character list file (default: repo_root/train.txt).",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Custom test character list file (default: repo_root/test.txt).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    args = parse_args(argv)

    # Patch paths before importing main generator.
    override_dataset_paths(
        font_dir=args.font_dir,
        out_dir=args.out_dir,
        metadata_dir=args.metadata_dir,
        train_file=args.train_file,
        test_file=args.test_file,
    )

    import scripts.generate_dataset as ds  # noqa: WPS433

    dataset_args: list[str] = []
    if args.force:
        dataset_args.append("--force")
    if args.dry_run:
        dataset_args.append("--dry-run")
    if args.fonts:
        dataset_args.extend(["--fonts", *args.fonts])
    if args.train_file:
        dataset_args.extend(["--train-file", str(args.train_file)])
    if args.test_file:
        dataset_args.extend(["--test-file", str(args.test_file)])

    return ds.main(dataset_args)


if __name__ == "__main__":
    raise SystemExit(main())
