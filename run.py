"""一键执行字体字符生成 + Demons 字体验证插值流水线。"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def ensure_sys_path(project_root: Path) -> None:
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字体数据集生成与 Demons 插值一键入口")
    parser.add_argument(
        "--source-font",
        default="思源黑体",
        help="Demons 插值的源字体 (moving)，默认思源黑体",
    )
    parser.add_argument(
        "--skip-dataset",
        action="store_true",
        help="跳过字体字符生成阶段",
    )
    parser.add_argument(
        "--skip-demons",
        action="store_true",
        help="跳过 Demons 插值阶段",
    )
    parser.add_argument(
        "--force-dataset",
        action="store_true",
        help="忽略 font_status，强制重新生成所有字体字符图像",
    )
    parser.add_argument(
        "--force-demons",
        action="store_true",
        help="忽略 demons_status，强制重新生成所有 Demons 插值结果",
    )
    parser.add_argument(
        "--demons-subset",
        choices=["train", "test", "all"],
        default="all",
        help="Demons 插值处理的子集 (train/test/all)，默认 all",
    )
    parser.add_argument(
        "--demons-max-chars",
        type=int,
        default=None,
        help="Demons 插值调试用：仅处理前 N 个字符",
    )
    parser.add_argument(
        "--demons-verbose",
        action="store_true",
        help="Demons 插值过程输出详细日志",
    )
    parser.add_argument(
        "--demons-workers",
        type=int,
        default=8,
        help="Demons 插值并行进程数（默认: 8）",
    )
    parser.add_argument(
        "--fonts",
        nargs="+",
        help="仅处理指定字体（文件名或不带后缀的字体名），不指定则处理所有字体",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="详细验证模式：检查所有文件完整性，重建状态文件，跳过有问题的项目",
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="自动修复模式：发现损坏文件时自动重新生成",
    )
    return parser.parse_args(argv)


def font_stems(font_dir: Path) -> List[str]:
    if not font_dir.exists():
        return []
    exts = {".ttf", ".otf", ".ttc"}
    return sorted(
        {p.stem for p in font_dir.iterdir() if p.is_file() and p.suffix.lower() in exts}
    )


def choose_subset_dir(out_dir: Path, font_name: str, subset: str) -> Path:
    """å¹¶å¯å…¼å®¹æ—§ç»“æž„çš„æ–‡ä»¶è·¯å¾„ï¼šä¼˜å…ˆä½¿ç”¨æ— target å±‚çš„æ–°ç»“æž„ã€?"""
    modern = out_dir / font_name / subset
    legacy = modern / "target"
    modern_has = modern.exists() and any(modern.glob("*.png"))
    legacy_has = legacy.exists() and any(legacy.glob("*.png"))

    if modern_has:
        return modern
    if legacy_has:
        return legacy
    if modern.exists():
        return modern
    if legacy.exists():
        return legacy
    return modern


def verify_font_dataset(font_name: str, out_dir: Path) -> Tuple[bool, str, int, int, List[str], List[str]]:
    """验证字体数据集的完整性，返回 (is_valid, reason, train_count, test_count, invalid_train_files, invalid_test_files)"""
    train_dir = choose_subset_dir(out_dir, font_name, "train")
    test_dir = choose_subset_dir(out_dir, font_name, "test")

    if not train_dir.exists() or not test_dir.exists():
        return False, "目录不存在", 0, 0, [], []

    train_files = list(train_dir.glob("*.png"))
    test_files = list(test_dir.glob("*.png"))

    train_count = len(train_files)
    test_count = len(test_files)

    # 检查PNG文件是否有效
    invalid_train = []
    invalid_test = []

    for png_file in train_files:
        if not is_valid_png(png_file):
            invalid_train.append(png_file.stem)  # 只保存字符名，不带.png

    for png_file in test_files:
        if not is_valid_png(png_file):
            invalid_test.append(png_file.stem)  # 只保存字符名，不带.png

    if invalid_train or invalid_test:
        return False, f"无效PNG文件: train({len(invalid_train)}), test({len(invalid_test)})", train_count, test_count, invalid_train, invalid_test

    return True, "正常", train_count, test_count, [], []


def verify_interpolation(font_name: str, source_font: str, out_dir: Path) -> Tuple[bool, str, Dict[str, int]]:
    """验证插值数据的完整性，返回 (is_valid, reason, stats)"""
    subsets = ["train", "test"]
    subset_stats = {}

    for subset in subsets:
        base_dir = out_dir / font_name / subset
        if not base_dir.exists():
            subset_stats[subset] = 0
            continue

        char_dirs = [d for d in base_dir.iterdir() if d.is_dir()]
        valid_count = 0
        invalid_count = 0

        for char_dir in char_dirs:
            png_files = list(char_dir.glob("*.png"))
            if len(png_files) == 12:
                # 检查所有12张图片是否都存在且有效
                all_valid = True
                for png_file in png_files:
                    if not is_valid_png(png_file):
                        all_valid = False
                        break
                if all_valid:
                    valid_count += 1
                else:
                    invalid_count += 1
            else:
                invalid_count += 1

        subset_stats[subset] = valid_count

    total_valid = sum(subset_stats.values())
    if total_valid == 0:
        return False, "没有有效的插值序列", subset_stats

    return True, f"有效插值序列: {total_valid}", subset_stats


def is_valid_png(png_path: Path) -> bool:
    """检查PNG文件是否有效"""
    try:
        from PIL import Image
        with Image.open(png_path) as img:
            img.verify()  # 验证文件完整性
        return True
    except Exception:
        return False


# 全局变量用于收集需要修复的字体和文件
fonts_needing_repair = set()
chars_needing_repair = {"train": set(), "test": set()}


def rebuild_status_files(verify_mode: bool = False) -> None:
    """重建所有状态文件"""
    print("🔍 重建状态文件...")

    # 确保metadata目录存在
    metadata_dir = Path("metadata")
    metadata_dir.mkdir(exist_ok=True)

    global fonts_needing_repair, chars_needing_repair
    fonts_needing_repair.clear()
    chars_needing_repair["train"].clear()
    chars_needing_repair["test"].clear()

    # 重建字体状态
    font_status = {}
    dataset_dir = Path("out")
    fonts_available = font_stems(Path("font"))

    for font_name in fonts_available:
        is_valid, reason, train_count, test_count, invalid_train, invalid_test = verify_font_dataset(font_name, dataset_dir)

        # 记录需要修复的字体和字符
        if invalid_train or invalid_test:
            fonts_needing_repair.add(font_name)
            chars_needing_repair["train"].update(invalid_train)
            chars_needing_repair["test"].update(invalid_test)
            if verify_mode:
                print(f"⚠️ 发现损坏文件: {font_name} - train({len(invalid_train)}), test({len(invalid_test)})")

        if is_valid or not verify_mode:  # verify_mode下只记录有效的
            font_status[font_name] = {
                "font_file": f"{font_name}.TTF",  # 简化处理
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "train_count": train_count,
                "test_count": test_count,
                "missing_count": 0,  # 暂时设为0，后续可以改进
                "completed": is_valid,
            }

    # 保存字体状态
    with open("metadata/font_status.json", "w", encoding="utf-8") as f:
        json.dump(font_status, f, ensure_ascii=False, indent=2)

    # 重建Demons状态
    demons_status = {}
    source_font = "思源黑体"

    for target_font in fonts_available:
        if target_font == source_font:
            continue

        is_valid, reason, subset_stats = verify_interpolation(target_font, source_font, dataset_dir)

        if is_valid or not verify_mode:  # verify_mode下只记录有效的
            demons_status[target_font] = {
                "source_font": source_font,
                "updated_at": datetime.now().isoformat(timespec="seconds"),
                "subset_mode": "all",
                "completed": is_valid,
                "subsets": {
                    "train": {
                        "generated": subset_stats.get("train", 0),
                        "completed": subset_stats.get("train", 0) > 0,
                    },
                    "test": {
                        "generated": subset_stats.get("test", 0),
                        "completed": subset_stats.get("test", 0) > 0,
                    }
                }
            }

    # 保存Demons状态
    with open("metadata/demons_status.json", "w", encoding="utf-8") as f:
        json.dump(demons_status, f, ensure_ascii=False, indent=2)

    # 在验证模式下报告发现的问题
    if verify_mode and (fonts_needing_repair or chars_needing_repair["train"] or chars_needing_repair["test"]):
        print(f"\n🔧 发现需要修复的内容:")
        if fonts_needing_repair:
            print(f"  - 需要重新生成数据集的字体: {', '.join(sorted(fonts_needing_repair))}")
        if chars_needing_repair["train"] or chars_needing_repair["test"]:
            print(f"  - train集中损坏的字符: {', '.join(sorted(chars_needing_repair['train']))}")
            print(f"  - test集中损坏的字符: {', '.join(sorted(chars_needing_repair['test']))}")
        print("\n💡 建议运行以下命令修复:")
        if fonts_needing_repair:
            print(f"  python run.py --fonts {' '.join(sorted(fonts_needing_repair))} --force-dataset")
        print("  python run.py --skip-dataset --force-demons")
    print("✅ 状态文件重建完成")


def has_generated_dataset(font_name: str, out_dir: Path) -> bool:
    train_dir = choose_subset_dir(out_dir, font_name, "train")
    test_dir = choose_subset_dir(out_dir, font_name, "test")
    return (
        train_dir.exists()
        and any(train_dir.glob("*.png"))
        and test_dir.exists()
        and any(test_dir.glob("*.png"))
    )


def dataset_counts(status: Dict[str, Dict]) -> Dict[str, int]:
    return {
        font: status.get(font, {}).get("train_count", 0)
        + status.get(font, {}).get("test_count", 0)
        for font in status
    }


def count_interpolation_sequences(
    font_name: str, subset: str, source_font: str, out_dir: Path
) -> int:
    base_dir = out_dir / font_name / subset
    if not base_dir.exists():
        return 0
    total = 0
    for char_dir in base_dir.iterdir():
        if char_dir.is_dir() and any(char_dir.glob("*.png")):
            total += 1
    return total


def interpolation_counts(
    fonts: Iterable[str], subsets: Iterable[str], source_font: str, out_dir: Path
) -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for font in fonts:
        stats[font] = {}
        for subset in subsets:
            stats[font][subset] = count_interpolation_sequences(
                font, subset, source_font, out_dir
            )
    return stats


def needs_dataset_generation(
    font_name: str,
    *,
    status: Dict[str, Dict],
    out_dir: Path,
    force: bool,
) -> bool:
    if force:
        return True
    if not has_generated_dataset(font_name, out_dir):
        return True
    entry = status.get(font_name)
    if not entry or not entry.get("completed"):
        return True
    return False


def needs_demons_generation(
    font_name: str,
    *,
    status: Dict[str, Dict],
    subsets: Iterable[str],
    source_font: str,
    force: bool,
) -> bool:
    if force:
        return True
    entry = status.get(font_name)
    if not entry or entry.get("source_font") != source_font:
        return True
    subset_info = entry.get("subsets", {})
    for subset in subsets:
        info = subset_info.get(subset)
        if not info or not info.get("completed"):
            return True
    return False


@dataclass
class StageResult:
    processed_fonts: List[str]
    created_units: int
    completed: bool


def dataset_stage(
    *,
    args: argparse.Namespace,
    fonts_available: List[str],
    dataset_module,
) -> StageResult:
    font_status = dataset_module.load_font_status()
    before_counts = dataset_counts(font_status)

    # 如果指定了字体列表，只处理这些字体
    if args.fonts:
        fonts_to_process = [
            font for font in fonts_available
            if any(spec_font.lower() in font.lower() or spec_font.lower() == font.lower().replace('.ttf', '').replace('.otf', '').replace('.ttc', '')
                   for spec_font in args.fonts)
        ]
        # 去重
        fonts_to_process = list(set(fonts_to_process))
    else:
        fonts_to_process = [
            font
            for font in fonts_available
            if needs_dataset_generation(
                font,
                status=font_status,
                out_dir=dataset_module.OUT_DIR,
                force=args.force_dataset,
            )
        ]

    # 如果有自动发现的损坏字体，也要加入处理列表
    if fonts_needing_repair:
        for font in fonts_needing_repair:
            if font in fonts_available and font not in fonts_to_process:
                fonts_to_process.append(font)
                if not args.fonts:  # 如果不是指定字体模式，给出提示
                    print(f"🔧 自动加入需要修复的字体: {font}")

    # 如果指定了字体或强制生成，需要包含这些字体
    if args.fonts or args.force_dataset:
        for font in fonts_available:
            if args.fonts and any(spec_font.lower() in font.lower() or spec_font.lower() == font.lower().replace('.ttf', '').replace('.otf', '').replace('.ttc', '')
                   for spec_font in args.fonts):
                if font not in fonts_to_process:
                    fonts_to_process.append(font)
            elif args.force_dataset and font not in fonts_to_process:
                fonts_to_process.append(font)

    if args.skip_dataset:
        print("[INFO] 已跳过字体字符生成阶段 (--skip-dataset)。")
        return StageResult([], 0, False)

    if not fonts_to_process:
        print("[INFO] 没有需要生成字符图像的字体。")
    else:
        print("[INFO] 待处理字体：", "、".join(fonts_to_process))
        dataset_args: List[str] = []
        if args.force_dataset:
            dataset_args.append("--force")
        dataset_args.extend(["--fonts", *fonts_to_process])
        ret = dataset_module.main(dataset_args)
        if ret != 0:
            raise SystemExit(ret)

    font_status_after = dataset_module.load_font_status()
    after_counts = dataset_counts(font_status_after)

    created_total = sum(
        max(0, after_counts.get(font, 0) - before_counts.get(font, 0))
        for font in fonts_available
    )

    all_completed = all(
        has_generated_dataset(font, dataset_module.OUT_DIR)
        and font_status_after.get(font, {}).get("completed")
        for font in fonts_available
    )

    if created_total > 0:
        print(f"[INFO] 本阶段新增字符图像 {created_total} 张。")

    return StageResult(fonts_to_process, created_total, all_completed)


def demons_stage(
    *,
    args: argparse.Namespace,
    fonts_available: List[str],
    subsets: List[str],
    dataset_module,
    demons_module,
) -> StageResult:
    target_fonts = [font for font in fonts_available if font != args.source_font]

    if args.skip_demons:
        print("[INFO] 已跳过 Demons 插值阶段 (--skip-demons)。")
        return StageResult([], 0, not target_fonts)

    if not target_fonts:
        print("[INFO] 没有可用于 Demons 插值的目标字体。")
        return StageResult([], 0, True)

    status_before = demons_module.load_status(demons_module.STATUS_PATH)
    before_counts = interpolation_counts(
        target_fonts, subsets, args.source_font, dataset_module.OUT_DIR
    )

    fonts_to_process = [
        font
        for font in target_fonts
        if needs_demons_generation(
            font,
            status=status_before,
            subsets=subsets,
            source_font=args.source_font,
            force=args.force_demons,
        )
    ]

    if not fonts_to_process:
        print("[INFO] 所有目标字体的 Demons 插值已完成。")
    else:
        print("[INFO] Demons 待处理字体：", "、".join(fonts_to_process))
        demons_args: List[str] = [
            "--source-font",
            args.source_font,
            "--target-fonts",
            *fonts_to_process,
        ]
        if args.force_demons:
            demons_args.append("--force")
        if args.demons_subset != "all":
            demons_args.extend(["--subset", args.demons_subset])
        if args.demons_max_chars is not None:
            demons_args.extend(["--max-chars", str(args.demons_max_chars)])
        if args.demons_verbose:
            demons_args.append("--verbose")
        demons_args.extend(["--workers", str(args.demons_workers)])
        ret = demons_module.main(demons_args)
        if ret != 0:
            raise SystemExit(ret)

    status_after = demons_module.load_status(demons_module.STATUS_PATH)
    after_counts = interpolation_counts(
        target_fonts, subsets, args.source_font, dataset_module.OUT_DIR
    )

    created_sequences = 0
    for font in target_fonts:
        for subset in subsets:
            created_sequences += max(
                0,
                after_counts.get(font, {}).get(subset, 0)
                - before_counts.get(font, {}).get(subset, 0),
            )

    all_completed = all(
        not needs_demons_generation(
            font,
            status=status_after,
            subsets=subsets,
            source_font=args.source_font,
            force=False,
        )
        for font in target_fonts
    )

    if created_sequences > 0:
        print(f"[INFO] 本阶段新增插值序列 {created_sequences} 组。")

    return StageResult(fonts_to_process, created_sequences, all_completed)


def main(argv: Sequence[str] | None = None) -> int:
    project_root = Path(__file__).resolve().parent
    ensure_sys_path(project_root)

    args = parse_args(argv or sys.argv[1:])

    import scripts.generate_dataset as dataset_module
    import scripts.generate_demons_interp as demons_module

    # 每次运行都重建状态文件，确保数据一致性
    rebuild_status_files(verify_mode=False)

    # 如果是验证模式，先重建状态文件
    if args.verify:
        print("🔍 进入详细验证模式...")
        rebuild_status_files(verify_mode=True)
        print("✅ 验证完成，所有状态文件已重建")
        return 0

    # 如果是自动修复模式，检查是否有损坏文件需要修复
    if args.auto_fix:
        print("🔧 进入自动修复模式...")
        rebuild_status_files(verify_mode=True)
        if fonts_needing_repair:
            print(f"📋 发现 {len(fonts_needing_repair)} 个字体需要修复，正在自动修复...")
            # 自动修复损坏的字体数据集
            auto_fix_args = ["--fonts"] + list(fonts_needing_repair) + ["--force"]
            try:
                dataset_module.main(auto_fix_args)
                print("✅ 字体数据集修复完成")
            except SystemExit as e:
                if e.code != 0:
                    print(f"❌ 字体数据集修复失败，退出码: {e.code}")
                    return e.code

            # 重新验证
            rebuild_status_files(verify_mode=False)
        else:
            print("✅ 没有发现需要修复的损坏文件")

    # 如果只是验证+修复模式，处理完就退出
    if args.verify and args.auto_fix:
        return 0

    fonts_available = font_stems(dataset_module.FONT_DIR)
    if not fonts_available:
        print("[ERROR] 未在 font/ 目录下找到任何字体文件。")
        return 1

    train_chars = dataset_module.load_characters(dataset_module.TRAIN_CHAR_FILE)
    test_chars = dataset_module.load_characters(dataset_module.TEST_CHAR_FILE)

    dataset_result = dataset_stage(
        args=args,
        fonts_available=fonts_available,
        dataset_module=dataset_module,
    )

    subset_option = args.demons_subset
    subsets = ["train", "test"] if subset_option == "all" else [subset_option]

    # 若 subset 指定，但对应字符集为空，直接跳过
    subset_map = {"train": train_chars, "test": test_chars}
    subsets = [subset for subset in subsets if subset_map.get(subset)]
    if not subsets and not args.skip_demons:
        print("[WARN] 未找到有效字符清单，跳过 Demons 插值。")
        demons_result = StageResult([], 0, True)
    else:
        demons_result = demons_stage(
            args=args,
            fonts_available=fonts_available,
            subsets=subsets,
            dataset_module=dataset_module,
            demons_module=demons_module,
        )

    if dataset_result.created_units == 0 and demons_result.created_units == 0:
        if dataset_result.completed and demons_result.completed:
            print("[INFO] 当前字体都制作过了。")
        else:
            print("[INFO] 没有新增内容。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
