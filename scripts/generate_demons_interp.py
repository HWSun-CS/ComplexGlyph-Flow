# -*- coding: utf-8 -*-
"""Batch-generate Demons interpolation sequences between fonts."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from tqdm import tqdm

from Demons import DEFAULT_EXE, generate_demons_sequence


ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "out"
METADATA_DIR = ROOT_DIR / "metadata"
TRAIN_CHAR_FILE = ROOT_DIR / "train.txt"
TEST_CHAR_FILE = ROOT_DIR / "test.txt"
STATUS_PATH = METADATA_DIR / "demons_status.json"
EXPECTED_FRAMES = 12


def is_valid_png(png_path: Path) -> bool:
    """检查PNG文件是否有效"""
    try:
        from PIL import Image
        with Image.open(png_path) as img:
            img.verify()  # 验证文件完整性
        return True
    except Exception:
        return False


# ===== util =====================================================================================
def load_characters(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"字符清单不存在: {path}")
    content = path.read_text(encoding="utf-8")
    return [token for token in content.split() if token.strip()]


def load_status(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[WARN] {path} 解析失败，将重新生成。")
    return {}


def save_status(path: Path, data: Dict) -> None:
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def available_fonts() -> List[str]:
    if not OUT_DIR.exists():
        return []
    return sorted([p.name for p in OUT_DIR.iterdir() if p.is_dir()])


def subset_dir(font_name: str, subset: str) -> Path:
    """å…¼å®¹æ—§ target å±‚ç»“æž„çš„å­—ä½“å­é›†è·¯å¾„ï¼Œä¼˜å…ˆä½¿ç”¨æ–°ç»“æž„ã€?"""
    modern = OUT_DIR / font_name / subset
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



def ensure_font_exists(font_name: str, subset: str) -> None:
    path = subset_dir(font_name, subset)
    if not path.exists():
        raise FileNotFoundError(f"未找到字体目录: {path}")


def count_frames(char_dir: Path) -> int:
    if not char_dir.exists():
        return 0
    return sum(1 for p in char_dir.glob("*.png") if p.is_file())


def append_failure(records: List[Dict], char: str, reason: str) -> None:
    for item in records:
        if item.get("char") == char:
            if reason not in item.get("reason", ""):
                item["reason"] = f"{item['reason']} | {reason}"
            return
    records.append({"char": char, "reason": reason})


def subset_to_char_file(subset: str, train_file: Path, test_file: Path) -> Path:
    if subset == "train":
        return train_file
    if subset == "test":
        return test_file
    raise ValueError(f"未知子集: {subset}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="批量生成 Demons 插值序列")
    parser.add_argument(
        "--source-font",
        default="思源黑体",
        help="源字体（moving），默认: 思源黑体",
    )
    parser.add_argument(
        "--target-fonts",
        nargs="+",
        help="只处理指定目标字体（空间分隔的字体名称）",
    )
    parser.add_argument(
        "--subset",
        choices=["train", "test", "all"],
        default="all",
        help="要处理的子集，train/test/all（默认 all）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重建已有的插值结果（会覆盖原有输出）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅列出将要生成的任务，不实际运行 Demons",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Demons 运行时打印详细日志（默认关闭）",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_CHAR_FILE,
        help="训练字符清单文件 (默认: train.txt)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TEST_CHAR_FILE,
        help="测试字符清单文件 (默认: test.txt)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=None,
        help="仅处理每个子集的前 N 个字符（调试用）",
    )
    parser.add_argument(
        "--exe-path",
        type=Path,
        default=DEFAULT_EXE,
        help="LogDomainDemonsRegistration 可执行文件路径 (默认: 项目根目录下同名文件)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="并行进程数（默认: 8）",
    )
    return parser.parse_args(argv)


def collect_target_fonts(source_font: str, requested: Sequence[str] | None) -> List[str]:
    fonts = available_fonts()
    if source_font not in fonts:
        raise FileNotFoundError(f"源字体未生成: {source_font} (out/{source_font})")

    if requested:
        unknown = [name for name in requested if name not in fonts]
        if unknown:
            raise FileNotFoundError(f"以下字体尚未生成 train/test 图像: {', '.join(unknown)}")
        targets = [name for name in requested if name != source_font]
    else:
        targets = [name for name in fonts if name != source_font]

    if not targets:
        raise ValueError("没有需要处理的目标字体。")
    return targets


def prepare_characters(subsets: Iterable[str], train_file: Path, test_file: Path, max_chars: int | None) -> Dict[str, List[str]]:
    chars: Dict[str, List[str]] = {}
    for subset in subsets:
        char_file = subset_to_char_file(subset, train_file, test_file)
        subset_chars = load_characters(char_file)
        if max_chars is not None:
            subset_chars = subset_chars[: max(0, max_chars)]
        chars[subset] = subset_chars
    return chars


def process_single_char(
    char: str,
    source_font: str,
    target_font: str,
    subset: str,
    exe_path: Path,
    force: bool,
    dry_run: bool,
    verbose: bool,
) -> Dict:
    """处理单个字符的Demons插值，返回结果字典。"""
    result = {
        "char": char,
        "success": False,
        "skipped": False,
        "planned": False,
        "missing_source": False,
        "missing_target": False,
        "error": None,
        "frames": 0,
    }
    
    source_subset_dir = subset_dir(source_font, subset)
    target_subset_dir = subset_dir(target_font, subset)
    moving_path = source_subset_dir / f"{char}.png"
    fixed_path = target_subset_dir / f"{char}.png"
    base_dir = OUT_DIR / target_font / subset
    char_dir = base_dir / char
    
    if not moving_path.exists():
        result["missing_source"] = True
        return result
    if not fixed_path.exists():
        result["missing_target"] = True
        return result
    
    if char_dir.exists():
        existing_count = count_frames(char_dir)
        if existing_count >= EXPECTED_FRAMES and not force:
            result["skipped"] = True
            result["frames"] = existing_count
            return result
        if existing_count > 0:
            shutil.rmtree(char_dir, ignore_errors=True)
    
    if dry_run:
        result["planned"] = True
        return result
    
    try:
        generated = generate_demons_sequence(
            moving_path=moving_path,
            fixed_path=fixed_path,
            output_dir=char_dir,
            exe_path=exe_path,
            verbose=verbose,
        )
        n_frames = len(generated)
        if n_frames < EXPECTED_FRAMES:
            result["error"] = f"生成帧数不足: {n_frames}/{EXPECTED_FRAMES}"
            shutil.rmtree(char_dir, ignore_errors=True)
            return result
        
        # 验证生成的插值是否完整（正好12张有效图片）
        if char_dir.exists():
            png_files = sorted(char_dir.glob("*.png"))
            if len(png_files) == EXPECTED_FRAMES:
                all_valid = True
                for png_file in png_files:
                    if not is_valid_png(png_file):
                        all_valid = False
                        break
                if all_valid:
                    result["success"] = True
                    result["frames"] = n_frames
                else:
                    result["error"] = f"生成的插值图片无效: {len(png_files)}张图片中部分无效"
                    shutil.rmtree(char_dir, ignore_errors=True)
            else:
                result["error"] = f"生成的插值图片数量不正确: {len(png_files)}/{EXPECTED_FRAMES}"
                shutil.rmtree(char_dir, ignore_errors=True)
        else:
            result["error"] = "插值目录未创建"
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"Demons失败: {exc}"
        if char_dir.exists():
            shutil.rmtree(char_dir, ignore_errors=True)

    return result


def generate_for_font(
    source_font: str,
    target_font: str,
    subsets: Iterable[str],
    chars_map: Dict[str, List[str]],
    *,
    exe_path: Path,
    force: bool,
    dry_run: bool,
    verbose: bool,
    workers: int = 8,
) -> Dict[str, Dict]:
    ensure_font_exists(source_font, "train")
    ensure_font_exists(source_font, "test")

    results: Dict[str, Dict] = {}

    for subset in subsets:
        ensure_font_exists(target_font, subset)
        chars = chars_map[subset]

        base_dir = OUT_DIR / target_font / subset
        subset_stats = {
            "total_chars": len(chars),
            "generated": 0,
            "skipped": 0,
            "planned": 0,
            "missing_source": [],
            "missing_target": [],
            "failed_chars": [],
            "generated_frames_total": 0,
            "avg_frames": 0.0,
        }
        
        # 使用进程池并行处理
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    process_single_char,
                    char, source_font, target_font, subset,
                    exe_path, force, dry_run, verbose
                ): char
                for char in chars
            }
            
            progress = tqdm(total=len(chars), desc=f"{target_font}/{subset}", unit="char")
            for future in as_completed(futures):
                char = futures[future]
                try:
                    result = future.result()
                    
                    if result["missing_source"]:
                        subset_stats["missing_source"].append(char)
                    elif result["missing_target"]:
                        subset_stats["missing_target"].append(char)
                    elif result["skipped"]:
                        subset_stats["skipped"] += 1
                    elif result["planned"]:
                        subset_stats["planned"] += 1
                    elif result["error"]:
                        append_failure(subset_stats["failed_chars"], char, result["error"])
                        tqdm.write(f"[ERROR] {target_font}/{subset}/{char}: {result['error']}")
                        # 如果是退出码1错误，建议检查输入文件
                        if "退出码 1" in result["error"]:
                            tqdm.write(f"[HINT] 检查输入文件: out/{target_font}/{subset}/{char}.png 和 out/{source_font}/{subset}/{char}.png")
                    elif result["success"]:
                        subset_stats["generated"] += 1
                    
                    progress.set_postfix({"gen": subset_stats["generated"], "skip": subset_stats["skipped"]})
                
                except Exception as exc:  # noqa: BLE001
                    reason = f"进程异常: {exc}"
                    append_failure(subset_stats["failed_chars"], char, reason)
                    tqdm.write(f"[ERROR] {target_font}/{subset}/{char}: {reason}")
                
                progress.update(1)
            
            progress.close()

        subset_stats["missing_source"] = sorted(set(subset_stats["missing_source"]))
        subset_stats["missing_target"] = sorted(set(subset_stats["missing_target"]))
        if subset_stats["failed_chars"]:
            tqdm.write(
                f"[WARN] {target_font}/{subset} 有 {len(subset_stats['failed_chars'])} 个字符插值失败，"
                "详情见 demons_status.json。"
            )
        if subset_stats["missing_source"] or subset_stats["missing_target"]:
            tqdm.write(
                f"[WARN] {target_font}/{subset} 缺失图像："
                f"source {len(subset_stats['missing_source'])} 个，"
                f"target {len(subset_stats['missing_target'])} 个。"
            )

        complete_chars: List[str] = []
        partial_chars: List[Tuple[str, int]] = []
        total_frames = 0
        for char in chars:
            frame_count = count_frames(base_dir / char)
            if frame_count >= EXPECTED_FRAMES:
                complete_chars.append(char)
                total_frames += frame_count
            elif frame_count > 0:
                partial_chars.append((char, frame_count))

        for char, fc in partial_chars:
            reason = f"帧数不足: {fc}/{EXPECTED_FRAMES}"
            append_failure(subset_stats["failed_chars"], char, reason)

        subset_stats["generated"] = len(complete_chars)
        subset_stats["generated_frames_total"] = total_frames
        subset_stats["avg_frames"] = (
            round(total_frames / len(complete_chars), 3) if complete_chars else 0.0
        )

        total_done = subset_stats["generated"] + subset_stats["skipped"]
        subset_stats["completed"] = (
            subset_stats["total_chars"] == 0
            or (
                total_done >= subset_stats["total_chars"]
                and not subset_stats["failed_chars"]
                and not subset_stats["missing_source"]
                and not subset_stats["missing_target"]
                and subset_stats["planned"] == 0
            )
        )

        results[subset] = subset_stats

    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    try:
        targets = collect_target_fonts(args.source_font, args.target_fonts)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        return 1

    subsets = ["train", "test"] if args.subset == "all" else [args.subset]

    try:
        chars_map = prepare_characters(subsets, args.train_file, args.test_file, args.max_chars)
    except FileNotFoundError as exc:
        print(f"[ERROR] {exc}")
        return 1

    status = load_status(STATUS_PATH)
    now = datetime.now().isoformat(timespec="seconds")

    if args.dry_run:
        print("[INFO] Dry-run 模式：不会执行 Demons，仅统计任务。")

    font_progress = tqdm(targets, desc="Demons 插值", unit="font")
    for target_font in font_progress:
        print("=" * 60)
        print(f"[INFO] 处理目标字体: {target_font}  (source={args.source_font})")
        subset_results = generate_for_font(
            args.source_font,
            target_font,
            subsets,
            chars_map,
            exe_path=args.exe_path,
            force=args.force,
            dry_run=args.dry_run,
            verbose=args.verbose,
            workers=args.workers,
        )

        completed = all(info.get("completed", False) for info in subset_results.values())

        status[target_font] = {
            "source_font": args.source_font,
            "updated_at": now,
            "subset_mode": args.subset,
            "completed": completed,
            "subsets": subset_results,
        }
        font_progress.set_postfix({"completed": int(completed)})

    font_progress.close()

    if args.dry_run:
        print("[INFO] Dry-run 完成，未写入任何文件。")
        return 0

    save_status(STATUS_PATH, status)
    print("[INFO] 全部处理完成。状态记录已更新。")
    return 0


if __name__ == "__main__":
    sys.exit(main())

