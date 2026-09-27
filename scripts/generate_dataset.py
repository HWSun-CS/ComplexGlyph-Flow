"""字体字符图像批量生成脚本。

功能概述：
1. 针对 `font/` 目录下的字体文件生成字符图像，输出至 `out/<字体名>/train` 和
   `out/<字体名>/test`。
2. 每次运行时自动跳过已完成生成的字体，避免重复工作，可通过命令行参数强制重建。
3. 当字体缺少某个字符的字形时自动跳过，同时记录缺失信息到 `metadata/missing_chars.csv`。
4. 支持命令行过滤字体、仅预览（dry-run）和自定义字符清单文件。

生成规则：
- 图片尺寸：256x256 像素
- 背景：白色 (#FFFFFF)
- 字体颜色：黑色 (#000000)
- 对齐：基于样本字符 bbox 平均值确定统一标准
- 字高占比：样本字符实际显示约占画布 60%

依赖：需要安装 Pillow 库 (pip install Pillow)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# 基础路径配置
ROOT_DIR = Path(__file__).resolve().parent.parent
FONT_DIR = ROOT_DIR / "font"
OUT_DIR = ROOT_DIR / "out"
METADATA_DIR = ROOT_DIR / "metadata"
TRAIN_CHAR_FILE = ROOT_DIR / "train.txt"
TEST_CHAR_FILE = ROOT_DIR / "test.txt"
MISSING_CSV = METADATA_DIR / "missing_chars.csv"
FONT_STATUS_JSON = METADATA_DIR / "font_status.json"


def subset_dir(font_name: str, subset: str) -> Path:
    """å½“å‰ç»“æž„ï¼šout/<font>/<subset>ï¼Œæ— target å±‚ã€?"""
    return OUT_DIR / font_name / subset


def legacy_subset_dir(font_name: str, subset: str) -> Path:
    """çº¦å®šè¿‡åŽ»ç»“æž„çš„å…¼å®¹è·¯å¾„ï¼šout/<font>/<subset>/targetã€?"""
    return OUT_DIR / font_name / subset / "target"


# 渲染参数
IMAGE_SIZE = 256
TARGET_LINEAR_SCALE = 0.6
BACKGROUND_COLOR = 255
FOREGROUND_COLOR = 0
SAMPLE_COUNT = 10


@dataclass
class RenderResult:
    character: str
    font_name: str
    subset: str
    succeeded: bool
    message: str = ""


@dataclass
class StandardMetrics:
    """统一对齐标准。"""
    font_size: int
    avg_offset_x: float
    avg_offset_y: float
    avg_width: float
    avg_height: float


def ensure_directories() -> None:
    """确保脚本运行所需的目录存在。"""

    FONT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def load_characters(path: Path) -> List[str]:
    """从文本文件加载字符清单。"""

    if not path.exists():
        print(f"[WARN] 未找到字符文件: {path}")
        return []

    content = path.read_text(encoding="utf-8")
    chars = [token for token in content.split() if token.strip()]
    return chars


def load_font_status() -> Dict[str, dict]:
    if FONT_STATUS_JSON.exists():
        try:
            return json.loads(FONT_STATUS_JSON.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print("[WARN] font_status.json 解析失败，将重新生成。")
    return {}


def save_font_status(status: Dict[str, dict]) -> None:
    FONT_STATUS_JSON.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _subset_has_images(font_name: str, subset: str) -> bool:
    """æŸ¥çœ‹æ–°/æ—§æ ‡é‡æ˜¯å¦å·²æœ‰PNGï¼Œä¾¿äºŽè·³è¿‡é‡ç”Ÿæˆã€?"""
    modern = subset_dir(font_name, subset)
    legacy = legacy_subset_dir(font_name, subset)
    if modern.exists() and any(modern.glob("*.png")):
        return True
    if legacy.exists() and any(legacy.glob("*.png")):
        return True
    return False


def should_skip_font(font_name: str, status: Dict[str, dict], *, force: bool = False) -> bool:
    if force:
        return False
    if status.get(font_name, {}).get("completed"):
        return True

    if _subset_has_images(font_name, "train") and _subset_has_images(font_name, "test"):
        return True
    return False


def find_font_files(font_dir: Path) -> List[Path]:
    exts = {".ttf", ".otf", ".ttc"}
    return sorted([
        p for p in font_dir.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    ])


def check_character_renderable(font: ImageFont.FreeTypeFont, character: str) -> bool:
    """检查字符是否可被字体渲染（非空白/缺字）。"""
    try:
        mask = font.getmask(character, mode="L")
        bbox = mask.getbbox()
        return bbox is not None
    except Exception:
        return False


def find_optimal_font_size_for_char(font_path: Path, character: str) -> Optional[Tuple[ImageFont.FreeTypeFont, int]]:
    """二分法寻找使单个字符接近目标占比的字号。"""
    
    min_size = 1
    max_size = IMAGE_SIZE * 2
    target_pixels = IMAGE_SIZE * TARGET_LINEAR_SCALE
    
    best_font: Optional[ImageFont.FreeTypeFont] = None
    best_size = 0
    best_diff = float("inf")
    
    while min_size <= max_size:
        size = (min_size + max_size) // 2
        try:
            font = ImageFont.truetype(str(font_path), size=size)
        except OSError:
            return None
        
        if not check_character_renderable(font, character):
            return None
        
        dummy = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), character, font=font, anchor="lt")
        
        if bbox is None:
            return None
        
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        
        if width <= 0 or height <= 0:
            return None
        
        if width > IMAGE_SIZE or height > IMAGE_SIZE:
            max_size = size - 1
            continue
        
        # 检查宽高是否都在目标范围内
        if width <= target_pixels and height <= target_pixels:
            max_dim = max(width, height)
            diff = abs(max_dim - target_pixels)
            
            if diff < best_diff:
                best_diff = diff
                best_font = font
                best_size = size
            
            min_size = size + 1
        else:
            max_size = size - 1
    
    if best_font:
        return best_font, best_size
    return None


def compute_standard_metrics(font_path: Path, sample_chars: Sequence[str]) -> Optional[StandardMetrics]:
    """基于样本字符计算标准 metrics。"""
    
    collected_sizes: List[int] = []
    collected_offsets: List[Tuple[float, float, float, float]] = []
    
    for char in sample_chars:
        result = find_optimal_font_size_for_char(font_path, char)
        if not result:
            continue
        
        font, font_size = result
        collected_sizes.append(font_size)
        
        dummy = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=BACKGROUND_COLOR)
        draw = ImageDraw.Draw(dummy)
        bbox = draw.textbbox((0, 0), char, font=font, anchor="lt")
        
        if bbox is None:
            continue
        
        left, top, right, bottom = bbox
        width = right - left
        height = bottom - top
        
        offset_x = (IMAGE_SIZE - width) / 2 - left
        offset_y = (IMAGE_SIZE - height) / 2 - top
        
        collected_offsets.append((offset_x, offset_y, width, height))
        
        if len(collected_sizes) >= SAMPLE_COUNT:
            break
    
    if not collected_sizes or not collected_offsets:
        return None
    
    avg_font_size = int(round(sum(collected_sizes) / len(collected_sizes)))
    avg_offset_x = sum(item[0] for item in collected_offsets) / len(collected_offsets)
    avg_offset_y = sum(item[1] for item in collected_offsets) / len(collected_offsets)
    avg_width = sum(item[2] for item in collected_offsets) / len(collected_offsets)
    avg_height = sum(item[3] for item in collected_offsets) / len(collected_offsets)
    
    return StandardMetrics(
        font_size=avg_font_size,
        avg_offset_x=avg_offset_x,
        avg_offset_y=avg_offset_y,
        avg_width=avg_width,
        avg_height=avg_height,
    )


def render_character_image(
    font: ImageFont.FreeTypeFont,
    character: str,
    metrics: StandardMetrics,
) -> Image.Image:
    """使用统一标准 metrics 渲染字符。"""
    image = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(image)
    
    draw.text(
        (metrics.avg_offset_x, metrics.avg_offset_y),
        character,
        fill=FOREGROUND_COLOR,
        font=font,
        anchor="lt",
    )
    
    return image


def save_image(image: Image.Image, output_path: Path, *, dry_run: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dry_run:
        return
    image.save(output_path)


def update_missing_csv(missing_map: Dict[str, List[str]]) -> None:
    if not missing_map:
        if MISSING_CSV.exists():
            MISSING_CSV.unlink()
        return

    with MISSING_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["font_name", "missing_chars"])
        for font_name, chars in sorted(missing_map.items()):
            writer.writerow([font_name, " ".join(chars)])


def load_existing_missing() -> Dict[str, List[str]]:
    if not MISSING_CSV.exists():
        return {}

    missing: Dict[str, List[str]] = {}
    with MISSING_CSV.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            font_name = row.get("font_name", "").strip()
            chars = row.get("missing_chars", "").split()
            if font_name:
                missing[font_name] = chars
    return missing


def process_font(
    font_path: Path,
    train_chars: Sequence[str],
    test_chars: Sequence[str],
    *,
    dry_run: bool = False,
) -> Tuple[List[RenderResult], List[str]]:
    font_name = font_path.stem
    results: List[RenderResult] = []
    missing_chars: List[str] = []

    # 选取样本字符
    combined_chars: List[str] = []
    seen: Set[str] = set()
    for chars in [train_chars, test_chars]:
        for char in chars:
            if char in seen:
                continue
            combined_chars.append(char)
            seen.add(char)
            if len(combined_chars) >= SAMPLE_COUNT * 2:
                break
        if len(combined_chars) >= SAMPLE_COUNT * 2:
            break
    
    # 计算标准 metrics
    std_metrics = compute_standard_metrics(font_path, combined_chars)
    
    if std_metrics is None:
        error_msg = "无法计算标准 metrics"
        print(f"[ERROR] {font_name}: {error_msg}")
        for subset_name, chars in [("train", train_chars), ("test", test_chars)]:
            for char in chars:
                results.append(
                    RenderResult(char, font_name, subset_name, False, error_msg)
                )
        return results, missing_chars
    
    print(f"[INFO] {font_name} 标准字号: {std_metrics.font_size}, "
          f"平均偏移: ({std_metrics.avg_offset_x:.1f}, {std_metrics.avg_offset_y:.1f})")
    
    # 用标准字号加载字体
    try:
        font = ImageFont.truetype(str(font_path), size=std_metrics.font_size)
    except OSError as exc:
        error_msg = f"载入字体失败: {exc}"
        print(f"[ERROR] {font_name}: {error_msg}")
        for subset_name, chars in [("train", train_chars), ("test", test_chars)]:
            for char in chars:
                results.append(
                    RenderResult(char, font_name, subset_name, False, error_msg)
                )
        return results, missing_chars
    
    work_items = [("train", train_chars), ("test", test_chars)]

    for subset, chars in work_items:
        subset_dir_path = subset_dir(font_name, subset)

        for char in chars:
            output_file = subset_dir_path / f"{char}.png"
            if output_file.exists() and not dry_run:
                results.append(
                    RenderResult(char, font_name, subset, True, "已存在，跳过")
                )
                continue

            if not check_character_renderable(font, char):
                missing_chars.append(char)
                results.append(
                    RenderResult(char, font_name, subset, False, "字形缺失，跳过")
                )
                continue

            try:
                image = render_character_image(font, char, std_metrics)
                save_image(image, output_file, dry_run=dry_run)
                
                results.append(
                    RenderResult(
                        char,
                        font_name,
                        subset,
                        True,
                        f"生成成功",
                    )
                )
            except Exception as exc:
                results.append(
                    RenderResult(char, font_name, subset, False, f"渲染失败: {exc}")
                )

    return results, missing_chars


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="字体字符图像生成脚本")
    parser.add_argument(
        "--fonts",
        nargs="+",
        help="仅处理指定字体（文件名或不带后缀的字体名）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="忽略状态，强制重新生成",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅检查需要生成的内容，不写入文件",
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        default=TRAIN_CHAR_FILE,
        help="训练字符清单文件路径 (默认: train.txt)",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        default=TEST_CHAR_FILE,
        help="测试字符清单文件路径 (默认: test.txt)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=IMAGE_SIZE,
        help="输出图片尺寸（正方形）",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=TARGET_LINEAR_SCALE,
        help="目标线性占比（字符最大边长 / 画布边长）",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=SAMPLE_COUNT,
        help="计算标准 metrics 的样本数量",
    )
    return parser.parse_args(argv)


def apply_config_from_args(args: argparse.Namespace) -> None:
    global IMAGE_SIZE, TARGET_LINEAR_SCALE, SAMPLE_COUNT

    IMAGE_SIZE = max(32, int(args.image_size))
    TARGET_LINEAR_SCALE = max(0.1, min(float(args.scale), 0.95))
    SAMPLE_COUNT = max(1, int(args.sample_count))


def font_selected(font_path: Path, fonts_filter: Optional[Set[str]]) -> bool:
    if not fonts_filter:
        return True

    stem = font_path.stem.lower()
    name = font_path.name.lower()
    return stem in fonts_filter or name in fonts_filter


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    ensure_directories()

    apply_config_from_args(args)

    train_chars = load_characters(args.train_file)
    test_chars = load_characters(args.test_file)

    if not train_chars and not test_chars:
        print("[ERROR] 未找到任何待生成字符，检查 train.txt / test.txt。")
        return 1

    font_files = find_font_files(FONT_DIR)
    if not font_files:
        print("[ERROR] 未在 font/ 目录下找到字体文件。")
        return 1

    font_status = load_font_status()
    missing_map = load_existing_missing()

    processed_any = False

    fonts_filter: Optional[Set[str]] = None
    if args.fonts:
        fonts_filter = {item.lower() for item in args.fonts}

    filtered_fonts = [f for f in font_files if font_selected(f, fonts_filter)]
    if not filtered_fonts:
        print("[WARN] 没有匹配的字体文件。")
        return 0

    font_loop = tqdm(filtered_fonts, desc="字体生成", unit="font")
    for font_path in font_loop:
        font_name = font_path.stem
        if should_skip_font(font_name, font_status, force=args.force):
            tqdm.write(f"[INFO] 字体 {font_name} 已生成，跳过。 (使用 --force 可重新生成)")
            continue

        tqdm.write(f"[INFO] 开始处理字体: {font_name}")
        results, missing_chars = process_font(
            font_path,
            train_chars,
            test_chars,
            dry_run=args.dry_run,
        )

        generated = [r for r in results if r.succeeded]
        failed = [r for r in results if not r.succeeded]

        if args.dry_run:
            for item in generated:
                print(f"  [DRY] {item.subset}: {item.character}")
        else:
            font_status[font_name] = {
                "font_file": font_path.name,
                "generated_at": datetime.now().isoformat(timespec="seconds"),
                "train_count": sum(1 for r in generated if r.subset == "train"),
                "test_count": sum(1 for r in generated if r.subset == "test"),
                "missing_count": len(missing_chars),
                "completed": True,
            }

        if missing_chars:
            missing_map[font_name] = missing_chars
            print(f"[WARN] 字体 {font_name} 缺少 {len(missing_chars)} 个字符，详情见 missing_chars.csv。")
        elif font_name in missing_map:
            del missing_map[font_name]

        processed_any = True

        font_loop.set_postfix(
            {
                "generated": sum(1 for r in generated if r.succeeded),
                "missing": len(missing_chars),
            }
        )

    if processed_any and not args.dry_run:
        save_font_status(font_status)
        update_missing_csv(missing_map)
        print("[INFO] 生成流程完成。")
    else:
        if args.dry_run and processed_any:
            print("[INFO] Dry-run 完成，未写入任何文件。")
        else:
            print("[INFO] 没有发现需要生成的字体。")

    return 0


if __name__ == "__main__":
    sys.exit(main())
