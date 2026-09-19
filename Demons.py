# -*- coding: utf-8 -*-
"""
封装 LogDomainDemonsRegistration + 双向插值，使其可作为模块被批量脚本复用。
默认生成与原脚本一致的 12 帧（000.png~011.png）。
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

np = None  # type: ignore[assignment]
sitk = None  # type: ignore[assignment]
Image = None  # type: ignore[assignment]


def _ensure_modules() -> None:
    """延迟导入依赖，以避免静态分析环境缺失导致的提示。"""
    global np, sitk, Image
    if np is None or sitk is None or Image is None:  # pragma: no cover
        import numpy as _np  # type: ignore[import]
        import SimpleITK as _sitk  # type: ignore[import]
        from PIL import Image as _Image  # type: ignore[import]  # 生成 GIF

        np = _np
        sitk = _sitk
        Image = _Image


_ensure_modules()


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXE = PROJECT_ROOT / "LogDomainDemonsRegistration"


# ===========【第一段：配准】===========
@dataclass
class DemonsRegistrationResult:
    fixed_path: Path
    moving_path: Path
    deformation_field: Path
    inverse_field: Path
    velocity_field: Path
    warped_image: Path


def run_demons_registration(
    fixed_path: Path,
    moving_path: Path,
    work_dir: Path,
    *,
    exe_path: Path = DEFAULT_EXE,
    iters: str = "100x50x20",
    sigma_vel: str = "2",
    sigma_up: str = "2",
    max_step: str = "3",
    update_rule: str = "1",
    grad_type: str = "0",
    bch_terms: str = "3",
    hist_match: bool = False,
    verbose: bool = False,
) -> DemonsRegistrationResult:
    work_dir = Path(work_dir)
    exe_path = Path(exe_path)
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)

    if not exe_path.exists():
        raise FileNotFoundError(f"未找到 LogDomainDemonsRegistration 可执行文件：{exe_path}")
    if not fixed_path.exists():
        raise FileNotFoundError(f"fixed 图像不存在：{fixed_path}")
    if not moving_path.exists():
        raise FileNotFoundError(f"moving 图像不存在：{moving_path}")

    work_dir.mkdir(parents=True, exist_ok=True)

    out_prefix = work_dir / "registered_image"
    out_img = out_prefix.with_suffix(".mha")
    out_def = Path(f"{out_prefix}-deformationField.mha")
    out_inv = Path(f"{out_prefix}-inverseDeformationField.mha")
    out_vel = Path(f"{out_prefix}-velocityField.mha")

    cmd: List[str] = [
        str(exe_path),
        "-f",
        str(fixed_path),
        "-m",
        str(moving_path),
        "-o",
        str(out_img),
        "-i",
        iters,
        "-s",
        sigma_vel,
        "-g",
        sigma_up,
        "-l",
        max_step,
        "-a",
        update_rule,
        "-t",
        grad_type,
        "-c",
        bch_terms,
        "-d",
        "1" if verbose else "0",
        "--outputDef-field",
        str(out_def),
        "--outputInvDef-field",
        str(out_inv),
        "--outputVel-field",
        str(out_vel),
    ]
    if hist_match:
        cmd.append("-e")

    if verbose:
        print("🚀 运行命令：\n", " ".join(cmd))
    t0 = time.time()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if verbose:
        print("=== STDOUT ===\n", proc.stdout)
        print("=== STDERR ===\n", proc.stderr)
        print(f"⏱️ 配准耗时：{time.time() - t0:.2f}s")

    if proc.returncode != 0:
        error_msg = f"LogDomainDemonsRegistration 失败（退出码 {proc.returncode}）：\n{proc.stderr}"
        if verbose:
            print("=== ERROR DETAILS ===")
            print(f"命令: {' '.join(cmd)}")
            print(f"退出码: {proc.returncode}")
            print(f"STDERR: {proc.stderr}")
            print(f"STDOUT: {proc.stdout}")
        raise RuntimeError(error_msg)

    return DemonsRegistrationResult(
        fixed_path=fixed_path,
        moving_path=moving_path,
        deformation_field=out_def,
        inverse_field=out_inv,
        velocity_field=out_vel,
        warped_image=out_img,
    )


# ===========【第二段：双向插值（白边 + 12 张）】===========
def deform_image(image, deformation_field, t_or_u):
    def_array = sitk.GetArrayFromImage(deformation_field)
    current_displacement = def_array * float(t_or_u)
    current_def_field = sitk.GetImageFromArray(current_displacement, isVector=True)
    current_def_field.CopyInformation(deformation_field)
    current_def_field = sitk.Cast(current_def_field, sitk.sitkVectorFloat64)

    transform = sitk.DisplacementFieldTransform(current_def_field)
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(image)
    resampler.SetTransform(transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(255.0)  # 白色补边
    return resampler.Execute(image)


def save_png(image, path):
    img8 = sitk.Cast(sitk.RescaleIntensity(image), sitk.sitkUInt8)
    sitk.WriteImage(img8, str(path))


def bidirectional_interpolation_12only(
    IMAGE_T,
    IMAGE_S,
    DEF_FIELD,
    INV_DEF_FIELD,
    OUTPUT_DIR,
    *,
    verbose: bool = False,
):
    """
    最终只输出 12 张编号帧 000..011：
      - 000..005 = A2B t=0..0.5
      - 006..011 = B2A u=0..0.5，但保存顺序为 u=0.5..0.0（即索引 5..0）
    """
    OUTPUT_DIR = Path(OUTPUT_DIR)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_t = sitk.ReadImage(IMAGE_T, sitk.sitkFloat32)  # moving
    image_s = sitk.ReadImage(IMAGE_S, sitk.sitkFloat32)  # fixed
    def_field = sitk.ReadImage(DEF_FIELD)  # A2B：moving→fixed
    inv_def_field = sitk.ReadImage(INV_DEF_FIELD)  # B2A：fixed→moving

    if verbose:
        print("🔄 开始双向插值（仅生成 12 张最终帧）...")

    # --- A2B：t 取 0..0.5（6点）
    ts_a2b = np.linspace(0.0, 0.5, 6)
    a2b_imgs = []
    for i, t in enumerate(ts_a2b):
        semantic_name = f"from_t_{i:02d}_t{t:.2f}.png"
        if verbose:
            print(f"  A2B  生成 {semantic_name}")
        a2b_imgs.append((semantic_name, deform_image(image_t, def_field, t)))

    # --- B2A：u 取 0..0.5（6点），命名用 u，稍后按 5..0 逆序接到 000..011 的后半段
    us_b2a = np.linspace(0.0, 0.5, 6)
    b2a_imgs = []
    for j, u in enumerate(us_b2a):
        semantic_name = f"from_s_{j:02d}_u{u:.2f}.png"
        if verbose:
            print(f"  B2A  生成 {semantic_name}")
        b2a_imgs.append((semantic_name, deform_image(image_s, inv_def_field, u)))

    # --- 保存 12 张（只保留编号帧），并打印对应语义名映射
    # 000..005 : A2B [0..5]
    for i in range(6):
        out_path = OUTPUT_DIR / f"{i:03d}.png"
        if verbose:
            print(f"→ 保存 {out_path.name}  ←  {a2b_imgs[i][0]}")
        save_png(a2b_imgs[i][1], out_path)

    # 006..011 : B2A [5..0] 逆序（把 u=0.5 接在中点后，使整体向 fixed 靠拢）
    for k, j in enumerate(range(5, -1, -1), start=6):
        out_path = OUTPUT_DIR / f"{k:03d}.png"
        if verbose:
            print(f"→ 保存 {out_path.name}  ←  {b2a_imgs[j][0]} (逆序接入)")
        save_png(b2a_imgs[j][1], out_path)

    if verbose:
        print(f"✅ 已生成 12 帧到：{OUTPUT_DIR}")
    return OUTPUT_DIR


# ===========【第三段：生成 2 秒 GIF】===========
def make_gif_from_frames(frame_dir, out_path="out.gif"):
    frame_dir = Path(frame_dir)
    frames = sorted(frame_dir.glob("*.png"))
    if not frames:
        print("⚠️ 未找到帧图，无法生成 GIF")
        return
    imgs = [Image.open(f).convert("RGB") for f in frames]
    duration = 166  # 每帧 166 ms ≈ 2 秒 / 12 帧
    gif_path = frame_dir / out_path
    imgs[0].save(
        gif_path,
        save_all=True,
        append_images=imgs[1:],
        duration=duration,
        loop=0,
        disposal=2,
    )
    print(f"🎞️ 已生成 2 秒 GIF: {gif_path}")


# ===========【高层封装】===========
def generate_demons_sequence(
    moving_path: Path,
    fixed_path: Path,
    output_dir: Path,
    *,
    exe_path: Path = DEFAULT_EXE,
    work_dir: Path | None = None,
    keep_work_dir: bool = False,
    verbose: bool = False,
) -> List[Path]:
    moving_path = Path(moving_path)
    fixed_path = Path(fixed_path)
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    if work_dir is None:
        work_dir = output_dir.parent / f".demons_tmp_{output_dir.name}"
    work_dir = Path(work_dir)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reg_result = run_demons_registration(
        fixed_path=fixed_path,
        moving_path=moving_path,
        work_dir=work_dir,
        exe_path=exe_path,
        verbose=verbose,
    )

    bidirectional_interpolation_12only(
        IMAGE_T=str(reg_result.moving_path),
        IMAGE_S=str(reg_result.fixed_path),
        DEF_FIELD=str(reg_result.deformation_field),
        INV_DEF_FIELD=str(reg_result.inverse_field),
        OUTPUT_DIR=output_dir,
        verbose=verbose,
    )

    if not keep_work_dir:
        shutil.rmtree(work_dir, ignore_errors=True)

    return sorted(output_dir.glob("*.png"))


# ===========【命令行入口】===========
def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LogDomainDemonsRegistration 辅助脚本")
    parser.add_argument("--moving", required=True, help="moving 图像路径（源）")
    parser.add_argument("--fixed", required=True, help="fixed 图像路径（目标）")
    parser.add_argument("--output", required=True, help="输出目录（生成 12 帧）")
    parser.add_argument("--exe", default=str(DEFAULT_EXE), help="LogDomainDemonsRegistration 路径")
    parser.add_argument("--work-dir", help="中间文件目录（默认输出目录旁临时目录）")
    parser.add_argument("--keep-work-dir", action="store_true", help="保留中间文件")
    parser.add_argument("--verbose", action="store_true", help="打印详细日志")
    parser.add_argument("--gif", action="store_true", help="生成 12 帧后额外输出 2 秒 GIF")
    parser.add_argument(
        "--gif-name",
        default="out.gif",
        help="GIF 文件名（配合 --gif 使用，默认 out.gif）",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    frames = generate_demons_sequence(
        moving_path=Path(args.moving),
        fixed_path=Path(args.fixed),
        output_dir=Path(args.output),
        exe_path=Path(args.exe),
        work_dir=Path(args.work_dir) if args.work_dir else None,
        keep_work_dir=args.keep_work_dir,
        verbose=args.verbose,
    )

    print(f"✅ 已生成 {len(frames)} 张帧图至 {args.output}")

    if args.gif:
        make_gif_from_frames(Path(args.output), out_path=args.gif_name)

    return 0


if __name__ == "__main__":
    main()
