"""
LSNet 画风批量聚类脚本（可独立运行）。

使用方式：
1) 修改本文件顶部的“配置区”全局变量（推荐先保持 DRY_RUN=True）
2) 运行：
   python style_cluster_script.py

输出（位于 OUTPUT_DIR）：
- features.npy / image_names.txt：特征矩阵与文件名列表
- summary.json：分组结果与统计信息
- scatter.png：二维可视化（如果启用）
- clusters/cluster_XXX/*：按分组归档后的文件（如果不是 DRY_RUN）

可选：支持用环境变量覆盖少量关键配置（仍以中文为默认文档语言）：
- LSNET_STYLE_INPUT_DIR：覆盖 INPUT_DIR
- LSNET_STYLE_OUTPUT_DIR：覆盖 OUTPUT_DIR
- LSNET_STYLE_MODEL_DIR：覆盖 MODEL_DIR
"""

from __future__ import annotations

import json
import os
import shutil
import hashlib
from datetime import datetime
from decimal import Decimal, InvalidOperation
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import matplotlib


# -----------------------------
# 配置区（编辑这里）
# -----------------------------

# 要扫描的图片目录
INPUT_DIR = r""

# 输出目录（特征/摘要/可视化 + 分组归档）
OUTPUT_DIR = r"output_style_clusters"

# 模型目录，需包含：
# - best_checkpoint.pth
# - class_mapping.csv
# - （可选）config.json（形如 {"model": "..."}，用于选择模型变体）
#
# 需求默认值： ".\Kaloscope-2.0"（相对本脚本路径）。
MODEL_DIR = r".\Kaloscope-2.0"

# 推理设备："cuda" 或 "cpu"
DEVICE = "cuda"

# 扫描参数
RECURSIVE_SCAN = True
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff")

# 特征提取
BATCH_SIZE = 16
NUM_WORKERS = 0  # 预留（保持最小实现；未使用 dataloader）

# 可选：先对特征做 PCA 降维再计算相似度（更快/更稳，代价是信息损失）
# 设为 0 表示关闭。常用：50。
PCA_REDUCE_DIM = 0

# 分组（两两余弦相似度 + 阈值 + 并查集）
# 阈值不再手工指定：脚本将通过“阈值扫参”自动选择最终使用的阈值。

# 可选：为了避免 O(N^2) 全比较造成“链式效应”/过度连通，可限制每个点只与 top_k 相似对象连边。
# 设为 0 表示关闭（全比较）。
TOP_K_EDGES = 0

# 余弦相似度计算的分块大小（越大越快但更占内存）
SIM_CHUNK_SIZE = 512

# 阈值扫参与自动选择（必需）
THRESH_SWEEP_ENABLE = True
THRESH_SWEEP_MIN = "0.92"
THRESH_SWEEP_MAX = "0.98"
THRESH_SWEEP_POINTS = 30  # 均匀取点
EXPECTED_N_CLUSTERS = 13  # 期望簇数（用于从扫参结果中挑选阈值）

# 输出组织
# - 按 INPUT_DIR 派生 task_id
# - 每次运行创建 runs/<timestamp>/ 子目录，避免覆盖
RUN_TIMESTAMP = ""  # 留空则自动生成，例如 20260408_231500

# 精灵图（Sprite Sheet / Contact Sheet）
# - 每个簇输出一张精灵图：OUTPUT_DIR/sprites/cluster_XXX.png
# - 全量输出：当单簇图片过多时，会自动分页多张精灵图输出
SPRITE_ENABLE = True
SPRITE_THUMB_SIZE = (192, 192)  # (宽, 高)
SPRITE_COLS = 12
SPRITE_MARGIN = 6
# 每张精灵图最多包含多少张图片；超过会分页输出 cluster_XXX_p001.png / p002.png ...
SPRITE_MAX_IMAGES_PER_SHEET = 240

# 可视化
ENABLE_VIZ = True
VIZ_METHOD = "tsne"  # "pca" or "tsne"
TSNE_PERPLEXITY = 30
TSNE_RANDOM_STATE = 42
PLOT_DPI = 200

# 文件操作
DRY_RUN = True  # 建议第一次先 True（只打印计划移动，不实际移动/复制）
COPY_MODE = True  # True=复制；False=移动（DRY_RUN=True 时忽略）
COLLISION_POLICY = "rename"  # "rename"=自动重命名；"skip"=跳过


# -----------------------------
# Implementation
# -----------------------------

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.models import create_model

from lsnet_model import lsnet_artist  # noqa: F401  (register timm models)

from inference_artist import (
    load_checkpoint_state,
    load_class_mapping,
    normalize_state_dict_keys,
    resolve_feature_dim,
    resolve_num_classes,
)


@dataclass(frozen=True)
class ModelBundle:
    model: torch.nn.Module
    transform: object
    device: str


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _cfg_value(key: str, default: str) -> str:
    v = os.environ.get(key)
    return v if v is not None and str(v).strip() != "" else default


def _parse_threshold(value) -> Tuple[float, str]:
    """
    解析相似度阈值，支持 float 或高精度小数字符串。

    返回 (float_value, raw_string)：
    - float_value 用于数值比较（sim >= threshold）
    - raw_string 用于写入 summary，保留用户输入精度
    """
    if isinstance(value, (int, float, np.floating)):
        f = float(value)
        return f, repr(value)
    s = str(value).strip()
    try:
        d = Decimal(s)
    except (InvalidOperation, ValueError):
        raise ValueError(f"阈值无法解析：{value!r}")
    return float(d), s


def _resolve_model_dir(model_dir: str) -> Path:
    p = Path(model_dir)
    if not p.is_absolute():
        p = (_script_dir() / p).resolve()
    return p


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _task_id_for_input_dir(input_dir: Path) -> str:
    # basename + short hash，既可读又稳定
    base = input_dir.name if input_dir.name else "input"
    h = _sha1_text(str(input_dir))[:10]
    safe = "".join([c if c.isalnum() or c in ("-", "_") else "_" for c in base])[:40].strip("_") or "input"
    return f"{safe}_{h}"


def _default_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _file_signature(paths: Sequence[Path]) -> dict:
    """
    计算输入文件清单签名（最小实现）：
    - 使用绝对路径 + size + mtime_ns
    """
    items = []
    for p in paths:
        st = p.stat()
        items.append({"path": str(p), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)})
    payload = json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    return {"count": len(items), "sha1": _sha1_text(payload)}


def _checkpoint_signature(model_dir: Path) -> dict:
    ckpt = model_dir / "best_checkpoint.pth"
    st = ckpt.stat()
    return {"path": str(ckpt), "size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _cache_key(
    input_dir: Path,
    image_sig: dict,
    model_sig: dict,
    pca_reduce_dim: int,
    model_dir: Path,
) -> str:
    payload = json.dumps(
        {
            "input_dir": str(input_dir),
            "image_sig": image_sig,
            "model_sig": model_sig,
            "pca_reduce_dim": int(pca_reduce_dim),
            "model_dir": str(model_dir),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha1_text(payload)


def discover_images(input_dir: str, recursive: bool, exts: Sequence[str]) -> List[Path]:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"INPUT_DIR not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"INPUT_DIR is not a directory: {root}")

    exts_l = tuple(e.lower() for e in exts)
    if recursive:
        paths = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts_l]
    else:
        paths = [p for p in root.iterdir() if p.is_file() and p.suffix.lower() in exts_l]

    paths.sort(key=lambda p: str(p).lower())
    return paths


def load_lsnet_bundle(model_dir: str, device: str) -> ModelBundle:
    model_root = _resolve_model_dir(model_dir)
    checkpoint_path = model_root / "best_checkpoint.pth"
    csv_path = model_root / "class_mapping.csv"
    config_path = model_root / "config.json"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"Class mapping CSV not found: {csv_path}")

    class_mapping = load_class_mapping(str(csv_path))
    state_dict = load_checkpoint_state(str(checkpoint_path))
    state_dict = normalize_state_dict_keys(state_dict)

    num_classes = resolve_num_classes(None, class_mapping, state_dict)
    feature_dim = resolve_feature_dim(None, state_dict)

    model_type = "lsnet_xl_artist"
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if isinstance(cfg, dict) and cfg.get("model"):
                model_type = str(cfg["model"])
        except Exception:
            pass

    model = create_model(
        model_type,
        pretrained=False,
        num_classes=num_classes,
        feature_dim=feature_dim,
    )
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    from lsnet_model.lsnet_artist import default_cfgs_artist

    input_size = 224
    if model_type in default_cfgs_artist:
        model_cfg = default_cfgs_artist[model_type]
        input_size = model_cfg.get("input_size", (3, 224, 224))[1]

    config = resolve_data_config({"input_size": (3, input_size, input_size)}, model=model)
    transform = create_transform(**config)

    return ModelBundle(model=model, transform=transform, device=device)


def _pil_from_path(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def _make_sprite_sheet(
    image_paths: Sequence[Path],
    out_path: Path,
    thumb_size: Tuple[int, int],
    cols: int,
    margin: int,
) -> None:
    """
    将一组图片拼成精灵图（contact sheet）。
    """
    if not image_paths:
        return

    cols = max(1, int(cols))
    tw, th = int(thumb_size[0]), int(thumb_size[1])
    n = len(image_paths)
    rows = (n + cols - 1) // cols

    sheet_w = cols * tw + (cols + 1) * margin
    sheet_h = rows * th + (rows + 1) * margin

    canvas = Image.new("RGB", (sheet_w, sheet_h), color=(18, 18, 18))

    for idx, p in enumerate(image_paths):
        try:
            img = _pil_from_path(p)
            img.thumbnail((tw, th), Image.BICUBIC)
            x = margin + (idx % cols) * (tw + margin) + (tw - img.size[0]) // 2
            y = margin + (idx // cols) * (th + margin) + (th - img.size[1]) // 2
            canvas.paste(img, (x, y))
        except Exception:
            # 跳过坏图，不中断整体输出
            continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def _group_indices_by_label(labels: np.ndarray) -> dict:
    groups: dict[int, list[int]] = {}
    for i, lab in enumerate(labels.astype(int).tolist()):
        groups.setdefault(int(lab), []).append(i)
    return groups


def extract_features(
    bundle: ModelBundle,
    image_paths: Sequence[Path],
    batch_size: int,
) -> np.ndarray:
    feats: List[np.ndarray] = []
    model = bundle.model
    transform = bundle.transform
    device = bundle.device

    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            tensors = []
            for p in batch_paths:
                img = _pil_from_path(p)
                t = transform(img).unsqueeze(0)
                tensors.append(t)

            batch = torch.cat(tensors, dim=0).to(device)
            f = model(batch, return_features=True)
            feats.append(f.detach().cpu().numpy())

            print(f"[features] {min(i + batch_size, len(image_paths))}/{len(image_paths)}")

    return np.concatenate(feats, axis=0)


def l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def maybe_reduce_dim(x: np.ndarray, dim: int) -> Tuple[np.ndarray, Optional[object]]:
    if dim is None or dim <= 0 or dim >= x.shape[1]:
        return x, None
    pca = PCA(n_components=dim, random_state=42)
    return pca.fit_transform(x), pca


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.size = [1] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> bool:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]
        return True


def build_clusters_by_threshold(
    x_norm: np.ndarray,
    threshold: float,
    top_k_edges: int = 0,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, dict]:
    """
    使用“余弦相似度阈值 + 并查集”分组。

    输入 x_norm 必须已做 L2 归一化（每行范数=1）。
    余弦相似度可直接用点积：sim(i,j)=x[i]·x[j]
    """
    n = x_norm.shape[0]
    uf = UnionFind(n)

    edges = 0
    merges = 0

    # 分块计算相似度，避免一次性构建 NxN 矩阵
    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        block = x_norm[i0:i1] @ x_norm.T  # (bi, n)

        for bi in range(i1 - i0):
            i = i0 + bi
            sims = block[bi]
            sims[i] = -1.0  # 排除自身

            if top_k_edges and top_k_edges > 0:
                # 只连接 top-k 且超过阈值的边
                k = min(top_k_edges, n - 1)
                idx = np.argpartition(-sims, kth=k - 1)[:k]
                for j in idx.tolist():
                    if j <= i:
                        continue
                    if float(sims[j]) >= threshold:
                        edges += 1
                        if uf.union(i, j):
                            merges += 1
            else:
                # 全比较：只处理 j>i，避免重复
                js = np.where(sims >= threshold)[0]
                for j in js.tolist():
                    if j <= i:
                        continue
                    edges += 1
                    if uf.union(i, j):
                        merges += 1

    # 根 -> 连续 cluster_id
    roots = [uf.find(i) for i in range(n)]
    uniq = {}
    labels = np.empty(n, dtype=int)
    next_id = 0
    for i, r in enumerate(roots):
        if r not in uniq:
            uniq[r] = next_id
            next_id += 1
        labels[i] = uniq[r]

    # 簇大小统计
    sizes = {}
    for lab in labels.tolist():
        sizes[lab] = sizes.get(lab, 0) + 1

    meta = {
        "threshold": float(threshold),
        "top_k_edges": int(top_k_edges),
        "edges_considered": int(edges),
        "merges": int(merges),
        "n_clusters": int(next_id),
        "cluster_sizes": dict(sorted(sizes.items(), key=lambda kv: (-kv[1], kv[0]))),
    }
    return labels, meta


def count_clusters_by_threshold(
    x_norm: np.ndarray,
    threshold: float,
    top_k_edges: int = 0,
    chunk_size: int = 512,
) -> Tuple[int, dict]:
    """
    仅计算簇数（用于扫参），避免构建 labels。
    """
    n = x_norm.shape[0]
    uf = UnionFind(n)
    edges = 0
    merges = 0

    for i0 in range(0, n, chunk_size):
        i1 = min(n, i0 + chunk_size)
        block = x_norm[i0:i1] @ x_norm.T
        for bi in range(i1 - i0):
            i = i0 + bi
            sims = block[bi]
            sims[i] = -1.0

            if top_k_edges and top_k_edges > 0:
                k = min(top_k_edges, n - 1)
                idx = np.argpartition(-sims, kth=k - 1)[:k]
                for j in idx.tolist():
                    if j <= i:
                        continue
                    if float(sims[j]) >= threshold:
                        edges += 1
                        if uf.union(i, j):
                            merges += 1
            else:
                js = np.where(sims >= threshold)[0]
                for j in js.tolist():
                    if j <= i:
                        continue
                    edges += 1
                    if uf.union(i, j):
                        merges += 1

    roots = set(uf.find(i) for i in range(n))
    meta = {"edges_considered": int(edges), "merges": int(merges)}
    return int(len(roots)), meta


def sweep_thresholds(
    x_norm: np.ndarray,
    sweep_min: str,
    sweep_max: str,
    points: int,
    expected_n_clusters: int,
    top_k_edges: int,
    chunk_size: int,
) -> Tuple[float, str, list, str]:
    """
    扫参并选择最接近期望簇数的阈值。

    返回：
    - chosen_threshold_float
    - chosen_threshold_raw
    - records: [{threshold, n_clusters}, ...]
    - plot_path_suffix（用于 run 目录内保存）
    """
    tmin_f, tmin_raw = _parse_threshold(sweep_min)
    tmax_f, tmax_raw = _parse_threshold(sweep_max)
    if tmax_f < tmin_f:
        tmin_f, tmax_f = tmax_f, tmin_f
        tmin_raw, tmax_raw = tmax_raw, tmin_raw

    pts = max(2, int(points))
    ts = np.linspace(tmin_f, tmax_f, pts, dtype=np.float64)

    records = []
    best_le = None  # (n_clusters, threshold)
    best_any = None  # fallback: minimal diff, then stricter (higher threshold)
    for t in ts.tolist():
        n_clusters, _ = count_clusters_by_threshold(x_norm, float(t), top_k_edges=top_k_edges, chunk_size=chunk_size)
        records.append({"threshold": float(t), "n_clusters": int(n_clusters)})
        # 选择规则：
        # - 优先选择 n_clusters <= expected 的结果里，n_clusters 最大的那个（最接近期望且不超过）
        # - 若并列 n_clusters，相同簇数下选更严格（阈值更高）
        if int(n_clusters) <= int(expected_n_clusters):
            cand = (int(n_clusters), float(t))
            if best_le is None or cand[0] > best_le[0] or (cand[0] == best_le[0] and cand[1] > best_le[1]):
                best_le = cand

        # 兜底：若没有 <= expected 的点，选择“最小 diff”，并列选更严格（阈值更高）
        diff = abs(int(n_clusters) - int(expected_n_clusters))
        cand_any = (diff, -float(t), int(n_clusters))
        if best_any is None or cand_any < best_any[0]:
            best_any = (cand_any, float(t), int(n_clusters))

    if best_le is not None:
        chosen_f = float(best_le[1])
        chosen_n = int(best_le[0])
        choice_reason = "chosen_max_n_clusters_le_expected"
    else:
        assert best_any is not None
        chosen_f = float(best_any[1])
        chosen_n = int(best_any[2])
        choice_reason = "fallback_min_diff"

    chosen_raw = f"{chosen_f:.12f}".rstrip("0").rstrip(".")
    # 把选择信息也塞进 records 末尾的 meta（写入 summary 时会单独记录）
    records_meta = {"choice_reason": choice_reason, "chosen_n_clusters": chosen_n, "expected_n_clusters": int(expected_n_clusters)}
    return chosen_f, chosen_raw, records + [{"_meta": records_meta}], f"threshold_sweep.png"


def project_2d(x: np.ndarray) -> np.ndarray:
    if VIZ_METHOD == "tsne":
        perplexity = min(TSNE_PERPLEXITY, max(2, (len(x) - 1) // 3))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=TSNE_RANDOM_STATE,
            init="pca",
            learning_rate="auto",
        )
        return tsne.fit_transform(x)
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(x)


def save_scatter_png(points_2d: np.ndarray, labels: np.ndarray, out_path: Path) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 尽量选择支持中文的字体（若系统无对应字体，将回退并可能出现缺字警告）
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    unique = sorted(set(int(x) for x in labels.tolist()))
    # Ensure noise last
    if -1 in unique:
        unique = [u for u in unique if u != -1] + [-1]

    plt.figure(figsize=(10, 8))
    for lab in unique:
        mask = labels == lab
        if lab == -1:
            plt.scatter(points_2d[mask, 0], points_2d[mask, 1], c="gray", s=10, alpha=0.6, marker="x", label="noise")
        else:
            plt.scatter(points_2d[mask, 0], points_2d[mask, 1], s=14, alpha=0.75, label=f"cluster_{lab}")
    plt.title(f"画风聚类（{VIZ_METHOD}）")
    plt.legend(markerscale=1.5, fontsize=8, ncols=2)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def save_threshold_sweep_png(records: list, out_path: Path, expected_n_clusters: int) -> None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    xs = [float(r["threshold"]) for r in records]
    ys = [int(r["n_clusters"]) for r in records]

    plt.figure(figsize=(10, 4.5))
    plt.plot(xs, ys, marker="o", linewidth=1.5)
    # 为每个点标注自身数值（只显示簇数）
    for x, y in zip(xs, ys):
        label = f"{y}"
        plt.annotate(
            label,
            (x, y),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=8,
            alpha=0.9,
        )
    plt.axhline(y=int(expected_n_clusters), linestyle="--", linewidth=1, color="orange", label=f"期望簇数={expected_n_clusters}")
    plt.xlabel("阈值（SIMILARITY_THRESHOLD）")
    plt.ylabel("簇数（n_clusters）")
    plt.title("阈值扫参：阈值 vs 簇数")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=PLOT_DPI)
    plt.close()


def _safe_dest_path(dest_dir: Path, filename: str) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    if COLLISION_POLICY == "skip":
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for k in range(1, 10_000):
        c = dest_dir / f"{stem}_{k}{suffix}"
        if not c.exists():
            return c
    raise RuntimeError(f"Too many collisions for {candidate}")


def _transfer_file(src: Path, dst: Path) -> None:
    if DRY_RUN:
        return
    if COPY_MODE:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def organize_files(
    image_paths: Sequence[Path],
    labels: np.ndarray,
    out_root: Path,
) -> None:
    clusters_root = out_root / "clusters"
    for img_path, lab in zip(image_paths, labels.tolist()):
        if lab == -1:
            dest_dir = clusters_root / "noise"
        else:
            dest_dir = clusters_root / f"cluster_{lab:03d}"

        dest_img = _safe_dest_path(dest_dir, img_path.name)
        if dest_img.exists() and COLLISION_POLICY == "skip":
            print(f"[skip] {img_path} -> {dest_img}")
            continue

        print(f"[file] {img_path} -> {dest_img}" + (" (dry-run)" if DRY_RUN else ""))
        _transfer_file(img_path, dest_img)

        sidecar = img_path.with_suffix(".txt")
        if sidecar.exists():
            dest_txt = dest_img.with_suffix(".txt")
            # If collision-renamed image, keep txt consistent with new basename
            dest_txt = _safe_dest_path(dest_dir, dest_txt.name)
            if dest_txt.exists() and COLLISION_POLICY == "skip":
                print(f"[skip] {sidecar} -> {dest_txt}")
                continue
            print(f"[sidecar] {sidecar} -> {dest_txt}" + (" (dry-run)" if DRY_RUN else ""))
            _transfer_file(sidecar, dest_txt)


def main() -> None:
    input_dir = _cfg_value("LSNET_STYLE_INPUT_DIR", INPUT_DIR)
    output_dir = _cfg_value("LSNET_STYLE_OUTPUT_DIR", OUTPUT_DIR)
    model_dir = _cfg_value("LSNET_STYLE_MODEL_DIR", MODEL_DIR)

    if not input_dir:
        raise ValueError("请在脚本顶部配置 INPUT_DIR，或设置环境变量 LSNET_STYLE_INPUT_DIR。")

    device = DEVICE
    if device.lower() == "cuda" and not torch.cuda.is_available():
        print("检测到当前 PyTorch 不支持 CUDA 或未启用 CUDA，已自动回退到 CPU。")
        device = "cpu"

    input_dir_p = Path(input_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    task_id = _task_id_for_input_dir(input_dir_p)
    task_dir = output_root / "tasks" / task_id
    cache_dir = task_dir / "cache"
    runs_dir = task_dir / "runs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    ts = RUN_TIMESTAMP.strip() if isinstance(RUN_TIMESTAMP, str) else ""
    if not ts:
        ts = _default_timestamp()
    run_dir = runs_dir / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    image_paths = discover_images(str(input_dir_p), RECURSIVE_SCAN, IMAGE_EXTS)
    if not image_paths:
        print("未找到图片文件。")
        return
    print(f"找到 {len(image_paths)} 张图片。")

    model_dir_p = _resolve_model_dir(model_dir)
    image_sig = _file_signature(image_paths)
    model_sig = _checkpoint_signature(model_dir_p)
    cache_key = _cache_key(input_dir_p, image_sig, model_sig, PCA_REDUCE_DIM, model_dir_p)
    cache_manifest_path = cache_dir / "manifest.json"
    cache_features_path = cache_dir / "features.npy"
    cache_names_path = cache_dir / "image_names.txt"

    cache_hit = False
    if cache_manifest_path.exists() and cache_features_path.exists() and cache_names_path.exists():
        try:
            m = json.loads(cache_manifest_path.read_text(encoding="utf-8"))
            cache_hit = m.get("cache_key") == cache_key
        except Exception:
            cache_hit = False

    if cache_hit:
        print("命中缓存：复用历史特征提取结果。")
        features = np.load(cache_features_path)
        # 仍然使用本次扫描的 image_paths（用于输出/移动），但它们应与缓存一致
    else:
        print("未命中缓存：开始提取特征（首次或输入/模型发生变化）。")
        bundle = load_lsnet_bundle(str(model_dir_p), device)
        features = extract_features(bundle, image_paths, batch_size=BATCH_SIZE)
        np.save(cache_features_path, features)
        cache_names_path.write_text("\n".join([p.name for p in image_paths]), encoding="utf-8")
        cache_manifest_path.write_text(
            json.dumps(
                {
                    "cache_key": cache_key,
                    "input_dir": str(input_dir_p),
                    "model_dir": str(model_dir_p),
                    "image_sig": image_sig,
                    "model_sig": model_sig,
                    "pca_reduce_dim": int(PCA_REDUCE_DIM),
                    "created_at": ts,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    # 将本次 run 的中间结果也落盘（便于对比不同 run）
    np.save(run_dir / "features.npy", features)
    (run_dir / "image_names.txt").write_text("\n".join([p.name for p in image_paths]), encoding="utf-8")

    features_n = l2_normalize_rows(features)
    feats_for_cluster, _ = maybe_reduce_dim(features_n, PCA_REDUCE_DIM)

    sweep_records = None
    chosen_from_sweep = None
    if not THRESH_SWEEP_ENABLE:
        raise ValueError("当前版本已移除手动 SIMILARITY_THRESHOLD，必须启用 THRESH_SWEEP_ENABLE 扫参选择阈值。")

    chosen_f, chosen_raw, records, sweep_png_name = sweep_thresholds(
        x_norm=l2_normalize_rows(feats_for_cluster),
        sweep_min=THRESH_SWEEP_MIN,
        sweep_max=THRESH_SWEEP_MAX,
        points=THRESH_SWEEP_POINTS,
        expected_n_clusters=EXPECTED_N_CLUSTERS,
        top_k_edges=TOP_K_EDGES,
        chunk_size=SIM_CHUNK_SIZE,
    )
    sweep_records = records
    chosen_from_sweep = {"chosen_threshold": chosen_f, "chosen_threshold_raw": chosen_raw}
    save_threshold_sweep_png([r for r in records if "threshold" in r], run_dir / sweep_png_name, EXPECTED_N_CLUSTERS)
    threshold_f, threshold_raw = chosen_f, chosen_raw
    print(f"扫参选择阈值：{threshold_raw}（期望簇数={EXPECTED_N_CLUSTERS}，规则：优先 ≤ 期望簇数）")
    labels, cluster_meta = build_clusters_by_threshold(
        x_norm=l2_normalize_rows(feats_for_cluster),
        threshold=threshold_f,
        top_k_edges=TOP_K_EDGES,
        chunk_size=SIM_CHUNK_SIZE,
    )
    n_clusters = int(cluster_meta["n_clusters"])

    coords_2d = None
    if ENABLE_VIZ:
        coords_2d = project_2d(feats_for_cluster)
        save_scatter_png(coords_2d, labels, run_dir / "scatter.png")

    summary = {
        "config": {
            "input_dir": str(input_dir_p),
            "output_root": str(output_root),
            "task_id": task_id,
            "task_dir": str(task_dir),
            "run_dir": str(run_dir),
            "model_dir": str(model_dir_p),
            "device": device,
            "recursive_scan": RECURSIVE_SCAN,
            "image_exts": list(IMAGE_EXTS),
            "batch_size": BATCH_SIZE,
            "pca_reduce_dim": PCA_REDUCE_DIM,
            "similarity_threshold": float(threshold_f),
            "similarity_threshold_raw": threshold_raw,
            "top_k_edges": int(TOP_K_EDGES),
            "threshold_sweep_enable": bool(THRESH_SWEEP_ENABLE),
            "threshold_sweep_min": str(THRESH_SWEEP_MIN),
            "threshold_sweep_max": str(THRESH_SWEEP_MAX),
            "threshold_sweep_points": int(THRESH_SWEEP_POINTS),
            "expected_n_clusters": int(EXPECTED_N_CLUSTERS),
            "sprite_enable": bool(SPRITE_ENABLE),
            "sprite_thumb_size": list(SPRITE_THUMB_SIZE),
            "sprite_cols": int(SPRITE_COLS),
            "sprite_margin": int(SPRITE_MARGIN),
            "sprite_max_images_per_sheet": int(SPRITE_MAX_IMAGES_PER_SHEET),
            "enable_viz": ENABLE_VIZ,
            "viz_method": VIZ_METHOD,
            "dry_run": DRY_RUN,
            "copy_mode": COPY_MODE,
            "collision_policy": COLLISION_POLICY,
        },
        "stats": {
            "n_images": len(image_paths),
            "n_clusters": n_clusters,
            "edges_considered": int(cluster_meta.get("edges_considered", 0)),
            "merges": int(cluster_meta.get("merges", 0)),
            "cluster_sizes": cluster_meta.get("cluster_sizes", {}),
            "cache_hit": bool(cache_hit),
            "cache_key": cache_key,
        },
        "items": [],
    }
    summary["cluster_meta"] = cluster_meta
    if sweep_records is not None:
        summary["threshold_sweep"] = sweep_records
    if chosen_from_sweep is not None:
        summary["threshold_choice"] = chosen_from_sweep

    groups = _group_indices_by_label(labels)
    # 更直观的“分了哪几类”
    classes = []
    for lab, idxs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        classes.append(
            {
                "cluster_id": int(lab),
                "size": int(len(idxs)),
                "files": [image_paths[i].name for i in idxs],
            }
        )
    summary["classes"] = classes

    for idx, (p, lab) in enumerate(zip(image_paths, labels.tolist())):
        item = {"index": idx, "filename": p.name, "path": str(p), "cluster_id": int(lab)}
        if coords_2d is not None:
            item["x"] = float(coords_2d[idx, 0])
            item["y"] = float(coords_2d[idx, 1])
        summary["items"].append(item)

    if SPRITE_ENABLE:
        sprites = []
        for lab, idxs in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            per_sheet = max(1, int(SPRITE_MAX_IMAGES_PER_SHEET))
            pages = (len(idxs) + per_sheet - 1) // per_sheet
            for page in range(pages):
                take = idxs[page * per_sheet : (page + 1) * per_sheet]
                suffix = f"_p{page+1:03d}" if pages > 1 else ""
                sprite_path = run_dir / "sprites" / f"cluster_{int(lab):03d}{suffix}.png"
                _make_sprite_sheet(
                    image_paths=[image_paths[i] for i in take],
                    out_path=sprite_path,
                    thumb_size=tuple(SPRITE_THUMB_SIZE),
                    cols=int(SPRITE_COLS),
                    margin=int(SPRITE_MARGIN),
                )
                sprites.append(
                    {
                        "cluster_id": int(lab),
                        "page": int(page + 1),
                        "pages": int(pages),
                        "sprite_path": str(sprite_path),
                        "n_images_used": int(len(take)),
                    }
                )
        summary["sprites"] = sprites

    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"分组簇数：{n_clusters}，阈值：{threshold_raw}，合并次数：{cluster_meta.get('merges', 0)}")

    organize_files(image_paths, labels, run_dir)


if __name__ == "__main__":
    main()

