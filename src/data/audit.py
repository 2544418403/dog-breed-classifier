"""
audit.py —— 数据泄漏自检

任务书第 5 条：计算 train 和 val 之间的图像相似度，看有没有重复图。

为什么要做这件事：
    即使你"记得"按实体划分了，也仍然可能漏。典型的漏法：
      - 同一个病人换了 ID 又来拍了一次（ID 不同，但人是同一个人）
      - 网站抓图时同一张图存了两次，文件名不同（md5 不同？不，md5 相同）
      - 数据增强后的图被误存进了原始目录
      - 同一张图被裁成两块，分别进了 train 和 val
      - 你按 patient_id 分组，但没意识到 image_date 才是真正的重复单元

    这些只有**在像素层面比对**才能发现。

本模块提供三层检测，成本从低到高：

    L0  md5           字节级完全相同     O(n) 读盘，最严格但只能查"一模一样"
    L1  感知哈希      dHash/aHash/pHash  近似重复（重编码、缩放、轻度裁切）
                      64 bit 汉明距离，几千张图秒级完成
    L2  CNN 嵌入余弦  ImageNet 预训练特征的语义相似度
                      能查出"同一只狗的不同照片"这种语义级重复
                      —— 这一层才是真正对应"同一个体泄漏"的

判定标准（经验值，可根据任务调整）：
    dHash 汉明距离 <= 6 / 64  -> 高度疑似近重复
    嵌入余弦相似度 >= 0.95    -> 同一实体不同帧（连拍 / 同一病人多次拍片）
    嵌入余弦相似度 >= 0.90    -> 值得人工看一眼
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

from .io_utils import file_md5, load_image_rgb

PathLike = Union[str, Path]


# ==============================================================================
# L0: md5
# ==============================================================================
def exact_duplicate_report(
    df: pd.DataFrame, root: PathLike = ".", path_col: str = "path", split_col: Optional[str] = "split"
) -> pd.DataFrame:
    """字节级重复。返回每个重复组的明细。"""
    root = str(root)
    md5s = []
    for rel in tqdm(df[path_col].astype(str).tolist(), desc="计算 md5", ncols=80):
        p = rel if _is_abs(rel) else f"{root}/{rel}"
        try:
            md5s.append(file_md5(p))
        except OSError:
            md5s.append("")
    d = df.copy()
    d["md5"] = md5s
    dup = d[d["md5"].duplicated(keep=False) & (d["md5"] != "")]
    if len(dup) == 0:
        return pd.DataFrame(columns=[path_col, "md5", split_col or "split"])
    if split_col is not None and split_col in d.columns:
        # 只关心跨 split 的重复
        g = dup.groupby("md5")[split_col].nunique()
        cross = set(g[g > 1].index)
        dup = dup[dup["md5"].isin(cross)]
    return dup.reset_index(drop=True)


# ==============================================================================
# L1: 感知哈希
# ==============================================================================
def dhash(img: Union[Image.Image, np.ndarray], size: int = 8) -> np.ndarray:
    """差值哈希：64 bit。对缩放/亮度变化鲁棒，对内容变化敏感。"""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR), dtype=np.float32)
    else:
        img = np.asarray(img, dtype=np.float32)
    diff = img[:, 1:] > img[:, :-1]
    return diff.flatten()


def ahash(img: Union[Image.Image, np.ndarray], size: int = 8) -> np.ndarray:
    """平均哈希：64 bit。比 dhash 更稳但更粗糙。"""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    m = img.mean()
    return (img > m).flatten()


def phash(img: Union[Image.Image, np.ndarray], size: int = 32, keep: int = 8) -> np.ndarray:
    """DCT 感知哈希：64 bit。对 JPEG 压缩、缩放最鲁棒。"""
    if isinstance(img, Image.Image):
        img = np.asarray(img.convert("L").resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)
    try:
        from scipy.fft import dct

        c = dct(dct(img, axis=0, norm="ortho"), axis=1, norm="ortho")
    except Exception:
        # 纯 numpy 兜底（scipy 缺失时）
        c = np.abs(np.fft.fft2(img))
    low = c[:keep, :keep].flatten()
    med = np.median(low)
    return (low > med).flatten()


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a.astype(np.uint8) != b.astype(np.uint8)))


def hash_matrix(
    df: pd.DataFrame,
    root: PathLike = ".",
    path_col: str = "path",
    hash_fn: str = "dhash",
    show_progress: bool = True,
) -> Tuple[np.ndarray, List[int]]:
    """对 df 里所有图算哈希，返回 (位打包矩阵 uint8 N x 8, 成功下标)。"""
    fn = {"dhash": dhash, "ahash": ahash, "phash": phash}[hash_fn]
    root = str(root)
    rows, keep = [], []
    it = enumerate(df[path_col].astype(str).tolist())
    if show_progress:
        it = tqdm(it, total=len(df), desc=f"计算 {hash_fn}", ncols=80)
    for i, rel in it:
        p = rel if _is_abs(rel) else f"{root}/{rel}"
        try:
            img = load_image_rgb(p)
            h = fn(img)
        except Exception:
            continue
        rows.append(np.packbits(h.astype(np.uint8)))
        keep.append(i)
    return np.asarray(rows, dtype=np.uint8), keep


def hamming_matrix(bits: np.ndarray) -> np.ndarray:
    """N x N 汉明距离矩阵（位异或 + popcount）。N<=5000 时可用，更快请用 faiss。"""
    N = len(bits)
    out = np.zeros((N, N), dtype=np.int16)
    lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
    for i in range(N):
        x = np.bitwise_xor(bits[i : i + 1], bits)
        out[i] = lut[x].sum(axis=1)
    return out


# ==============================================================================
# L2: CNN 嵌入余弦相似度
# ==============================================================================
class EmbeddingExtractor:
    """用 ImageNet 预训练骨干抽全局特征（不做分类，只用特征）。

    默认 resnet50 + avgpool 后的 2048 维。这个特征空间里，
    "同一只狗的两张照片" 的余弦相似度会明显高于 "两只不同品种的狗"。
    """

    def __init__(self, backbone: str = "resnet50", device: Optional[str] = None, batch_size: int = 32):
        import timm

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = timm.create_model(backbone, pretrained=True, num_classes=0)  # num_classes=0 -> 只要特征
        self.model.eval().to(self.device)
        self.batch_size = batch_size
        data_cfg = timm.data.resolve_model_data_config(self.model)
        self.tfm = timm.data.create_transform(**data_cfg, is_training=False)
        self.feat_dim = self.model.num_features

    @torch.no_grad()
    def __call__(self, paths: Sequence[str], show_progress: bool = True) -> np.ndarray:
        out = []
        from PIL import Image

        it = range(0, len(paths), self.batch_size)
        if show_progress:
            it = tqdm(it, desc="抽取嵌入", ncols=80)
        for s in it:
            batch = paths[s : s + self.batch_size]
            xs = []
            for p in batch:
                try:
                    xs.append(self.tfm(load_image_rgb(p)))
                except Exception:
                    xs.append(torch.zeros(3, self.tfm.transforms[0].size[0], self.tfm.transforms[0].size[0]))
            x = torch.stack(xs).to(self.device)
            f = self.model(x).float()
            f = torch.nn.functional.normalize(f, dim=1)
            out.append(f.cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.zeros((0, self.feat_dim))


def pairwise_cosine(emb: np.ndarray) -> np.ndarray:
    """embedding 已 L2 归一化时，余弦相似度 = 点积。"""
    return emb @ emb.T


# ==============================================================================
# 综合报告
# ==============================================================================
@dataclass
class LeakAuditResult:
    n_train: int
    n_val: int
    hash_threshold: int
    cos_threshold: float
    n_hash_pairs: int
    n_cos_pairs: int
    max_cos: float
    mean_max_cos: float          # 每张 val 图在 train 里的最相似图的相似度，取平均
    top_pairs: pd.DataFrame
    hash_dist: Optional[np.ndarray] = None
    cos_sim: Optional[np.ndarray] = None

    def summary_text(self) -> str:
        lines = [
            "=" * 66,
            "数据泄漏自检报告（train vs val）",
            "=" * 66,
            f"train {self.n_train} 张 / val {self.n_val} 张",
            "",
            f"[L1 感知哈希] 汉明距离 <= {self.hash_threshold}/64 的跨集配对数 : {self.n_hash_pairs}",
            f"[L2 CNN 嵌入] 余弦相似度 >= {self.cos_threshold} 的跨集配对数     : {self.n_cos_pairs}",
            f"              全体跨集余弦相似度最大值            : {self.max_cos:.4f}",
            f"              每张 val 图对 train 的最大相似度均值: {self.mean_max_cos:.4f}",
        ]
        if self.n_cos_pairs == 0 and self.n_hash_pairs == 0:
            lines += ["", "结论：未发现明显近重复，划分看起来是干净的。"]
        else:
            lines += [
                "",
                f"结论：发现疑似泄漏。**当前 val_acc 不能代表真实泛化能力。**",
                "处理方式：按这些近重复对所在的实体重新分组划分（splits.group_split）。",
            ]
        lines.append("=" * 66)
        return "\n".join(lines)


def leakage_audit(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    root: PathLike = ".",
    path_col: str = "path",
    label_col: Optional[str] = "label",
    hash_fn: str = "dhash",
    hash_threshold: int = 6,
    cos_threshold: float = 0.90,
    use_embedding: bool = True,
    backbone: str = "resnet50",
    top_k: int = 12,
) -> LeakAuditResult:
    """train / val 之间的一站式泄漏自检。

    返回 LeakAuditResult，其中 top_pairs 是可直接人工复核的最相似配对。
    """
    root = str(root)
    tr_paths = [_join(root, p) for p in train_df[path_col].astype(str)]
    va_paths = [_join(root, p) for p in val_df[path_col].astype(str)]

    # --- L1 ---
    bits_tr, keep_tr = hash_matrix(train_df, root, path_col, hash_fn, show_progress=False)
    bits_va, keep_va = hash_matrix(val_df, root, path_col, hash_fn, show_progress=False)
    n_hash_pairs = 0
    hash_dist = None
    if len(bits_tr) and len(bits_va):
        # 分批算汉明距离，避免 N x M 一次性爆内存
        hash_dist = np.zeros((len(bits_va), len(bits_tr)), dtype=np.int16)
        lut = np.array([bin(i).count("1") for i in range(256)], dtype=np.int16)
        for i in range(len(bits_va)):
            hash_dist[i] = lut[np.bitwise_xor(bits_va[i : i + 1], bits_tr)].sum(axis=1)
        n_hash_pairs = int((hash_dist <= hash_threshold).sum())

    # --- L2 ---
    n_cos_pairs = 0
    max_cos = float("nan")
    mean_max_cos = float("nan")
    cos_sim = None
    top_pairs = pd.DataFrame()
    if use_embedding:
        ex = EmbeddingExtractor(backbone)
        emb_tr = ex(tr_paths)
        emb_va = ex(va_paths)
        cos_sim = pairwise_cosine(emb_tr)  # tr x tr (占位)
        cos_sim = emb_va @ emb_tr.T        # va x tr
        per_val_max = cos_sim.max(axis=1) if cos_sim.size else np.zeros(0)
        mean_max_cos = float(per_val_max.mean()) if per_val_max.size else float("nan")
        max_cos = float(cos_sim.max()) if cos_sim.size else float("nan")
        n_cos_pairs = int((cos_sim >= cos_threshold).sum())

        # top-k 最相似配对
        if cos_sim.size:
            flat = np.argsort(cos_sim.ravel())[::-1][:top_k]
            rows = []
            for f in flat:
                j, i = divmod(int(f), cos_sim.shape[1])
                rows.append(
                    {
                        "val_path": val_df[path_col].astype(str).iloc[j],
                        "train_path": train_df[path_col].astype(str).iloc[i],
                        "cosine": float(cos_sim[j, i]),
                        "val_label": (val_df[label_col].iloc[j] if label_col in val_df.columns else None),
                        "train_label": (train_df[label_col].iloc[i] if label_col in train_df.columns else None),
                    }
                )
            top_pairs = pd.DataFrame(rows)

    return LeakAuditResult(
        n_train=len(train_df),
        n_val=len(val_df),
        hash_threshold=hash_threshold,
        cos_threshold=cos_threshold,
        n_hash_pairs=n_hash_pairs,
        n_cos_pairs=n_cos_pairs,
        max_cos=max_cos,
        mean_max_cos=mean_max_cos,
        top_pairs=top_pairs,
        hash_dist=hash_dist,
        cos_sim=cos_sim,
    )


def _join(root: str, p: str) -> str:
    return p if _is_abs(p) else f"{root}/{p}"


def _is_abs(p: str) -> bool:
    return len(p) > 1 and (p[1] == ":" or p.startswith("/") or p.startswith("\\"))
