"""
RanSMAP 数据模块（同域学习 + 异域测试）
================================================

功能：
- 数据选择：从指定域（如 original/celeron-gen6/ddr4-1600-8g）中收集 trace 目录，
  可限制每类最多采样多少个 trace 以控制规模；可附加 OOD 测试域（mix/、variants/ 等）。
- 数据划分：按 trace/类别做分层 80/10/10（train/val/test），避免同一 trace 泄漏到多个集合。
- 数据处理：将 6 条事件流（mem_* 与 ata_*）按统一时间窗聚合为固定维度特征；
  再切成定长序列以供 LSTM/GRU/TCN 使用。
- 缓存：将每个 trace 的“窗口特征矩阵”缓存到磁盘（.npz），避免反复 CSV->特征计算；
  训练仅在缓存上进行。保存 scaler（均值/方差）与 manifest（样本清单）。
- 输出：返回 PyTorch 的 train/val/test DataLoader。

优化要点：
- 面向机械硬盘：顺序构建缓存；训练阶段使用较少的 num_workers（建议 2~4），
  persistent_workers=True，prefetch_factor=2，避免高并发随机小文件 IO。
- 时间/空间：先缓存“窗口级矩阵”，__getitem__ 时按需切片成序列；避免将所有序列展开到磁盘。

使用示例：
---------
from data import build_dataloaders
loaders, info = build_dataloaders(
    dataset_root="/path/to/RanSMAP",
    cache_root="/path/to/cache",
    train_domain="original/celeron-gen6/ddr4-1600-8g",
    ood_domains=["mix/celeron-gen6/ddr4-1600-8g", "variants/celeron-gen6/ddr4-1600-8g"],
    max_traces_per_category=10,  # 小规模训练
    window_ms=100, seq_len=256, stride=64,
    batch_size=32, num_workers=2,
)
train_loader, val_loader, test_loader = loaders

"""

from __future__ import annotations
import os
import json
import math
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

try:
    from tqdm import tqdm
except Exception:  # tqdm 可选
    def tqdm(x, *args, **kwargs):
        return x

# ----------------------- 配置常量 -----------------------
# 已知类别与良性集合（用于生成 is_ransomware 标签）
ALL_CATEGORIES = [
    'AESCrypt','Conti','Darkside','Firefox','Idle','LockBit',
    'Office','REvil','Ryuk','SDelete','WannaCry','Zip'
]
BENIGN_CATEGORIES = {'AESCrypt','Zip','SDelete','Firefox','Office','Idle'}
MALICIOUS_CATEGORIES = set(ALL_CATEGORIES).difference(BENIGN_CATEGORIES)

# 六条通道（文件名, 通道名, 地址列名）
CHANNEL_SPECS = [
    ('mem_read.csv',      'mem_read',      'GPA'),
    ('mem_write.csv',     'mem_write',     'GPA'),
    ('mem_readwrite.csv', 'mem_readwrite', 'GPA'),
    ('mem_exec.csv',      'mem_exec',      'GPA'),
    ('ata_read.csv',      'ata_read',      'LBA'),
    ('ata_write.csv',     'ata_write',     'LBA'),
]

# ----------------------- 工具函数 -----------------------

def _hash_path(p: str) -> str:
    """将路径哈希成安全文件名。"""
    return hashlib.md5(p.encode('utf-8')).hexdigest()


def _infer_category_from_path(p: Path) -> Optional[str]:
    """从路径各级目录名中推断类别名（如 'Conti'、'Firefox'）。找不到则返回 None。"""
    if "original" in str(p):
        parts = {x.lower() for x in p.parts}
        for c in ALL_CATEGORIES:
            if c.lower() in parts:
                return c
    elif "mix" in str(p):
        return "Conti"
    else:
        raise NotImplementedError(f"{p} not supported")


def _list_trace_dirs(domain_root: Path) -> List[Path]:
    """递归枚举包含 mem_read.csv 的目录，视为一个 trace 目录。"""
    out = []
    for dirpath, dirnames, filenames in os.walk(domain_root):
        if 'mem_read.csv' in filenames:  # 以存在 mem_read.csv 作为 trace 判据
            out.append(Path(dirpath))
    return out


def _stratified_split_by_category(
    trace_paths: List[Path],
    train_ratio=0.8, val_ratio=0.1, test_ratio=0.1,
    seed: int = 42
) -> Tuple[List[Path], List[Path], List[Path]]:
    """按类别分层将 trace 目录划分为 train/val/test（比例相加=1）。
    - 单位是 trace（目录），避免同一 trace 泄漏到多个集合。
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
    by_cat: Dict[str, List[Path]] = {}
    for p in trace_paths:
        cat = _infer_category_from_path(p)
        if cat is None:
            continue
        by_cat.setdefault(cat, []).append(p)

    rng = np.random.default_rng(seed)
    train, val, test = [], [], []
    for cat, items in by_cat.items():
        items = list(items)
        rng.shuffle(items)
        n = len(items)
        n_train = int(np.floor(n * train_ratio))
        n_val   = int(np.floor(n * val_ratio))
        n_test  = n - n_train - n_val

        # 确保 n >= 3 时，val 与 test 至少各 1 个
        if n >= 3:
            if n_val == 0:
                if n_train >= n_test and n_train > 0:
                    n_train -= 1
                elif n_test > 0:
                    n_test -= 1
                else:
                    n_train -= 1
                n_val += 1

            if n_test == 0:
                if n_train >= n_val and n_train > 0:
                    n_train -= 1
                elif n_val > 0:
                    n_val -= 1
                else:
                    n_train -= 1
                n_test += 1
        
        train += items[:n_train]
        val   += items[n_train:n_train+n_val]
        test  += items[n_train+n_val:]

    return train, val, test


# ----------------------- 特征构建（单个 trace） -----------------------

def build_windows_for_trace(trace_dir: Path, window_ms: int = 100) -> Tuple[np.ndarray, str, int]:
    """
    使用 23 维特征（存储5 + 内存18）
    type 列为 int8；全读入内存后做 bin 聚合。
    """
    cat = _infer_category_from_path(trace_dir)
    if cat is None:
        raise ValueError(f"无法从路径推断类别: {trace_dir}")
    label = 0 if cat in BENIGN_CATEGORIES else 1

    def _read_csv(csv_path: Path, names: List[str], dtypes: Dict[str, str]):
        # 假设 RanSMAP CSV 列数稳定；使用 usecols=range(len(names)) 保守读取
        df = pd.read_csv(
            csv_path, header=None, names=names, usecols=list(range(len(names))),
            engine='c', dtype=dtypes
        )
        # 统一 ts（纳秒）
        sec = df['UNIX_time_sec'].to_numpy('int64')
        ns  = df['UNIX_time_ns'].to_numpy('int64')
        df['ts'] = sec * 1_000_000_000 + ns

        # 补列与类型
        if 'size' not in df.columns:    df['size'] = np.int64(0)
        if 'entropy' not in df.columns: df['entropy'] = np.float64(-1.0)
        if 'type' not in df.columns:    df['type'] = np.int8(0)
        if 'LBA' in names and 'LBA' not in df.columns: df['LBA'] = np.int64(-1)
        if 'GPA' in names and 'GPA' not in df.columns: df['GPA'] = np.int64(-1)

        # 强制精确 dtype（节省内存）
        df['size']    = df['size'].astype('int64', copy=False)
        df['entropy'] = df['entropy'].astype('float64', copy=False)
        if 'type' in df: df['type'] = df['type'].astype('int8', copy=False)
        if 'LBA' in df:  df['LBA']  = df['LBA'].astype('int64', copy=False)
        if 'GPA' in df:  df['GPA']  = df['GPA'].astype('int64', copy=False)
        df['ts'] = df['ts'].astype('int64', copy=False)
        return df

    def _maybe_read(name, names, dtypes, is_mem=False):
        p = trace_dir / name
        if p.exists():
            return _read_csv(p, names, dtypes)
        # 空表（列齐全，后续聚合便于对齐）
        cols = {'UNIX_time_sec': np.array([], dtype='int64'),
                'UNIX_time_ns': np.array([], dtype='int64'),
                'ts': np.array([], dtype='int64'),
                'size': np.array([], dtype='int64'),
                'entropy': np.array([], dtype='float64')}
        if is_mem:
            cols['GPA']  = np.array([], dtype='int64')
            cols['type'] = np.array([], dtype='int8')
        else:
            cols['LBA']  = np.array([], dtype='int64')
        return pd.DataFrame(cols)

    # 列定义与 dtypes
    cols_mem  = ['UNIX_time_sec','UNIX_time_ns','GPA','size','entropy','type']
    dtypes_mem = {'UNIX_time_sec':'int64','UNIX_time_ns':'int64','GPA':'int64','size':'int64','entropy':'float64','type':'int8'}
    cols_ata  = ['UNIX_time_sec','UNIX_time_ns','LBA','size','entropy']
    dtypes_ata = {'UNIX_time_sec':'int64','UNIX_time_ns':'int64','LBA':'int64','size':'int64','entropy':'float64'}

    # 一次性读入内存
    df_mem_read      = _maybe_read('mem_read.csv',      cols_mem, dtypes_mem, is_mem=True)
    df_mem_write     = _maybe_read('mem_write.csv',     cols_mem, dtypes_mem, is_mem=True)
    df_mem_readwrite = _maybe_read('mem_readwrite.csv', cols_mem, dtypes_mem, is_mem=True)
    df_mem_exec      = _maybe_read('mem_exec.csv',      cols_mem, dtypes_mem, is_mem=True)
    df_ata_read      = _maybe_read('ata_read.csv',      cols_ata, dtypes_ata, is_mem=False)
    df_ata_write     = _maybe_read('ata_write.csv',     cols_ata, dtypes_ata, is_mem=False)

    # 时间范围
    ts_min_candidates = []
    ts_max_candidates = []
    for df in [df_mem_read, df_mem_write, df_mem_readwrite, df_mem_exec, df_ata_read, df_ata_write]:
        if not df.empty:
            ts_min_candidates.append(int(df['ts'].min()))
            ts_max_candidates.append(int(df['ts'].max()))
    if not ts_min_candidates:
        return np.zeros((0, 23), dtype=np.float64), cat, label

    t0 = min(ts_min_candidates)
    t1 = max(ts_max_candidates)
    T_window_ns = int(window_ms) * 1_000_000
    n_bins = int((t1 - t0) // T_window_ns + 1)

    # 预分配 23 维数组（默认值：熵=-1，其余=0）
    T_sr  = np.zeros(n_bins, dtype=np.float64)
    T_sw  = np.zeros(n_bins, dtype=np.float64)
    V_sr  = np.zeros(n_bins, dtype=np.float64)
    V_sw  = np.zeros(n_bins, dtype=np.float64)
    H_sw  = np.full(n_bins, -1.0, dtype=np.float64)

    H_mw  = np.full(n_bins, -1.0, dtype=np.float64)
    H_mrw = np.full(n_bins, -1.0, dtype=np.float64)

    C_4KBr  = np.zeros(n_bins, dtype=np.float64)
    C_4KBw  = np.zeros(n_bins, dtype=np.float64)
    C_4KBrw = np.zeros(n_bins, dtype=np.float64)
    C_4KBx  = np.zeros(n_bins, dtype=np.float64)

    C_2MBr  = np.zeros(n_bins, dtype=np.float64)
    C_2MBw  = np.zeros(n_bins, dtype=np.float64)
    C_2MBrw = np.zeros(n_bins, dtype=np.float64)
    C_2MBx  = np.zeros(n_bins, dtype=np.float64)

    C_MMIOr  = np.zeros(n_bins, dtype=np.float64)
    C_MMIOw  = np.zeros(n_bins, dtype=np.float64)
    C_MMIOrw = np.zeros(n_bins, dtype=np.float64)
    C_MMIOx  = np.zeros(n_bins, dtype=np.float64)

    V_mr  = np.zeros(n_bins, dtype=np.float64)
    V_mw  = np.zeros(n_bins, dtype=np.float64)
    V_mrw = np.zeros(n_bins, dtype=np.float64)
    V_mx  = np.zeros(n_bins, dtype=np.float64)

    # ---------- 工具：对一个 DF 计算 bin ----------
    def _add_bin(df):
        if df.empty: return df
        # 直接在数组层面算 bin，减少中间对象
        bin_idx = ((df['ts'].to_numpy('int64') - t0) // T_window_ns).astype(np.int64)
        df = df.copy()  # 避免修改原 DF
        df['bin'] = bin_idx
        return df

    # ---------- 存储通道：read/write ----------
    if not df_ata_read.empty:
        dfr = _add_bin(df_ata_read)
        g = dfr.groupby('bin', sort=False)
        # Throughput & variance
        T_sr[g.size().index.to_numpy()] = g['size'].sum().to_numpy(dtype=np.float64) / T_window_ns
        # pandas 的 var 默认 ddof=1；用 np.var 语义 -> ddof=0
        V_sr[g.size().index.to_numpy()] = g['LBA'].agg(lambda s: float(np.var(s.to_numpy('int64'), ddof=0))).to_numpy()
    if not df_ata_write.empty:
        dfw = _add_bin(df_ata_write)
        g = dfw.groupby('bin', sort=False)
        T_sw[g.size().index.to_numpy()] = g['size'].sum().to_numpy(dtype=np.float64) / T_window_ns
        V_sw[g.size().index.to_numpy()] = g['LBA'].agg(lambda s: float(np.var(s.to_numpy('int64'), ddof=0))).to_numpy()
        H_sw[g.size().index.to_numpy()] = g['entropy'].mean().to_numpy(dtype=np.float64)

    # ---------- 内存通道：read/write/readwrite/exec ----------
    def _mem_entropy(df, out_arr):
        if df.empty: return
        d = _add_bin(df)
        g = d.groupby('bin', sort=False)
        out_arr[g.size().index.to_numpy()] = g['entropy'].mean().to_numpy(dtype=np.float64)

    def _mem_var_gpa(df, out_arr):
        if df.empty: return
        d = _add_bin(df)
        g = d.groupby('bin', sort=False)
        out_arr[g.size().index.to_numpy()] = g['GPA'].agg(lambda s: float(np.var(s.to_numpy('int64'), ddof=0))).to_numpy()

    def _mem_type_counts(df, out_1, out_2, out_4):
        if df.empty: return
        d = _add_bin(df)
        # 统计 (bin, type) 计数（type 为 int8，低内存）
        ct = pd.crosstab(d['bin'], d['type'])
        if 1 in ct.columns: out_1[ct.index.to_numpy()] = ct[1].to_numpy(dtype=np.float64)
        if 2 in ct.columns: out_2[ct.index.to_numpy()] = ct[2].to_numpy(dtype=np.float64)
        if 4 in ct.columns: out_4[ct.index.to_numpy()] = ct[4].to_numpy(dtype=np.float64)

    # 熵
    _mem_entropy(df_mem_write,     H_mw)
    _mem_entropy(df_mem_readwrite, H_mrw)
    # 计数（4KB=1, 2MB=2, MMIO=4）
    _mem_type_counts(df_mem_read,      C_4KBr,  C_2MBr,  C_MMIOr)
    _mem_type_counts(df_mem_write,     C_4KBw,  C_2MBw,  C_MMIOw)
    _mem_type_counts(df_mem_readwrite, C_4KBrw, C_2MBrw, C_MMIOrw)
    _mem_type_counts(df_mem_exec,      C_4KBx,  C_2MBx,  C_MMIOx)
    # GPA 方差
    _mem_var_gpa(df_mem_read,      V_mr)
    _mem_var_gpa(df_mem_write,     V_mw)
    _mem_var_gpa(df_mem_readwrite, V_mrw)
    _mem_var_gpa(df_mem_exec,      V_mx)

    # 组装 [n_bins, 23]
    windows = np.column_stack([
        T_sr, T_sw, V_sr, V_sw, H_sw,
        H_mw, H_mrw,
        C_4KBr, C_4KBw, C_4KBrw, C_4KBx,
        C_2MBr, C_2MBw, C_2MBrw, C_2MBx,
        C_MMIOr, C_MMIOw, C_MMIOrw, C_MMIOx,
        V_mr, V_mw, V_mrw, V_mx
    ]).astype(np.float64, copy=False)

    return windows, cat, label


# ----------------------- 缓存与清单 -----------------------

def build_cache(
    trace_dirs: List[Path], cache_root: Path,
    window_ms: int = 100
) -> List[Dict]:
    """为给定 trace 列表构建/复用缓存。返回 manifest 项列表：
    [{'trace_id','category','label','cache_file','n_windows'}]
    """
    cache_root.mkdir(parents=True, exist_ok=True)
    (cache_root / 'windows').mkdir(parents=True, exist_ok=True)

    manifest: List[Dict] = []
    for td in tqdm(trace_dirs, desc='构建缓存'):
        trace_id = str(td.resolve())
        hid = _hash_path(trace_id)
        out_path = cache_root / 'windows' / f'{hid}.npz'
        meta = {
            'trace_id': trace_id,
            'cache_file': str(out_path),
        }
        if out_path.exists():
            with np.load(out_path, mmap_mode='r') as npz:
                meta['category'] = str(npz['category'])
                meta['label'] = int(npz['label'])
                meta['n_windows'] = int(npz['windows'].shape[0])
        else:
            try:
                windows, cat, label = build_windows_for_trace(td, window_ms)
            except Exception as e:
                print(f"[跳过] {td}: {e}")
                import traceback
                traceback.print_exc()
                continue
            np.savez(out_path, windows=windows, category=np.array(cat), label=np.array(label))
            meta['category'] = cat
            meta['label'] = label
            meta['n_windows'] = int(windows.shape[0])
        manifest.append(meta)
    # 保存清单
    with open(cache_root / 'manifest.json', 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def compute_standardizer(cache_entries: List[Dict], eps: float = 1e-6) -> Dict[str, np.ndarray]:
    """仅用训练集的窗口矩阵计算特征均值与标准差，返回 {'mean':..., 'std':...}。"""
    s1 = None
    s2 = None
    n_total = 0
    for m in tqdm(cache_entries, desc='计算标准化参数'):
        with np.load(m['cache_file'], mmap_mode='r') as npz:
            W = npz['windows']  # [T, F]
            lo = int(m.get('win_lo', 0))                 # 仅用训练子区间
            hi = int(m.get('win_hi', W.shape[0]))
            Ws = W[lo:hi]
            if Ws.size == 0:
                continue
            if s1 is None:
                F = Ws.shape[1]
                s1 = np.zeros(F, dtype=np.float64)
                s2 = np.zeros(F, dtype=np.float64)
            s1 += Ws.sum(axis=0)
            s2 += (Ws * Ws).sum(axis=0)
            n_total += Ws.shape[0]

    mean = s1 / max(1, n_total)
    var = s2 / max(1, n_total) - mean * mean
    std = np.sqrt(np.clip(var, 0.0, None))
    std = np.where(std < eps, 1.0, std)
    return {'mean': mean.astype('float64'), 'std': std.astype('float64')}


# ----------------------- 数据集 -----------------------

class SeqDataset(Dataset):
    """基于缓存窗口，按需切片为定长序列。
    - 为避免大规模 IO，__getitem__ 动态从对应 trace 的 windows 切片。
    - 支持传入标准化参数（mean/std）。
    """
    def __init__(
        self,
        entries: List[Dict],
        seq_len: int = 256,
        stride: int = 64,
        standardizer: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.entries = entries
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.standardizer = standardizer

        # 预构建索引：列表元素为 (entry_idx, start_bin)
        self.index: List[Tuple[int,int]] = []
        for ei, m in enumerate(self.entries):
            n_all = int(m.get('n_windows', 0))
            lo = int(m.get('win_lo', 0))
            hi = int(m.get('win_hi', n_all))

            L = self.seq_len
            S = self.stride
            if n_all >= L:
                for st in range(0, n_all - L + 1, S):
                    self.index.append((ei, st))
            else:
                print(f"[Warning] not enough data length {n_all}. Try to lower seq_len {L}")

        if not self.index:
            raise RuntimeError('序列样本数为 0，请检查窗口参数/数据规模。')

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int):
        ei, st = self.index[idx]
        m = self.entries[ei]
        with np.load(m['cache_file'], mmap_mode='r') as npz:
            W = npz['windows']  # [T, F]
            lo = int(m.get('win_lo', 0))  # 子区间起点
            X = W[lo + st : lo + st + self.seq_len]  # 在子区间内切片
            y = int(npz['label'])

        if self.standardizer is not None:
            X = (X - self.standardizer['mean']) / self.standardizer['std']
        # 返回 torch 张量（float32 更省显存）
        return torch.from_numpy(X.astype('float32')), torch.tensor(y, dtype=torch.long)


# ----------------------- 主入口：构建 DataLoader -----------------------

def build_dataloaders(
    dataset_root: str,
    cache_root: str,
    train_domain: str = "original/celeron-gen6/ddr4-1600-8g",
    ood_domains: Optional[List[str]] = None,
    max_traces_per_category: Optional[int] = 3,
    window_ms: int = 100,
    seq_len: int = 256,
    stride: int = 64,
    batch_size: int = 32,
    is_hdd: bool = False,
    seed: int = 42,
) -> Tuple[Tuple[DataLoader, DataLoader, DataLoader], Dict]:
    """构建 train/val/test DataLoader。

    参数：
    - dataset_root: RanSMAP 根目录。
    - cache_root: 缓存根目录（会创建 windows/、manifest.json、scaler.npz 等）。
    - train_domain: 同域训练的硬件/场景子路径（如 original/celeron-gen6/ddr4-1600-8g）。
    - ood_domains: OOD 测试域（如 mix/...、variants/...），可为 None。
    - max_traces_per_category: 每个类别最多选多少个 trace，None 则不限制。
    - window_ms, seq_len, stride: 时间窗与序列参数。
    - batch_size, num_workers, pin_memory: DataLoader 参数。

    返回：((train_loader, val_loader, test_loader), info_dict)
    """
    rng = np.random.default_rng(seed)
    dataset_root = Path(dataset_root)
    cache_root = Path(cache_root)

    # 1) 收集同域训练 trace 目录
    train_domain_root = dataset_root / train_domain
    assert train_domain_root.exists(), f"路径不存在: {train_domain_root}"
    all_traces = _list_trace_dirs(train_domain_root)

    # 按类别采样上限
    by_cat: Dict[str, List[Path]] = {}
    for p in all_traces:
        cat = _infer_category_from_path(p)
        if cat is None:
            continue
        by_cat.setdefault(cat, []).append(p)
    selected: List[Path] = []
    for cat, items in by_cat.items():
        items = list(items)
        rng.shuffle(items)
        if max_traces_per_category is not None:
            items = items[:max_traces_per_category]
        selected += items

    # 2) 划分 train/val/test（in-domain）
    tr_ids, va_ids, te_ids_in = _stratified_split_by_category(selected, 0.8, 0.1, 0.1, seed)

    # 3) 附加 OOD 测试域（可选）
    te_ids_ood: List[Path] = []
    if ood_domains:
        for d in ood_domains:
            root = dataset_root / d
            if not root.exists():
                print(f"[跳过 OOD] 不存在: {root}")
                continue
            ood_traces = _list_trace_dirs(root)
            # 可按需要再限流
            by_cat_ood: Dict[str, List[Path]] = {}
            for p in ood_traces:
                cat = _infer_category_from_path(p)
                if cat is None:
                    print(f"[跳过] 无法推断类别: {p}")
                    continue
                by_cat_ood.setdefault(cat, []).append(p)
            for cat, items in by_cat_ood.items():
                rng.shuffle(items)
                te_ids_ood += items
    
    print(f"te_ids_ood: {len(te_ids_ood)}")

    # 4) 构建/复用缓存
    cache_traces = tr_ids + va_ids + te_ids_in + te_ids_ood
    manifest = build_cache(cache_traces, cache_root, window_ms)
    # 建立 trace_id -> manifest 映射
    m_by_id = {m['trace_id']: m for m in manifest}

    # 5) 组装各集合的 manifest
    # --- 当每个类别仅 1 个 trace（或总体无法目录级划分）时，按 trace 内时间划分 80/10/10 ---
    by_cat_counts = {}
    for p in selected:
        cat = _infer_category_from_path(p)
        if cat:
            by_cat_counts[cat] = by_cat_counts.get(cat, 0) + 1

    need_intra_trace_split = all(cnt <= 1 for cnt in by_cat_counts.values() if cnt > 0)

    if need_intra_trace_split:
        print("need intra trace split")
        tr_entries, va_entries, te_entries = [], [], []

        def sub_entry(m, lo, hi):
            mm = dict(m)               # 浅拷贝即可
            mm['win_lo'] = int(lo)
            mm['win_hi'] = int(hi)
            # 为了让数据集长度估算更准确，顺手更新 n_windows 为子区间长度
            mm['n_windows'] = int(max(0, hi - lo))
            return mm

        for p in selected:
            mp = m_by_id.get(str(p.resolve()))
            if not mp:
                continue
            n = int(mp['n_windows'])
            if n <= 0:
                continue
            b0 = int(n * 0.80)
            b1 = int(n * 0.90)
            # 三段： [0,b0), [b0,b1), [b1,n)
            tr_entries.append(sub_entry(mp, 0,  b0))
            va_entries.append(sub_entry(mp, b0, b1))
            te_entries.append(sub_entry(mp, b1, n))
    else:
        # 保持原逻辑：目录级（trace 级）分层划分
        tr_entries = [m_by_id[str(p.resolve())] for p in tr_ids if str(p.resolve()) in m_by_id]
        va_entries = [m_by_id[str(p.resolve())] for p in va_ids if str(p.resolve()) in m_by_id]
        te_entries = [m_by_id[str(p.resolve())] for p in (te_ids_in + te_ids_ood) if str(p.resolve()) in m_by_id]
    
    # 6) 计算标准化（仅基于训练集）
    scaler = compute_standardizer(tr_entries)
    np.savez(cache_root / 'scaler.npz', mean=scaler['mean'], std=scaler['std'])

    # 7) 构建 Dataset 与 DataLoader
    train_ds = SeqDataset(tr_entries, seq_len, stride, scaler)
    val_ds   = SeqDataset(va_entries, seq_len, seq_len, scaler)
    test_ds  = SeqDataset(te_entries, seq_len, seq_len, scaler)

    if is_hdd:
        # DataLoader 的 HDD 友好配置
        dl_kwargs = dict(batch_size=batch_size, num_workers=2, pin_memory=True,
                         shuffle=True, persistent_workers=True, prefetch_factor=2)
    else:
        use_cuda = torch.cuda.is_available()
        num_cpu = os.cpu_count() or 14 
        dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=6,
            pin_memory=use_cuda,
            shuffle=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

    train_loader = DataLoader(train_ds, **dl_kwargs)
    val_loader   = DataLoader(val_ds,   **{**dl_kwargs, 'shuffle': False})
    test_loader  = DataLoader(test_ds,  **{**dl_kwargs, 'shuffle': False})

    info = {
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'feature_dim': int(scaler['mean'].shape[0]),
        'seq_len': seq_len,
        'stride': stride,
        'window_ms': window_ms,
        'train_domain': str(train_domain_root),
        'ood_domains': ood_domains or [],
        'cache_root': str(cache_root),
    }
    # 保存信息
    with open(cache_root / 'info.json', 'w', encoding='utf-8') as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    return (train_loader, val_loader, test_loader), info


# ----------------------- 可选：直接运行做一次构建 -----------------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='RanSMAP 数据模块')
    parser.add_argument('--dataset_root', type=str, default=r"G:\T\Projects\计算机病毒原理及实践大作业\RanSMAP\dataset")
    parser.add_argument('--cache_root', type=str, default="data_cache")
    parser.add_argument('--train_domain', type=str, default='original/celeron-gen6/ddr4-1600-8g')
    parser.add_argument('--ood_domains', type=str, nargs='*', default=None)
    parser.add_argument('-m', '--max_traces_per_category', type=int, default=3)
    parser.add_argument('--window_ms', type=int, default=100)
    parser.add_argument('--seq_len', type=int, default=256)
    parser.add_argument('--stride', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    loaders, info = build_dataloaders(
        dataset_root=args.dataset_root,
        cache_root=args.cache_root,
        train_domain=args.train_domain,
        ood_domains=args.ood_domains,
        max_traces_per_category=args.max_traces_per_category,
        window_ms=args.window_ms,
        seq_len=args.seq_len,
        stride=args.stride,
        batch_size=args.batch_size,
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))
