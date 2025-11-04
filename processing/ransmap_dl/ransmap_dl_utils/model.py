from __future__ import annotations
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn

# ---------------------------
# 轻量 RNN 分类器（与训练脚本一致）
# ---------------------------
class RNNClassifier(nn.Module):
    """时间序列二分类器：RNN(GRU/LSTM) + 最后一步 + MLP"""
    def __init__(self, input_dim: int, hidden_size: int, num_layers: int, dropout: float, backbone: str = 'gru'):
        super().__init__()
        rnn_cls = nn.GRU if backbone.lower() == 'gru' else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_dim, hidden_size=hidden_size, num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0, batch_first=True, bidirectional=False
        )
        self.drop = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x):
        out, h = self.rnn(x)
        if isinstance(h, tuple):  # LSTM: (h, c)
            h = h[0]
        feat = self.drop(h[-1])   # 取最后一层的最后步
        return self.head(feat)

# ---------------------------
# 工具：模型与标准化载入
# ---------------------------
def load_model_from_ckpt(ckpt_path: str, input_dim: int, device: str = "cpu") -> Tuple[nn.Module, Dict]:
    """从 ckpt_best.pth 重建模型（按训练时 cfg），仅 CPU 推理以保证兼容 worker 环境。"""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("cfg", {})  # 训练时保存的 TrainConfig 字典
    backbone = cfg.get("backbone", "gru")
    hidden_size = int(cfg.get("hidden_size", 128))
    num_layers = int(cfg.get("num_layers", 2))
    dropout = float(cfg.get("dropout", 0.2))
    model = RNNClassifier(input_dim, hidden_size, num_layers, dropout, backbone=backbone)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    return model, cfg

def load_scaler_npz(path: str) -> Dict[str, np.ndarray]:
    """载入 scaler.npz（包含 mean/std）"""
    with np.load(path) as npz:
        mean = npz["mean"].astype("float64")
        std  = npz["std"].astype("float64")
    std = np.where(std < 1e-6, 1.0, std)
    return {"mean": mean, "std": std}

def standardize_windows(windows: np.ndarray, scaler: Dict[str, np.ndarray]) -> np.ndarray:
    """对窗口特征做标准化（与训练一致）。"""
    return (windows - scaler["mean"]) / scaler["std"]

# ---------------------------
# 工具：序列切片
# ---------------------------
def make_sequences(windows: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """
    将 [T, F] 的窗口矩阵切为 [N, L, F] 的序列；若 T < L，则零填充形成 1 条序列。
    - 验证/测试建议 stride=L（不重叠）；这里默认用配置项的 stride。
    """
    T, F = windows.shape
    if T <= 0:
        return np.zeros((0, seq_len, F), dtype=np.float32)
    if T < seq_len:
        pad = np.zeros((seq_len - T, F), dtype=windows.dtype)
        one = np.concatenate([windows, pad], axis=0)[None, ...]
        return one.astype(np.float32, copy=False)
    idxs = list(range(0, T - seq_len + 1, max(1, stride)))
    seqs = np.stack([windows[i:i+seq_len] for i in idxs], axis=0)
    return seqs.astype(np.float32, copy=False)
