# -*- coding: utf-8 -*-
"""
FAME ProcessingModule：RanSMAP 深度学习分类器（23维窗口特征 → GRU/LSTM）
-----------------------------------------------------------------------
功能：
1) 接收网页上传的 RanSMAP trace：可以是目录，或 .zip（自动解压到临时目录）
2) 构建 23 维窗口特征（依赖你已有的数据模块 build_windows_for_trace）
3) 切成定长序列，加载 ckpt_best.pth 推理，给出良性/勒索分类与置信度
4) 输出 tags、results，并附加 JSON 报告为 Support File

参考 FAME 文档（ProcessingModule API / acts_on / add_support_file / results 等）：
- https://fame.readthedocs.io/en/latest/modules.html
"""
from __future__ import annotations
import os
import io
import sys
import json
import shutil
import zipfile
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional

try:
    import numpy as np
    import torch
    import torch.nn as nn
    HAVE_NUMPY_AND_TORCH = True
except ImportError:
    HAVE_NUMPY_AND_TORCH = False

from fame.common.exceptions import ModuleInitializationError
from fame.core.module import ProcessingModule  # FAME 提供

CURRENT_DIR = Path(__file__).parent

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

# ---------------------------
# 主模块
# ---------------------------
class RansmapDLClassifier(ProcessingModule):
    """
    处理 RanSMAP trace（目录或 zip），输出是否“勒索软件”的判断与置信度。
    """
    name = "ransmap_dl_classifier"
    description = "Classify a RanSMAP trace (directory/zip) as ransomware or benign using a GRU/LSTM model."
    # 让 FAME 都能调到，再在 each() 里自行判断是否是 zip/目录
    acts_on = None

    # 在 Web UI 中可配置的参数（FAME 文档：Module.config）
    config = [
        {"name": "model_path",   "type": "str", "default": str(CURRENT_DIR/"ckpt_best.pt"), "description": "训练得到的 ckpt_best.pt 路径"},
        {"name": "scaler_path",  "type": "str", "default": str(CURRENT_DIR/"scaler.npz"), "description": "标准化参数 scaler.npz 路径"},
        {"name": "code_root",    "type": "str", "default": str(CURRENT_DIR/"ransmap_dl_utils"), "description": "可选：ransom_data_module.py 所在目录（为空则按系统路径导入）"},
        {"name": "window_ms",    "type": "integer", "default": 100, "description": "窗口宽度（毫秒），需与训练一致"},
        {"name": "seq_len",      "type": "integer", "default": 256, "description": "序列长度 L"},
        {"name": "stride",       "type": "integer", "default": 256, "description": "序列步长 S（测试推荐与 L 相同）"},
        {"name": "threshold",    "type": "text", "default": "0.5", "description": "判定阈值（平均概率 > 阈值 则判为 ransomware）"},
    ]

    def initialize(self):
        """模块初始化：依赖检查、导入数据处理函数、加载权重/标准化。"""
        if not HAVE_NUMPY_AND_TORCH:
            raise ModuleInitializationError(self, "Missing dependency")

        # 可选：把你的数据模块所在目录加入 sys.path（方便在 FAME worker 容器内定位）
        code_root = (self.get_setting("code_root") or "").strip()
        if code_root:
            sys.path.insert(0, code_root)

        # 导入你前面实现的窗口构建函数（已用 pyarrow + bin 聚合 + 23 维）
        try:
            from ransom_data_module import build_windows_for_trace  # noqa: F401
            self.build_windows_for_trace = build_windows_for_trace
        except Exception as e:
            raise RuntimeError(
                "无法导入 ransom_data_module.build_windows_for_trace，请在“设置->code_root”中指向包含该模块的目录，"
                "或将其安装到 FAME worker 的 PYTHONPATH。原始错误：" + str(e)
            )

        # 加载 scaler（用于标准化）、并预创建“假输入”以获知 input_dim
        scaler_path = self.get_setting("scaler_path")
        if not scaler_path or not os.path.exists(scaler_path):
            raise RuntimeError("请在模块设置中提供有效的 scaler_path（scaler.npz）")
        self.scaler = load_scaler_npz(scaler_path)
        input_dim = int(self.scaler["mean"].shape[0])

        # 加载模型
        model_path = self.get_setting("model_path")
        if not model_path or not os.path.exists(model_path):
            raise RuntimeError("请在模块设置中提供有效的 model_path（ckpt_best.pth）")
        self.device = "cpu"  # FAME worker 环境通常无 GPU，统一走 CPU
        self.model, self.train_cfg = load_model_from_ckpt(model_path, input_dim=input_dim, device=self.device)

        # 读取基础配置
        self.window_ms = int(self.get_setting("window_ms") or 100)
        self.seq_len   = int(self.get_setting("seq_len") or 256)
        self.stride    = int(self.get_setting("stride") or self.seq_len)
        try:
            self.threshold = float(self.get_setting("threshold") or "0.5")
        except Exception:
            self.threshold = 0.5

    # ---------------------------
    # FAME 将针对每个“目标文件”调用 each(target)
    # ---------------------------
    def each(self, target: str) -> bool:
        """
        参数：
          target: FAME 传入的待分析对象路径（通常是文件路径；我们支持目录或 .zip）
        返回：
          True 表示分析成功（将自动把 tags / results 添加到分析中）；False 表示没做事
        """
        path = Path(target)
        if not path.exists():
            self.log("error", f"目标不存在：{path}")
            return False

        # 1) 若是 zip 则解压到临时目录；若为目录则直接用
        tmpdir_obj = None
        try:
            if path.is_file() and path.suffix.lower() == ".zip":
                tmpdir_obj = tempfile.TemporaryDirectory(prefix="ransmap_zip_")
                with zipfile.ZipFile(str(path), "r") as zf:
                    zf.extractall(tmpdir_obj.name)
                trace_dir = Path(tmpdir_obj.name)
            elif path.is_dir():
                trace_dir = path
            else:
                # 既不是目录也不是 zip，忽略（返回 False 以便 FAME 继续其它模块）
                self.log("info", f"非 RanSMAP 目录或 zip，跳过：{path}")
                return False

            # 2) 构建窗口特征（[T, 23]）
            windows, cat, label = self.build_windows_for_trace(trace_dir, window_ms=self.window_ms)
            if windows.shape[0] == 0:
                self.log("warning", "未提取到任何窗口，可能目录结构不完整（缺少 mem_*/ata_* CSV）")
                return False

            # 3) 标准化 + 切序列
            windows_std = standardize_windows(windows, self.scaler)  # [T, 23]
            X = make_sequences(windows_std, self.seq_len, self.stride)  # [N, L, 23]
            if X.shape[0] == 0:
                self.log("warning", "有效序列数为 0，跳过")
                return False

            # 4) 推理
            with torch.no_grad():
                xb = torch.from_numpy(X).to(self.device)
                logits = self.model(xb)                   # [N, 2]
                probs = torch.softmax(logits, dim=1)      # [N, 2]
                probs_np = probs.cpu().numpy()
                p_mal = float(np.mean(probs_np[:, 1]))    # 多序列概率求平均
                p_ben = 1.0 - p_mal
                pred_label = "ransomware" if p_mal >= self.threshold else "benign"

            # 5) 结果与标注
            self.results["summary"] = {
                "pred_label": pred_label,
                "prob_ransomware": round(p_mal, 6),
                "prob_benign": round(p_ben, 6),
                "threshold": self.threshold,
            }
            self.results["detail"] = {
                "num_windows": int(windows.shape[0]),
                "num_sequences": int(X.shape[0]),
                "window_ms": self.window_ms,
                "seq_len": self.seq_len,
                "stride": self.stride,
                "train_backbone": self.train_cfg.get("backbone", "gru"),
                "train_hidden_size": int(self.train_cfg.get("hidden_size", 128)),
                "train_num_layers": int(self.train_cfg.get("num_layers", 2)),
                "train_dropout": float(self.train_cfg.get("dropout", 0.2)),
                "category_inferred_from_path": cat,  # 仅来自路径推断，用于参考
            }

            # tag：按照判定添加
            self.add_tag(pred_label)
            self.add_tag("ransmap")

            # 支持文件：保存一份 JSON 报告，便于下载
            out_dir = Path(tempfile.mkdtemp(prefix="ransmap_out_"))
            rep_path = out_dir / f"{self.name}_report.json"
            with open(rep_path, "w", encoding="utf-8") as f:
                json.dump({"results": self.results}, f, ensure_ascii=False, indent=2)
            self.add_support_file(f"{self.name}_report.json", str(rep_path))

            return True

        finally:
            if tmpdir_obj is not None:
                tmpdir_obj.cleanup()
