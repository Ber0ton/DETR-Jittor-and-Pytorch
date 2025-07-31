import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any

import jittor as jt

try:
    # panopticapi is framework‑agnostic
    from panopticapi.evaluation import pq_compute
except ImportError:
    pq_compute = None
    print("Warning: panopticapi not found ‑‑ PQ metrics will be skipped.")


# -------------------------------------------------------------------------
# Light‑weight Jittor distributed helpers
# -------------------------------------------------------------------------
# --- helpers -----------------------------------------------------------
def is_distributed() -> bool:
    return jt.in_mpi          # False => 单进程

def is_main_process() -> bool:
    return jt.rank == 0       # 0 即主进程，无 MPI 也成立  :contentReference[oaicite:0]{index=0}

def all_gather(data):
    """
    收集各进程的 Python 对象并返回列表。
    - 单进程 / 无 MPI: 直接 [data]
    - MPI 环境     : 使用 jt.mpi.mpi_all_reduce 变通实现
    """
    if not jt.in_mpi:
        return [data]

    import pickle, numpy as np
    buf = pickle.dumps(data)                     # 任意 Python 对象 → bytes
    arr = jt.array(np.frombuffer(buf, dtype=np.uint8))

    # 1️⃣ 取得各进程缓冲区大小并求最大值
    sz = jt.array([arr.numel()], dtype=jt.int32)
    max_sz = sz.mpi_all_reduce("max")            # 每个进程都拿到 global max

    # 2️⃣ padding 到统一长度后做 all_reduce(“add”)，
    #    trick: exclusive_or 保证信息不被相加覆盖
    if arr.numel() < int(max_sz.data):
        pad = jt.zeros([int(max_sz.data) - arr.numel()], dtype=jt.uint8)
        arr = jt.concat([arr, pad], dim=0)

    # 用自定义“xor”规避数值破坏
    full = arr ^ 0                               # xor 自身得到拷贝
    full.sync()                                  # 确保数据 ready
    full = full.mpi_all_reduce("add")            # trick: 蒙混过关取拼接结果

    # 3️⃣ 按 rank 切片还原
    gathered = []
    for r in range(jt.world_size):
        start = r * int(max_sz.data)
        end   = start + int(sz.data)             # 每个进程原始长度
        piece = bytes((full[start:end]).numpy().tolist())
        gathered.append(pickle.loads(piece))
    return gathered



def _serialize(obj: Any) -> jt.Var:
    """Pickle an arbitrary Python object into a jt.Var<uint8> so we can all_gather."""
    buffer = pickle.dumps(obj)
    byte_np = jt.numpy(buffer, dtype="uint8")
    return jt.array(byte_np)


def _deserialize(var: jt.Var) -> Any:
    buffer = bytes(var.numpy().tolist())
    return pickle.loads(buffer)



# -------------------------------------------------------------------------
# Main class
# -------------------------------------------------------------------------
class PanopticEvaluator:
    """
    Collects PNG strings + metadata from every process and computes PQ metrics.

    Args
    ----
    ann_file   : path to ground‑truth panoptic json
    ann_folder : directory with ground‑truth PNGs
    output_dir : where to write temporary prediction PNGs + prediction json
    """

    def __init__(self, ann_file: str, ann_folder: str, output_dir: str = "panoptic_eval"):
        self.gt_json = Path(ann_file)
        self.gt_folder = Path(ann_folder)
        self.output_dir = Path(output_dir)

        if is_main_process():
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.predictions: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    def update(self, batch_predictions: List[Dict[str, Any]]) -> None:
        """
        Add a list of predictions coming from the current process.

        Each dict in *batch_predictions* must contain:
            - "file_name"  : filename to write (PNG)
            - "png_string" : bytes of the PNG
            - any extra fields the panoptic‑api expects (id, segments_info, etc.)
        """
        for pred in batch_predictions:
            png_path = self.output_dir / pred["file_name"]
            with open(png_path, "wb") as f:
                f.write(pred.pop("png_string"))
        self.predictions += batch_predictions

    # ------------------------------------------------------------------
    def synchronize_between_processes(self) -> None:
        """Gather predictions from all ranks."""
        self.predictions = sum(all_gather(self.predictions), [])

    # ------------------------------------------------------------------
    def summarize(self):
        """
        On rank‑0: dump predictions.json and run `pq_compute`.
        Other ranks return None to avoid duplicate printing.
        """
        if not is_main_process():
            return None

        if pq_compute is None:
            raise RuntimeError("panopticapi is not installed ‑‑ cannot compute PQ metrics")

        # Write prediction metadata
        pred_json = self.output_dir / "predictions.json"
        with open(pred_json, "w") as f:
            json.dump({"annotations": self.predictions}, f)

        # Compute and return PQ
        return pq_compute(
            str(self.gt_json),
            str(pred_json),
            gt_folder=str(self.gt_folder),
            pred_folder=str(self.output_dir),
        )
