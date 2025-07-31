import os
import subprocess
import time
import datetime
import pickle
from collections import defaultdict, deque
from typing import Optional, List

import jittor as jt


def detr_batchify_fn(batch):
    """
    batch: List[Tuple[image, target]]
      - image : jt.Var [3,H,W] (不同样本 H/W 可不同)
      - target: dict( jt.Var / 数字 等 )
    返回:
      - NestedTensor(images, masks)  ← 见 util.misc.nested_tensor_from_tensor_list
      - List[target]                 ← 与 PyTorch 版保持一致
    """
    imgs, targets = zip(*batch)                # 拆分
    imgs = nested_tensor_from_tensor_list(list(imgs))
    return imgs, list(targets)


# -----------------------------------------------------------------------------
# Compatibility aliases so that existing type hints keep working --------------
# -----------------------------------------------------------------------------
Tensor = jt.Var  # type alias so the original type annotations stay valid

# -----------------------------------------------------------------------------
# Small helpers ----------------------------------------------------------------
# -----------------------------------------------------------------------------

def _cuda_available() -> bool:
    """Return True if CUDA is enabled inside Jittor."""
    return bool(getattr(jt.flags, "has_cuda", False)) and jt.flags.has_cuda


def _max_cuda_mem_mb() -> float:
    """Return peak allocated memory in MB (fallbacks to 0 if not available)."""
    try:
        # Jittor doesn't expose the exact PyTorch API; provide best‑effort value.
        return 0.0  # placeholder – keep interface without raising.
    except Exception:
        return 0.0



# -----------------------------------------------------------------------------
# Smoothed value across multiple iterations -----------------------------------
# -----------------------------------------------------------------------------

class SmoothedValue:
    """Track a series of values and provide access to smoothed values."""

    def __init__(self, window_size: int = 20, fmt: Optional[str] = None):
        self.deque: deque = deque(maxlen=window_size)
        self.total: float = 0.0
        self.count: int = 0
        self.fmt = fmt or "{median:.4f} ({global_avg:.4f})"

    def update(self, value: float, n: int = 1):
        self.deque.append(float(value))
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """Synchronize the total and count across all MPI ranks."""
        if not is_dist_avail_and_initialized():
            return
        t = jt.array([self.count, self.total], dtype="float64")
        t = jt.mpi.mpi_all_reduce(t, op="add")
        self.count = int(t[0].item())
        self.total = float(t[1].item())

    # Properties -------------------------------------------------------------

    @property
    def median(self):
        d = jt.array(list(self.deque))
        return float(jt.median(d).item())

    @property
    def avg(self):
        d = jt.array(list(self.deque))
        return float(jt.mean(d).item())

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)

    @property
    def max(self):
        return max(self.deque) if self.deque else 0.0

    @property
    def value(self):
        return self.deque[-1] if self.deque else 0.0

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )

# -----------------------------------------------------------------------------
# MPI helpers -----------------------------------------------------------------
# -----------------------------------------------------------------------------

def all_gather(data):
    """Gather picklable *data* from every MPI rank into a list."""
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # Serialize to bytes then to jt.Var(uint8)
    buffer = pickle.dumps(data)
    tensor = jt.array(list(buffer), dtype="uint8")
    local_size = jt.array([tensor.numel()], dtype="int32")
    size_tensor = jt.mpi.mpi_all_reduce(local_size, op="add")  # total size
    max_size = int(size_tensor.item())  # crude upper bound

    # Pad to max_size for uniform all_reduce – safest but not most efficient.
    if tensor.numel() < max_size:
        pad = jt.zeros((max_size - tensor.numel(),), dtype="uint8")
        tensor = jt.concat([tensor, pad], dim=0)

    gathered = jt.mpi.mpi_all_reduce(tensor, op="add")  # now every rank has sum

    data_list = [pickle.loads(bytes(gathered.tolist()[:max_size]))]
    return data_list


def reduce_dict(input_dict, average: bool = True):
    """Reduce a dict of scalar jt.Var across all ranks."""
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    names, values = zip(*sorted(input_dict.items()))
    stacked = jt.stack(values, dim=0)
    stacked = jt.mpi.mpi_all_reduce(stacked, op="add")
    if average:
        stacked /= world_size
    return {k: v for k, v in zip(names, stacked)}

# -----------------------------------------------------------------------------
# Metric logger ---------------------------------------------------------------
# -----------------------------------------------------------------------------

class MetricLogger:
    def __init__(self, delimiter: str = "\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, jt.Var):
                v = float(v.item())
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
    def add_meter(self, name: str, meter: SmoothedValue):
        """Register a pre‑configured SmoothedValue under ``name``."""
        self.meters[name] = meter

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {attr!s}")

    def __str__(self):
        return self.delimiter.join(f"{name}: {meter}" for name, meter in self.meters.items())

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq: int, header: Optional[str] = None):
        i = 0
        header = header or ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        digits = len(str(len(iterable)))
        space_fmt = f":{digits}d"

        if _cuda_available():
            log_msg = self.delimiter.join([
                header,
                f"[{{0{space_fmt}}}/{{1}}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
                "max mem: {memory:.0f}",
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                f"[{{0{space_fmt}}}/{{1}}]",
                "eta: {eta}",
                "{meters}",
                "time: {time}",
                "data: {data}",
            ])

        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if _cuda_available():
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                            memory=_max_cuda_mem_mb(),
                        )
                    )
                else:
                    print(
                        log_msg.format(
                            i,
                            len(iterable),
                            eta=eta_string,
                            meters=str(self),
                            time=str(iter_time),
                            data=str(data_time),
                        )
                    )
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)")

# -----------------------------------------------------------------------------
# Misc utilities --------------------------------------------------------------
# -----------------------------------------------------------------------------

def get_sha():
    cwd = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
        diff_state = subprocess.check_output(["git", "diff-index", "HEAD"], cwd=cwd).decode().strip()
        diff = "has uncommitted changes" if diff_state else "clean"
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=cwd).decode().strip()
    except Exception:
        sha, diff, branch = "N/A", "clean", "N/A"
    return f"sha: {sha}, status: {diff}, branch: {branch}"


def collate_fn(batch):
    images, targets = zip(*batch)
    images = nested_tensor_from_tensor_list(list(images))
    return images, targets


def _max_by_axis(the_list: List[List[int]]):
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for idx, item in enumerate(sublist):
            maxes[idx] = max(maxes[idx], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors: Tensor, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device: str):
        tensors = self.tensors
        if device == "cuda":
            jt.flags.use_cuda = 1
            tensors = tensors.cuda()
        mask = self.mask.cuda() if self.mask is not None else None
        return NestedTensor(tensors, mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

# -----------------------------------------------------------------------------
# Creating a NestedTensor -----------------------------------------------------
# -----------------------------------------------------------------------------

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    if tensor_list[0].ndim != 3:
        raise ValueError("Only 3‑D tensors are supported")

    # Compute max size across the batch (C,H,W)
    max_size = [max(sizes) for sizes in zip(*[list(t.shape) for t in tensor_list])]
    batch_shape = [len(tensor_list)] + max_size
    dtype = tensor_list[0].dtype
    device = tensor_list[0].device

    tensor = jt.zeros(batch_shape, dtype=dtype)
    mask   = jt.ones((batch_shape[0], batch_shape[2], batch_shape[3]), dtype="bool")

    for i, img in enumerate(tensor_list):
        c, h, w = img.shape
        tensor[i, :c, :h, :w] = img
        mask[i, :h, :w] = False

    return NestedTensor(tensor, mask)


# -----------------------------------------------------------------------------
# Distributed helpers ---------------------------------------------------------
# -----------------------------------------------------------------------------

def is_dist_avail_and_initialized():
    return bool(getattr(jt, "in_mpi", False)) and jt.in_mpi


def get_world_size():
    return int(getattr(jt, "world_size", 1))


def get_rank():
    return int(getattr(jt, "rank", 0))


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        jt.save(*args, **kwargs)

# Dummy replacement – Jittor relies on mpirun so explicit initialization is
# usually unnecessary. Retained for API compatibility.

def init_distributed_mode(args):
    print("Jittor handles distributed initialization via mpirun – no action taken.")
    args.distributed = getattr(args, "distributed", False)

# -----------------------------------------------------------------------------
# Accuracy and interpolation helpers -----------------------------------------
# -----------------------------------------------------------------------------

def accuracy(output: Tensor, target: Tensor, topk=(1,)):
    if target.numel() == 0:
        return [jt.array(0.0)]
    maxk = max(topk)
    _, pred = jt.misc.topk(output, k=maxk, dim=1)
    pred = pred.transpose()
    correct = pred == target.view(1, -1).expand_as(pred)

    res = []
    batch_size = target.shape[0]
    for k in topk:
        correct_k = correct[:k].reshape(-1).float32().sum()
        res.append(correct_k * (100.0 / batch_size))
    return res


def interpolate(input: Tensor, size=None, scale_factor=None, mode="nearest", align_corners=None):
    """Wrapper around jt.nn.interpolate with empty‑batch support."""
    if input.numel() == 0:
        if size is None and scale_factor is not None:
            size = [int(dim * scale_factor) for dim in input.shape[-2:]]
        out_shape = list(input.shape[:-2]) + list(size)
        return jt.zeros(out_shape, dtype=input.dtype)
    return jt.nn.interpolate(input, size=size, scale_factor=scale_factor, mode=mode, align_corners=align_corners)






