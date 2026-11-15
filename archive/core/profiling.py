"""Measurement helpers for beam-search steps: utilisation sampling, random beams, profiling.

The heuristic used here is a fixed-size MLP. Its outputs are meaningless for solving;
it exists only so that the cost of heuristic evaluation is realistic.
"""
from __future__ import annotations

import subprocess
import threading
import time
from typing import Any

import numpy as np
import torch

from core.beam import CudaStageTimer, OneHot, beam_step
from core.torch_engine import TorchGatherPuzzle


class ProbeHeuristic(torch.nn.Module):
    """Timing-only stand-in for a learned heuristic."""

    def __init__(self, in_dim: int, hidden: int = 1024, depth: int = 3) -> None:
        super().__init__()
        layers: list[torch.nn.Module] = [torch.nn.Linear(in_dim, hidden), torch.nn.ReLU()]
        for _ in range(depth - 1):
            layers += [torch.nn.Linear(hidden, hidden), torch.nn.ReLU()]
        layers.append(torch.nn.Linear(hidden, 1))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GpuSampler:
    """Polls nvidia-smi in a background thread. Reports None if nvidia-smi is unavailable."""

    def __init__(self, interval_s: float = 0.5) -> None:
        self.interval = interval_s
        self.samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._failed = False

    @staticmethod
    def _query() -> tuple[float, float]:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout.strip().splitlines()[0]
        util, mem = (float(x) for x in out.split(","))
        return util, mem

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._query())
            except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                self._failed = True
                return
            self._stop.wait(self.interval)

    def start(self) -> None:
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self._failed or not self.samples:
            return {"gpu_util_mean": None, "gpu_util_max": None, "gpu_mem_mb_max": None}
        utils = [u for u, _ in self.samples]
        mems = [m for _, m in self.samples]
        return {
            "gpu_util_mean": float(np.mean(utils)),
            "gpu_util_max": float(np.max(utils)),
            "gpu_mem_mb_max": float(np.max(mems)),
        }


@torch.no_grad()
def random_beam(tp: TorchGatherPuzzle, n: int, depth: int) -> torch.Tensor:
    """n states from the goal, each after `depth` random moves (no undo check; profiling only)."""
    beam = tp.goal_batch(n)
    for _ in range(depth):
        choice = torch.randint(0, tp.num_moves, (n,), device=beam.device)
        new = beam.clone()
        for m in range(tp.num_moves):
            sel = choice == m
            if bool(sel.any()):
                new[sel] = tp.apply(beam[sel], m)
        beam = new
    return beam


def profile_beam(
    tp: TorchGatherPuzzle,
    cardinality: np.ndarray,
    beam_width: int,
    steps: int = 3,
    chunk: int = 1 << 17,
    depth: int = 20,
    hidden: int = 1024,
    amp: bool = False,
    seed: int = 0,
) -> dict[str, Any]:
    """Time `steps` beam-search steps at a fixed beam width and report cost and utilisation.

    amp=True runs the heuristic in float16 under CUDA autocast (tensor cores).
    """
    device = tp.device
    is_cuda = device.type == "cuda"
    torch.manual_seed(seed)

    onehot = OneHot(cardinality, device)
    model = ProbeHeuristic(onehot.width, hidden=hidden).to(device).eval()
    use_amp = amp and is_cuda

    def score(states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(onehot(states)).float()

    hash_vec = torch.randint(1, 2**62, (tp.dim,), dtype=torch.int64, device=device)
    result: dict[str, Any] = {"beam_width": beam_width, "candidates_per_step": tp.num_moves * beam_width}

    try:
        beam = random_beam(tp, beam_width, depth)
        beam_step(tp, beam, score, hash_vec, chunk=chunk)  # warm-up, not timed
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        timer = CudaStageTimer() if is_cuda else None

        sampler = GpuSampler()
        if is_cuda:
            sampler.start()
        t0 = time.perf_counter()
        for _ in range(steps):
            beam = beam_step(tp, beam, score, hash_vec, chunk=chunk, timer=timer)
        if is_cuda:
            stage_ms = timer.totals_ms()
            torch.cuda.synchronize()
        else:
            stage_ms = {}
        wall_s = time.perf_counter() - t0
        util = sampler.stop() if is_cuda else {}

        result.update(
            status="ok",
            step_ms=1000 * wall_s / steps,
            candidates_per_s=steps * result["candidates_per_step"] / wall_s,
            stage_ms={k: v / steps for k, v in stage_ms.items()},
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if is_cuda else None,
            **util,
        )
    except torch.cuda.OutOfMemoryError:
        result.update(status="OOM", step_ms=None, candidates_per_s=None, stage_ms={}, peak_mem_gb=None)
    finally:
        if is_cuda:
            torch.cuda.empty_cache()
    return result
