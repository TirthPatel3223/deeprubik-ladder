"""Measure the streamed beam search: step time, candidates/s, peak memory, GPU utilisation."""
from __future__ import annotations

import time
from dataclasses import fields
from typing import Any

import numpy as np
import torch

from core.beam import OneHot
from core.profiling import GpuSampler, ProbeHeuristic, random_beam
from core.streamed_beam import Run, StreamedBeam
from core.torch_engine import TorchGatherPuzzle


def random_run(beam: StreamedBeam, tp: TorchGatherPuzzle, n: int, depth: int, chunk: int = 1 << 20) -> Run:
    """A starting run of n random states, built chunk by chunk so setup never holds all states at once."""
    parts = []
    for s in range(0, n, chunk):
        parts.append(beam.run_from_states(random_beam(tp, min(chunk, n - s), depth)))
    return Run(**{f.name: torch.cat([getattr(p, f.name) for p in parts]) for f in fields(Run)})


def profile_streamed(
    tp: TorchGatherPuzzle,
    cardinality: np.ndarray,
    width: int,
    steps: int = 2,
    beam_chunk: int = 1 << 17,
    score_chunk: int = 1 << 16,
    hidden: int = 1024,
    amp: bool = True,
    depth: int = 20,
    seed: int = 0,
) -> dict[str, Any]:
    device = tp.device
    is_cuda = device.type == "cuda"
    torch.manual_seed(seed)
    use_amp = amp and is_cuda

    onehot = OneHot(cardinality, device)
    model = ProbeHeuristic(onehot.width, hidden=hidden).to(device).eval()

    def score(states: torch.Tensor) -> torch.Tensor:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            return model(onehot(states)).float()

    hash_vec = torch.randint(1, 2**62, (tp.dim,), dtype=torch.int64, device=device)
    result: dict[str, Any] = {"beam_width": width, "candidates_per_step": 5 * width}
    try:
        beam = StreamedBeam(tp, cardinality, score, hash_vec, width, beam_chunk=beam_chunk, score_chunk=score_chunk)
        run = random_run(beam, tp, width, depth)
        run, _ = beam.step(run)  # warm-up, not timed
        if is_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        sampler = GpuSampler()
        if is_cuda:
            sampler.start()
        t0 = time.perf_counter()
        for _ in range(steps):
            run, _ = beam.step(run)
        if is_cuda:
            torch.cuda.synchronize()
        wall_s = time.perf_counter() - t0
        util = sampler.stop() if is_cuda else {}
        result.update(
            status="ok",
            step_ms=1000 * wall_s / steps,
            candidates_per_s=steps * result["candidates_per_step"] / wall_s,
            kept=int(run.keys.shape[0]),
            peak_mem_gb=(torch.cuda.max_memory_allocated() / 2**30) if is_cuda else None,
            **util,
        )
    except torch.cuda.OutOfMemoryError:
        result.update(status="OOM", step_ms=None, candidates_per_s=None, peak_mem_gb=None)
    finally:
        if is_cuda:
            torch.cuda.empty_cache()
    return result
