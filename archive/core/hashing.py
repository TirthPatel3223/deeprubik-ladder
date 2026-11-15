"""Deterministic 64-bit state hashing, shared by the beam search and the goal ball.

Both must use the same hash vector, or the ball lookups silently miss. The seed is
stored with the ball, and the beam search refuses a ball built with a different one.
"""
from __future__ import annotations

import torch


def make_hash_vec(dim: int, seed: int = 12345, device: str | torch.device = "cpu") -> torch.Tensor:
    gen = torch.Generator().manual_seed(seed)
    vec = torch.randint(1, 2**62, (dim,), dtype=torch.int64, generator=gen)
    return vec.to(device)


@torch.no_grad()
def hash_states(states: torch.Tensor, hash_vec: torch.Tensor, chunk: int = 1 << 15) -> torch.Tensor:
    """Hash each row of an (N, D) uint8 tensor to one int64. Collisions are possible but rare."""
    n = states.shape[0]
    out = torch.empty(n, dtype=torch.int64, device=states.device)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        out[s:e] = (states[s:e].to(torch.int64) * hash_vec).sum(1)
    return out
