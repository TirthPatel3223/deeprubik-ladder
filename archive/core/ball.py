"""Goal ball: every state within `depth` moves of the goal, stored as sorted 64-bit hashes.

Built layer by layer with breadth-first search. Layer k holds the states at distance exactly k.
Each layer is saved to disk as it is built, so an interrupted build resumes where it stopped.
The packed frontier at depth k-1 is kept only as long as it is needed to build layer k.

Memory during a build is dominated by the candidate hashes of one layer (6 x the previous
frontier, 8 bytes each), not by the total ball. For depth 12 on this puzzle that is about 3 GB
of candidate hashes plus the packed frontier at depth 11 (about 3 GB).

Hashes can collide. A collision can drop a state from the ball, or make a non-member look like
a member. Path reconstruction checks every step, and every returned solution is replayed on the
reference simulator, so a collision can cost time but cannot produce a wrong answer.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from core.hashing import hash_states, make_hash_vec
from core.streamed_beam import PackCodec, bits_for_cardinality
from core.torch_engine import TorchGatherPuzzle


class GoalBall:
    def __init__(
        self,
        tp: TorchGatherPuzzle,
        cardinality: np.ndarray,
        directory: str | Path,
        depth: int,
        hash_seed: int = 12345,
        chunk: int = 1 << 20,
    ) -> None:
        self.tp = tp
        self.device = tp.device
        self.dir = Path(directory)
        self.depth = depth
        self.hash_seed = hash_seed
        self.chunk = chunk
        self.hash_vec = make_hash_vec(tp.dim, hash_seed, self.device)
        self._hash_vec_host = self.hash_vec.cpu().numpy()
        self.codec = PackCodec(tp.dim, bits_for_cardinality(cardinality), self.device)
        self.layers: list[np.ndarray] = []
        self._check_metadata()

    # -- files ------------------------------------------------------------------
    def _layer_path(self, k: int) -> Path:
        return self.dir / f"layer_{k:02d}.npy"

    def _frontier_path(self, k: int) -> Path:
        return self.dir / f"frontier_{k:02d}.npy"

    def _check_metadata(self) -> None:
        meta_path = self.dir / "ball_meta.json"
        meta = {"puzzle": self.tp.puzzle.name, "dim": self.tp.dim,
                "hash_seed": self.hash_seed, "bits": self.codec.bits}
        self.dir.mkdir(parents=True, exist_ok=True)
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            if existing != meta:
                raise ValueError(f"ball in {self.dir} was built with different settings: {existing} != {meta}")
        else:
            meta_path.write_text(json.dumps(meta, indent=2))

    # -- building ---------------------------------------------------------------
    def build(self, log: Callable[[str], None] = print) -> None:
        if not self._layer_path(0).exists():
            goal = self.tp.goal[None, :]
            np.save(self._layer_path(0), hash_states(goal, self.hash_vec).cpu().numpy())
            np.save(self._frontier_path(0), self.codec.pack(goal).cpu().numpy())
        for k in range(self.depth):
            done = self._layer_path(k + 1).exists() and (
                k + 1 == self.depth or self._frontier_path(k + 1).exists()
            )
            if done:
                continue
            t0 = time.perf_counter()
            frontier = np.load(self._frontier_path(k), mmap_mode="r")
            visited = np.sort(np.concatenate([np.load(self._layer_path(j)) for j in range(k + 1)]))
            need_states = k + 1 < self.depth
            new_hashes, new_states = self._expand(frontier, visited, need_states)
            np.save(self._layer_path(k + 1), new_hashes)
            if new_states is not None:
                np.save(self._frontier_path(k + 1), new_states)
            log(f"layer {k + 1:2d}: {new_hashes.size:>13,d} states in {time.perf_counter() - t0:7.1f} s")
            del new_hashes, new_states, visited
        self.load()

    def load(self) -> None:
        self.layers = [np.load(self._layer_path(k)) for k in range(self.depth + 1)]

    @torch.no_grad()
    def _expand(self, frontier: np.ndarray, visited: np.ndarray, need_states: bool):
        """Return (new_hashes, new_packed_states) for the states one move beyond `frontier`."""
        n = frontier.shape[0]
        parts = []
        for s in range(0, n, self.chunk):
            block = torch.from_numpy(np.array(frontier[s:s + self.chunk], dtype=np.uint8)).to(self.device)
            states = self.codec.unpack(block)
            for m in range(self.tp.num_moves):
                parts.append(hash_states(self.tp.apply(states, m), self.hash_vec).cpu().numpy())
        cand = np.unique(np.concatenate(parts))
        del parts
        if visited.size:
            idx = np.minimum(np.searchsorted(visited, cand), visited.size - 1)
            cand = cand[visited[idx] != cand]
        if not need_states:
            return cand, None

        # Second pass: store one packed state per new hash, aligned with the sorted hashes.
        nh = torch.from_numpy(cand).to(self.device)
        out = torch.empty((cand.size, self.codec.nbytes), dtype=torch.uint8, device=self.device)
        filled = torch.zeros(cand.size, dtype=torch.bool, device=self.device)
        for s in range(0, n, self.chunk):
            block = torch.from_numpy(np.array(frontier[s:s + self.chunk], dtype=np.uint8)).to(self.device)
            states = self.codec.unpack(block)
            for m in range(self.tp.num_moves):
                cstates = self.tp.apply(states, m)
                h = hash_states(cstates, self.hash_vec)
                pos = torch.searchsorted(nh, h).clamp_(max=nh.numel() - 1)
                ok = (nh[pos] == h) & ~filled[pos]
                sel = ok.nonzero(as_tuple=True)[0]
                if sel.numel():
                    p = pos[sel]
                    out[p] = self.codec.pack(cstates[sel])
                    filled[p] = True
        if not bool(filled.all()):
            raise RuntimeError("some layer hashes had no state in the frontier (should not happen)")
        return cand, out.cpu().numpy()

    # -- queries ----------------------------------------------------------------
    def _host_hash(self, state: np.ndarray) -> np.int64:
        return np.int64((state.astype(np.int64) * self._hash_vec_host).sum())

    @staticmethod
    def _in_sorted(arr: np.ndarray, h: np.int64) -> bool:
        i = np.searchsorted(arr, h)
        return bool(i < arr.size and arr[i] == h)

    def depth_of(self, state: np.ndarray) -> int | None:
        """Exact distance from the goal if the state is in the ball, else None."""
        h = self._host_hash(np.asarray(state, dtype=np.uint8))
        for k, layer in enumerate(self.layers):
            if self._in_sorted(layer, h):
                return k
        return None

    def path_to_goal(self, state: np.ndarray) -> list[int] | None:
        """A shortest move list from `state` to the goal, or None if the state is outside the ball."""
        cur = np.asarray(state, dtype=np.uint8)
        k = self.depth_of(cur)
        if k is None:
            return None
        path: list[int] = []
        while k > 0:
            for m in range(self.tp.num_moves):
                nxt = self.tp.puzzle.apply(cur[None, :], m)[0]
                if self._in_sorted(self.layers[k - 1], self._host_hash(nxt)):
                    path.append(m)
                    cur = nxt
                    k -= 1
                    break
            else:
                raise RuntimeError("no neighbour one step closer to the goal (hash collision in the ball)")
        return path

    def device_hashes(self) -> torch.Tensor:
        """All hashes in the ball, sorted, on the puzzle's device (for membership tests in the beam search)."""
        all_hashes = np.sort(np.concatenate(self.layers))
        return torch.from_numpy(all_hashes).to(self.device)

    def device_hashes_with_depth(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Sorted hashes and their exact depths (uint8), on the puzzle's device, for training labels."""
        hashes = np.concatenate(self.layers)
        depths = np.concatenate([np.full(layer.size, k, dtype=np.uint8) for k, layer in enumerate(self.layers)])
        order = np.argsort(hashes, kind="stable")
        return (torch.from_numpy(hashes[order]).to(self.device),
                torch.from_numpy(depths[order]).to(self.device))

    def sizes(self) -> list[int]:
        return [int(layer.size) for layer in self.layers]
