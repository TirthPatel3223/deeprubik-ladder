"""Memory-bounded beam search.

Instead of holding all candidates (W * moves) at once, this keeps only the best
`width` states, stored packed, and merges candidates in chunks. Peak memory is
about width * (packed size + 17 bytes) plus one chunk of work, independent of
the branching factor.

Each beam entry records (parent index, move) so a solution path can be rebuilt.
Hash collisions can only drop a candidate, never produce a wrong answer, because
every returned path is replayed on the reference simulator before it is trusted.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from core.hashing import hash_states
from core.torch_engine import TorchGatherPuzzle

ScoreFn = Callable[[torch.Tensor], torch.Tensor]
ROOT_MOVE = 255  # move marker for the start state (never equal to a real move index)


class PackCodec:
    """Packs (N, dim) uint8 states into (N, nbytes) uint8 using `bits` bits per coordinate."""

    def __init__(self, dim: int, bits: int, device: torch.device) -> None:
        self.dim = dim
        self.bits = bits
        self.nbytes = (dim * bits + 7) // 8
        self.pad = self.nbytes * 8 - dim * bits
        self.device = device
        self._shifts = torch.arange(bits - 1, -1, -1, dtype=torch.uint8, device=device)

    @torch.no_grad()
    def pack(self, states: torch.Tensor) -> torch.Tensor:
        n = states.shape[0]
        bits = (states.unsqueeze(-1) >> self._shifts) & 1          # (n, dim, bits)
        bits = bits.reshape(n, self.dim * self.bits)
        if self.pad:
            bits = torch.cat([bits, torch.zeros((n, self.pad), dtype=torch.uint8, device=self.device)], 1)
        bits = bits.reshape(n, self.nbytes, 8)
        out = torch.zeros((n, self.nbytes), dtype=torch.uint8, device=self.device)
        for j in range(8):
            out |= bits[:, :, j] << (7 - j)
        return out

    @torch.no_grad()
    def unpack(self, packed: torch.Tensor) -> torch.Tensor:
        n = packed.shape[0]
        bits = torch.empty((n, self.nbytes, 8), dtype=torch.uint8, device=self.device)
        for j in range(8):
            bits[:, :, j] = (packed >> (7 - j)) & 1
        bits = bits.reshape(n, self.nbytes * 8)[:, : self.dim * self.bits]
        bits = bits.reshape(n, self.dim, self.bits)
        out = torch.zeros((n, self.dim), dtype=torch.uint8, device=self.device)
        for k in range(self.bits):
            out |= bits[:, :, k] << (self.bits - 1 - k)
        return out


def bits_for_cardinality(cardinality: np.ndarray) -> int:
    """Bits per coordinate needed to hold values 0 .. max(cardinality)-1."""
    return max(1, int(int(np.max(cardinality)) - 1).bit_length())


@dataclass
class Run:
    """One beam: every array has length = number of kept states."""
    packed: torch.Tensor   # (n, nbytes) uint8
    keys: torch.Tensor     # (n,) int64, dedup key = state hash * 256 + last move
    scores: torch.Tensor   # (n,) float32, lower is better
    parent: torch.Tensor   # (n,) int32, index into the previous run (-1 for the root)
    move: torch.Tensor     # (n,) uint8, move that produced the entry (ROOT_MOVE for the root)


class StreamedBeam:
    def __init__(
        self,
        tp: TorchGatherPuzzle,
        cardinality: np.ndarray,
        score_fn: ScoreFn,
        hash_vec: torch.Tensor,
        width: int,
        beam_chunk: int = 1 << 17,
        score_chunk: int = 1 << 16,
        no_undo: bool = True,
        ball=None,
    ) -> None:
        """cardinality[i] = number of values coordinate i takes; it sets the packed bit width.

        ball: optional GoalBall. A beam state found in the ball is finished exactly from the ball.
        """
        if ball is not None and not torch.equal(ball.hash_vec.to(hash_vec.device), hash_vec):
            raise ValueError("hash vector differs from the one the goal ball was built with")
        self.tp = tp
        self.device = tp.device
        self.score_fn = score_fn
        self.hash_vec = hash_vec
        self.width = width
        self.beam_chunk = beam_chunk
        self.score_chunk = score_chunk
        self.no_undo = no_undo
        self.ball = ball
        self._ball_hashes = ball.device_hashes() if ball is not None else None
        self.codec = PackCodec(tp.dim, bits_for_cardinality(cardinality), self.device)
        self._inverse = [int(x) for x in tp.inverse.tolist()]
        self.goal_key = int(hash_states(tp.goal[None, :], hash_vec)[0])

    # -- helpers ----------------------------------------------------------------
    @torch.no_grad()
    def _score(self, states: torch.Tensor) -> torch.Tensor:
        n = states.shape[0]
        out = torch.empty(n, dtype=torch.float32, device=self.device)
        for s in range(0, n, self.score_chunk):
            e = min(s + self.score_chunk, n)
            out[s:e] = self.score_fn(states[s:e]).reshape(-1).float()
        return out

    def _empty(self) -> Run:
        nb = self.codec.nbytes
        d = self.device
        return Run(
            packed=torch.empty((0, nb), dtype=torch.uint8, device=d),
            keys=torch.empty(0, dtype=torch.int64, device=d),
            scores=torch.empty(0, dtype=torch.float32, device=d),
            parent=torch.empty(0, dtype=torch.int32, device=d),
            move=torch.empty(0, dtype=torch.uint8, device=d),
        )

    @torch.no_grad()
    def run_from_states(self, states: torch.Tensor) -> Run:
        """A root run from explicit states (parents set to -1, move to ROOT_MOVE)."""
        states = states.to(self.device)
        n = states.shape[0]
        return Run(
            packed=self.codec.pack(states),
            keys=hash_states(states, self.hash_vec) * 256 + ROOT_MOVE,
            scores=self._score(states),
            parent=torch.full((n,), -1, dtype=torch.int32, device=self.device),
            move=torch.full((n,), ROOT_MOVE, dtype=torch.uint8, device=self.device),
        )

    @torch.no_grad()
    def _merge(self, cur: Run, keys, scores, packed, parent, move) -> Run:
        """Combine the kept beam with new candidates, drop duplicate states, keep the best `width`."""
        all_keys = torch.cat([cur.keys, keys])
        all_scores = torch.cat([cur.scores, scores])
        order = torch.argsort(all_keys)
        sorted_keys = all_keys[order]
        first = torch.ones(sorted_keys.shape[0], dtype=torch.bool, device=self.device)
        first[1:] = sorted_keys[1:] != sorted_keys[:-1]
        unique = order[first]
        k = min(self.width, unique.numel())
        top = torch.topk(all_scores[unique], k, largest=False).indices
        sel = unique[top]
        all_packed = torch.cat([cur.packed, packed])
        all_parent = torch.cat([cur.parent, parent])
        all_move = torch.cat([cur.move, move])
        return Run(all_packed[sel], all_keys[sel], all_scores[sel], all_parent[sel], all_move[sel])

    def _hit(self, cand: torch.Tensor, parent: torch.Tensor, move: torch.Tensor, i: int) -> dict:
        return {"parent": int(parent[i]), "move": int(move[i]), "state": cand[i].cpu().numpy()}

    # -- one step ---------------------------------------------------------------
    @torch.no_grad()
    def step(self, prev: Run):
        """Expand `prev` by one move.

        Returns (next_run, hit). hit is None, or a dict with the parent index in `prev`,
        the move that reached the hit state, and that state. The hit state is the goal
        or a state inside the goal ball.
        """
        nxt = self._empty()
        n_prev = prev.keys.shape[0]
        for s in range(0, n_prev, self.beam_chunk):
            e = min(s + self.beam_chunk, n_prev)
            states = self.codec.unpack(prev.packed[s:e])
            last = prev.move[s:e]
            cand_parts, parent_parts, move_parts = [], [], []
            for m in range(self.tp.num_moves):
                if self.no_undo:
                    idx = (last != self._inverse[m]).nonzero(as_tuple=True)[0]
                else:
                    idx = torch.arange(e - s, device=self.device)
                if idx.numel() == 0:
                    continue
                cand_parts.append(self.tp.apply(states[idx], m))
                parent_parts.append((idx + s).to(torch.int32))
                move_parts.append(torch.full((idx.numel(),), m, dtype=torch.uint8, device=self.device))
            if not cand_parts:
                continue
            cand = torch.cat(cand_parts)
            parent = torch.cat(parent_parts)
            move = torch.cat(move_parts)
            state_keys = hash_states(cand, self.hash_vec)

            hits = (state_keys == self.goal_key).nonzero(as_tuple=True)[0]
            if hits.numel():
                return nxt, self._hit(cand, parent, move, int(hits[0]))
            if self._ball_hashes is not None:
                pos = torch.searchsorted(self._ball_hashes, state_keys).clamp_(max=self._ball_hashes.numel() - 1)
                inball = (self._ball_hashes[pos] == state_keys).nonzero(as_tuple=True)[0]
                if inball.numel():
                    return nxt, self._hit(cand, parent, move, int(inball[0]))

            # Deduplicate on (state, last move): with the no-undo rule, the same state
            # reached by different last moves can have different legal continuations.
            keys = state_keys * 256 + move.to(torch.int64)
            scores = self._score(cand)
            packed = self.codec.pack(cand)
            nxt = self._merge(nxt, keys, scores, packed, parent, move)
        return nxt, None

    # -- search -----------------------------------------------------------------
    def solve(self, start: np.ndarray, max_steps: int) -> list[int] | None:
        """Return a move list from `start` to the goal, or None. The path is replay-checked."""
        start_t = torch.as_tensor(np.asarray(start, dtype=np.uint8)[None, :], device=self.device)
        run = self.run_from_states(start_t)
        history: list[tuple[np.ndarray, np.ndarray]] = []  # history[j-1] = (parent, move) of run j
        for t in range(1, max_steps + 1):
            nxt, hit = self.step(run)
            if hit is not None:
                moves = _prefix(history, t - 1, hit["parent"])
                moves.append(hit["move"])
                if np.array_equal(hit["state"], self.tp.puzzle.goal):
                    tail = []
                else:
                    tail = self.ball.path_to_goal(hit["state"])
                    if tail is None:
                        raise RuntimeError("hit state is not in the goal ball (bug or hash collision)")
                moves += tail
                if not _replays_to_goal(self.tp, start, moves):
                    raise RuntimeError("reconstructed path does not reach the goal (bug or hash collision)")
                return moves
            history.append((nxt.parent.cpu().numpy(), nxt.move.cpu().numpy()))
            run = nxt
            if run.keys.shape[0] == 0:
                return None
        return None


def _prefix(history: list[tuple[np.ndarray, np.ndarray]], t_prev: int, idx: int) -> list[int]:
    """Moves from the root to entry `idx` of run `t_prev`. history[j-1] holds (parent, move) of run j."""
    moves: list[int] = []
    for j in range(t_prev, 0, -1):
        parents, moves_arr = history[j - 1]
        moves.append(int(moves_arr[idx]))
        idx = int(parents[idx])
    moves.reverse()
    return moves


def _replays_to_goal(tp: TorchGatherPuzzle, start: np.ndarray, moves: list[int]) -> bool:
    """Check the path on the NumPy reference simulator, independent of the GPU code."""
    reference = tp.puzzle
    state = reference.apply_word(np.asarray(start, dtype=np.uint8), moves)
    return bool(np.array_equal(state, reference.goal))
