"""Which beam searches a learned rung tries, and the timeout they share. No torch needed here, so modules
that only use exact tables can import it."""
from __future__ import annotations

from dataclasses import dataclass


class SearchTimeout(Exception):
    """The deadline passed before the search found a solution."""


@dataclass(frozen=True)
class SearchSchedule:
    """Beam searches to try in order until one reaches the goal: (width, weight on the cost so far)."""
    attempts: tuple[tuple[int, float], ...]

    @staticmethod
    def standard(width: int = 4096, weight: float = 0.3, max_width: int = 32768,
                 max_attempts: int = 3) -> "SearchSchedule":
        """``width``, then 8x wider each time (at most ``max_attempts`` widths), never beyond ``max_width``.
        The default is 4096, then 32768."""
        widths, w = [], int(width)
        while len(widths) < max_attempts:
            widths.append(min(w, max_width))
            if w >= max_width:
                break
            w *= 8
        return SearchSchedule(tuple((w, weight) for w in dict.fromkeys(widths)))

    @staticmethod
    def extended(width: int = 32768, weight: float = 0.3, max_width: int = 131072,
                 greedy_weight: float = 0.1) -> "SearchSchedule":
        """A second chance after the standard schedule has failed: a greedier search at the standard
        maximum width, then the widest beam, then the widest beam greedier still."""
        return SearchSchedule(tuple(dict.fromkeys(((width, greedy_weight), (max_width, weight),
                                                   (max_width, greedy_weight)))))
