"""Numerically check the one-point commutator lemma and the conjugate-relabel rule.

Lemma (derived in docs/piece_isolation_algorithms.md): if supp(A) and supp(B)
meet in exactly one point x, then the commutator C = A B A^-1 B^-1 is the
3-cycle  x -> A(x) -> B(x) -> x  and moves exactly those three points.

Conjugate-relabel rule: for a cycle t = (x y z) and any permutation s,
s t s^-1 = (s(x) s(y) s(z)).

Convention: a permutation is a tuple p with p[i] = image of i.
compose(f, g)(i) = f[g[i]]  (apply g first, then f).

Usage: python scripts/verify_one_point_commutator.py
Exit code 0 if every sampled case passes, 1 otherwise. Standard library only.
"""

import random
import sys


def compose(f, g):
    return tuple(f[g[i]] for i in range(len(f)))


def inverse(p):
    inv = [0] * len(p)
    for i, j in enumerate(p):
        inv[j] = i
    return tuple(inv)


def support(p):
    return {i for i, j in enumerate(p) if i != j}


def commutator(a, b):
    return compose(compose(compose(a, b), inverse(a)), inverse(b))


def cycle_perm(cycle, n):
    p = list(range(n))
    for i, x in enumerate(cycle):
        p[x] = cycle[(i + 1) % len(cycle)]
    return tuple(p)


def random_perm_on(points, n, rng):
    pts = list(points)
    img = pts[:]
    rng.shuffle(img)
    p = list(range(n))
    for s, t in zip(pts, img):
        p[s] = t
    return tuple(p)


def check_one_point_lemma(trials, n, seed):
    rng = random.Random(seed)
    tested = 0
    failures = []
    while tested < trials:
        a = random_perm_on(rng.sample(range(n), rng.randint(2, n)), n, rng)
        b = random_perm_on(rng.sample(range(n), rng.randint(2, n)), n, rng)
        common = support(a) & support(b)
        if len(common) != 1:
            continue
        (x,) = common
        c = commutator(a, b)
        expected = {x, a[x], b[x]}
        ok = (
            c[x] == a[x]
            and c[a[x]] == b[x]
            and c[b[x]] == x
            and support(c) == expected
        )
        tested += 1
        if not ok:
            failures.append((a, b, c))
    return tested, failures


def check_conjugate_relabel(trials, n, seed):
    rng = random.Random(seed)
    failures = 0
    for _ in range(trials):
        x, y, z = rng.sample(range(n), 3)
        t = cycle_perm([x, y, z], n)
        s = random_perm_on(range(n), n, rng)
        lhs = compose(compose(s, t), inverse(s))
        rhs = cycle_perm([s[x], s[y], s[z]], n)
        if lhs != rhs:
            failures += 1
    return trials, failures


def main():
    # Worked example from the notes (0-indexed): A = (0 1 2), B = (2 3 4).
    # Shared point x = 2, A(2) = 0, B(2) = 3, so C should be 2 -> 0 -> 3 -> 2.
    a = cycle_perm([0, 1, 2], 5)
    b = cycle_perm([2, 3, 4], 5)
    c = commutator(a, b)
    print("worked example: C maps 2->%d, 0->%d, 3->%d, support=%s"
          % (c[2], c[0], c[3], sorted(support(c))))

    total_fail = 0
    for n, seed in [(5, 1), (8, 2), (12, 3)]:
        tested, failures = check_one_point_lemma(20000, n, seed)
        print("one-point lemma, n=%d: tested=%d failures=%d" % (n, tested, len(failures)))
        total_fail += len(failures)
        if failures:
            a_, b_, c_ = failures[0]
            print("  first failure: A=%s B=%s C=%s" % (a_, b_, c_))

    tested, failures = check_conjugate_relabel(20000, 9, 4)
    print("conjugate relabel, n=9: tested=%d failures=%d" % (tested, failures))
    total_fail += failures

    return 0 if total_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
