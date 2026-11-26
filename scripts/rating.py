#!/usr/bin/env python3
"""Fits one rating to match results against opponents of known rating.

    scripts/rating.py FILE        (or pipe the lines in)

Each input line is "opponent_rating wins losses draws", counted from the
point of view of the engine being rated. Prints the results per opponent and
the single rating that best explains all of them under the usual Elo model
(expected score = 1 / (1 + 10^((opponent - rating) / 400))).

The interval printed is a 95% confidence interval for the luck of the games
only. It says nothing about how good the opponents' ratings are.
"""
import math
import sys

ELO_PER_LOGIT = 400.0 / math.log(10.0)


def expected(rating, opponent):
    return 1.0 / (1.0 + 10.0 ** ((opponent - rating) / 400.0))


def fit(rows):
    """Returns (rating, half-width of the 95% interval), or None if every game had the same winner."""
    games = sum(w + l + d for _, w, l, d in rows)
    points = sum(w + d / 2 for _, w, l, d in rows)
    if points <= 0 or points >= games:
        return None

    # The total score minus the total expected score falls as the rating
    # rises, so the rating where they are equal can be found by bisection.
    lo, hi = -4000.0, 8000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        surplus = sum((w + d / 2) - (w + l + d) * expected(mid, opp) for opp, w, l, d in rows)
        lo, hi = (mid, hi) if surplus > 0 else (lo, mid)
    rating = (lo + hi) / 2

    # Error propagation: the variance of the total score, measured from the
    # games themselves, divided by the squared slope of the expected total.
    variance = 0.0
    slope = 0.0
    for opp, w, l, d in rows:
        n = w + l + d
        mean = (w + d / 2) / n
        variance += w * (1 - mean) ** 2 + d * (0.5 - mean) ** 2 + l * mean ** 2
        e = expected(rating, opp)
        slope += n * e * (1 - e) / ELO_PER_LOGIT
    return rating, 1.96 * math.sqrt(variance) / slope


def main():
    source = open(sys.argv[1]) if len(sys.argv) > 1 else sys.stdin
    rows = []
    for line in source:
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) != 4:
            sys.exit(f"expected 'opponent_rating wins losses draws', got: {line.strip()}")
        opp, w, l, d = map(int, parts)
        if w + l + d > 0:
            rows.append((opp, w, l, d))
    if not rows:
        sys.exit("no results to rate")

    print("opponent  games    W    L    D   score   rating implied by this row")
    for opp, w, l, d in rows:
        n = w + l + d
        score = (w + d / 2) / n
        implied = f"{opp + ELO_PER_LOGIT * math.log(score / (1 - score)):.0f}" if 0 < score < 1 else "out of range"
        print(f"{opp:8d}  {n:5d}  {w:3d}  {l:3d}  {d:3d}  {100 * score:5.1f}%   {implied}")

    result = fit(rows)
    if result is None:
        side = "above" if sum(w for _, w, _, _ in rows) > 0 else "below"
        print(f"estimated rating: {side} what these opponents can measure (every game had the same result)")
    else:
        rating, margin = result
        print(f"estimated rating: {rating:.0f} +/- {margin:.0f}")


if __name__ == "__main__":
    main()
