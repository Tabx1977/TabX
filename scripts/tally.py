#!/usr/bin/env python3
"""Summarise a fastchess PGN from an SPRT run: scripts/tally.py [games.pgn]
With no argument, uses the most recent run under target/sprt/."""
import glob, math, os, re, sys

path = sys.argv[1] if len(sys.argv) > 1 else max(glob.glob("target/sprt/run-*/games.pgn"), key=os.path.getmtime)
txt = open(path).read()
games = re.findall(r'\[Round "([^"]+)"\]\n\[White "([^"]+)"\]\n\[Black "([^"]+)"\]\n\[Result "([^"]+)"\]', txt)
w = d = l = 0
pairs = {}
for rnd, white, black, res in games:
    s = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(res)
    if s is None:
        continue
    if white != "new":
        s = 1 - s
    w += s == 1
    d += s == 0.5
    l += s == 0
    pairs.setdefault(rnd.split(".")[0], []).append(s)
n = w + d + l
term = {}
for t in re.findall(r'\[Termination "([^"]+)"\]', txt):
    term[t] = term.get(t, 0) + 1
print(f"{path}\ngames {n}: new +{w} ={d} -{l}  terminations {term}")
if n:
    p = (w + d / 2) / n
    if 0 < p < 1:
        print(f"score {p:.3f}  approx Elo {-400 * math.log10(1 / p - 1):+.0f}")
