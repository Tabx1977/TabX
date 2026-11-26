#!/usr/bin/env python3
"""What is each piece worth to the engine?

    tools/lichess/venv/bin/python scripts/piece_values.py target/release/tabx [depth]

The evaluation is a neural network: nowhere does it say that a rook is worth
five pawns. This removes one white piece at a time from a few quiet positions
and looks at how far the engine's search score falls, which shows the values
the network has learned. Needs the python-chess package.
"""
import subprocess
import sys

import chess

engine_path, depth = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 14
BASES = {
    "start position": chess.STARTING_FEN,
    "Italian, both sides developed": "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 7",
    "Queen's Gambit, both sides developed": "r1bq1rk1/pp2bppp/2n1pn2/2pp4/2PP4/2N1PN2/PP2BPPP/R1BQ1RK1 w - - 0 8",
    "open middlegame, queens on": "r2q1rk1/pp2bppp/2n1bn2/3p4/3N4/2N1B1P1/PP2PPBP/R2Q1RK1 w - - 0 11",
}
engine = subprocess.Popen([engine_path], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)


def send(line):
    engine.stdin.write(line + "\n")
    engine.stdin.flush()


def score(fen):
    send("ucinewgame")
    send(f"position fen {fen}")
    send(f"go depth {depth}")
    value = None
    while True:
        line = engine.stdout.readline()
        words = line.split()
        if "score" in words and words[words.index("score") + 1] == "cp":
            value = int(words[words.index("score") + 2])
        if line.startswith("bestmove"):
            return value


send("uci")
while not engine.stdout.readline().startswith("uciok"):
    pass
names = {chess.PAWN: "pawn", chess.KNIGHT: "knight", chess.BISHOP: "bishop", chess.ROOK: "rook", chess.QUEEN: "queen"}
totals = {t: [] for t in names}
for label, fen in BASES.items():
    board = chess.Board(fen)
    base = score(board.fen())
    row = []
    for piece_type in names:
        drops = []
        for square in board.pieces(piece_type, chess.WHITE):
            copy = board.copy()
            copy.remove_piece_at(square)
            copy.castling_rights = copy.clean_castling_rights()
            if copy.is_valid():
                drops.append(base - score(copy.fen()))
        if drops:
            mean = sum(drops) / len(drops)
            totals[piece_type].append(mean)
            row.append(f"{names[piece_type]} {mean:.0f}")
    print(f"{label} (score {base:+d}): " + ", ".join(row))
pawn = sum(totals[chess.PAWN]) / len(totals[chess.PAWN])
print("average, in the engine's own centipawns and in pawns:")
for piece_type, name in names.items():
    mean = sum(totals[piece_type]) / len(totals[piece_type])
    print(f"    {name:7} {mean:6.0f}   = {mean / pawn:4.1f} pawns")
send("quit")
