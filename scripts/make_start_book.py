#!/usr/bin/env python3
"""Makes a book of very early positions from a PGN opening book.

    tools/lichess/venv/bin/python scripts/make_start_book.py tools/books/8moves_v3.pgn tools/books/start_6ply.epd

Every game in the PGN is cut off after --plies plies (default 6) and each
distinct position reached is written once, as a full FEN with its move number.

Why: the usual test book starts games around move 10 and its positions carry
no move number, so the engine believes it is on move 1 of a game that is
already in the middlegame. Time management depends on the move number and on
how the opening is played, so tests of it start from these positions instead:
close to the real start of a game, as on Lichess, but varied enough that the
games differ.

Needs the python-chess package, which the Lichess setup installs.
"""
import argparse

import chess.pgn


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pgn", help="opening book in PGN format")
    parser.add_argument("out", help="file to write, one FEN per line")
    parser.add_argument("--plies", type=int, default=6, help="plies to play from the start of each game [6]")
    args = parser.parse_args()

    positions = {}
    games = 0
    with open(args.pgn, errors="replace") as stream:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            if "FEN" in game.headers:
                continue
            board = game.board()
            moves = list(game.mainline_moves())[: args.plies]
            if len(moves) < args.plies:
                continue
            for move in moves:
                board.push(move)
            games += 1
            # Keyed without the move counters, so transpositions count once.
            positions.setdefault(board.epd(), board.fen())
    with open(args.out, "w") as out:
        for fen in sorted(positions.values()):
            out.write(fen + "\n")
    print(f"{games} games, {len(positions)} distinct positions after {args.plies} plies -> {args.out}")


if __name__ == "__main__":
    main()
