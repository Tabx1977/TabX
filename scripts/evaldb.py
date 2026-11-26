#!/usr/bin/env python3
"""Looks positions up in an index of Lichess's database of Stockfish evaluations.

    tools/lichess/venv/bin/python scripts/evaldb.py "e4 c6 d4 d5 e5 Bf5"
    tools/lichess/venv/bin/python scripts/evaldb.py "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"

For each position (moves from the start position, or a FEN) it prints what
the database has: Stockfish's score, the depth it searched to, its move, and
the same for the position after every legal move, best first.

The database (https://database.lichess.org/#evals, 22.5 GB) has one line of
text per position, 418 million of them, in no order: finding a position
means reading all of it, five minutes at best. The index is made from it
once,

    zstd -dc tools/lichess-evals/lichess_db_eval.jsonl.zst | target/release/tabx index-evals

and has one record of 16 bytes per position, sorted by a hash of the
position (6.7 GB, in tools/lichess-evals/evals.idx), so that a lookup is a
bisection. The layout of a record is described in src/evalimport.rs.

Other scripts use this file as a module: database_choice below picks the
move that Stockfish's analysis in the database supports, for an opening
book or for a player that has none for the position. Needs the python-chess
package.
"""
import collections
import mmap
import struct
import sys

import chess

DEFAULT_INDEX = "tools/lichess-evals/evals.idx"
RECORD = struct.Struct("<QhHBBBx")
MATE = 32000        # a forced mate in n moves is stored as MATE - n
MATE_FROM = 31000   # scores from here on are forced mates

# cp: centipawns for White, or a forced mate (see MATE); move: Stockfish's move, as python-chess has it.
Eval = collections.namedtuple("Eval", "cp depth move evals lines")


def key_of(epd):
    """The index's key of a position: FNV-1a over the first four fields of its FEN."""
    value = 0xCBF29CE484222325
    for byte in epd.encode():
        value = ((value ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return value


class EvalIndex:
    def __init__(self, path=DEFAULT_INDEX, remember=2_000_000):
        self.file = open(path, "rb")
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        self.count = len(self.map) // RECORD.size
        self.cache = {}
        self.remember = remember    # lookups kept in memory before they are forgotten

    def __len__(self):
        return self.count

    def key_at(self, index):
        return struct.unpack_from("<Q", self.map, index * RECORD.size)[0]

    def find(self, epd):
        """The raw record of a position (score, move as 16 bits, depth, evaluations, lines) or None."""
        key = key_of(epd)
        # The keys are spread evenly, so the record is close to this place.
        guess = key * self.count >> 64
        low, high = max(0, guess - 65536), min(self.count, guess + 65536)
        if low > 0 and self.key_at(low) > key:
            low = 0
        if high < self.count and self.key_at(high - 1) < key:
            high = self.count
        while low < high:
            middle = (low + high) // 2
            if self.key_at(middle) < key:
                low = middle + 1
            else:
                high = middle
        if low == self.count:
            return None
        found, cp, move, depth, evals, lines = RECORD.unpack_from(self.map, low * RECORD.size)
        return (cp, move, depth, evals, lines) if found == key else None

    def lookup(self, board):
        """What the database has of a position, or None. Results are remembered."""
        epd = board.epd()
        if epd in self.cache:
            return self.cache[epd]
        raw = self.find(epd)
        result = None
        if raw is not None:
            cp, packed, depth, evals, lines = raw
            result = Eval(cp, depth, unpack_move(board, packed), evals, lines)
        if len(self.cache) >= self.remember:
            self.cache.clear()
        self.cache[epd] = result
        return result

    def children(self, board):
        """[(move, Eval)] for every legal move into a position the database has."""
        found = []
        for move in list(board.legal_moves):
            board.push(move)
            entry = self.lookup(board)
            board.pop()
            if entry is not None:
                found.append((move, entry))
        return found


def unpack_move(board, packed):
    """A move of the index as a python-chess move. The database may write castling as
    the king taking its own rook. None if the move is not legal in the position."""
    from_square, to_square, promotion = packed & 63, packed >> 6 & 63, packed >> 12
    piece = board.piece_at(from_square)
    target = board.piece_at(to_square)
    if (piece is not None and target is not None and piece.piece_type == chess.KING
            and target.piece_type == chess.ROOK and target.color == piece.color):
        file = 6 if chess.square_file(to_square) > chess.square_file(from_square) else 2
        to_square = chess.square(file, chess.square_rank(from_square))
    move = chess.Move(from_square, to_square, promotion + 1 if promotion else None)
    return move if move in board.legal_moves else None


def for_side(cp, white):
    """A score for White as a score for one side."""
    return cp if white else -cp


def database_choice(board, index, min_depth=24, window=30, noise=8, avoided=()):
    """The move that Stockfish's analysis in the database supports in a position, and its
    score for the side to move; None if the database says too little.

    The database has the position itself, with Stockfish's move and score, and the
    positions after some of the legal moves, each with a score of its own from a search
    that began one move later. Such searches differ in depth and never agree to the
    centipawn, and the rule is that the search that looked furthest along a move speaks
    for it. That is normally the search of the position after the move: on Lichess the
    bot's 14th move in one line was Stockfish's choice with -0.25 by a depth 26 search of
    the position, and -1.13 by the search of the position after it. For Stockfish's own
    choice it is the search of the position itself where that went deeper. The moves that
    come within `window` centipawns of the best are then looked at one move further, in
    the same way: of the replies the database has searched at least as far, the one that
    is worst for the side to move counts (together with the move's own score if
    Stockfish's reply is not among them). Stockfish's choice in the position itself is
    played if it comes within `noise` of the best. Evaluations shallower than `min_depth`
    are not used, nor are the moves in `avoided` (UCI notation).

    The board comes back as it was."""
    white = board.turn == chess.WHITE
    own = index.lookup(board)
    if own is not None and (own.depth < min_depth or own.move is None or own.move.uci() in avoided):
        own = None
    candidates = []     # (score, move, what the database has after it, plies searched beyond the move)
    for move, child in index.children(board):
        if child.depth < min_depth or move.uci() in avoided:
            continue
        score, depth = for_side(child.cp, white), child.depth
        if own is not None and move == own.move and own.depth - 1 > depth:
            score, depth = for_side(own.cp, white), own.depth - 1
        candidates.append((score, move, child, depth))
    if own is not None and all(move != own.move for _, move, _, _ in candidates):
        # Stockfish's move leads out of the database: its score there is all there is.
        candidates.append((for_side(own.cp, white), own.move, None, own.depth - 1))
    if not candidates:
        return None
    best = max(score for score, _, _, _ in candidates)
    judged = []
    for score, move, child, depth in candidates:
        if score < best - window:
            continue
        if child is not None:
            board.push(move)
            replies = [(reply_move, reply) for reply_move, reply in index.children(board)
                       if reply.depth >= min_depth and reply.depth + 1 >= depth]
            board.pop()
            if replies:
                after = min(for_side(reply.cp, white) for _, reply in replies)
                expected = any(reply_move == child.move for reply_move, _ in replies)
                score = after if expected else min(score, after)
        judged.append((score, move))
    top = max(score for score, _ in judged)
    for score, move in judged:
        if own is not None and move == own.move and score >= top - noise:
            return move, score
    return max(judged, key=lambda item: item[0])[::-1]


def show(cp):
    """A score for printing: pawns, or the moves to a forced mate."""
    if abs(cp) >= MATE_FROM:
        return f"{'+' if cp > 0 else '-'}M{MATE - abs(cp)}"
    return f"{cp / 100:+.2f}"


def board_of(text):
    """A position given as a FEN or as moves in SAN from the start position."""
    if "/" in text:
        return chess.Board(text)
    board = chess.Board()
    for san in text.split():
        if not san.rstrip(".").isdigit():
            board.push_san(san)
    return board


def main():
    args = sys.argv[1:]
    path = DEFAULT_INDEX
    if "--index" in args:
        at = args.index("--index")
        path = args[at + 1]
        del args[at:at + 2]
    if not args:
        sys.exit(__doc__)
    index = EvalIndex(path)
    for text in args:
        board = board_of(text)
        entry = index.lookup(board)
        print(f"{text}\n  {board.epd()}")
        if entry is None:
            print("  not in the database")
        else:
            move = board.san(entry.move) if entry.move else "?"
            print(f"  {show(entry.cp)} for White at depth {entry.depth}, Stockfish plays {move} "
                  f"({entry.evals} evaluations, up to {entry.lines} lines)")
        white = board.turn == chess.WHITE
        children = sorted(index.children(board), key=lambda item: -for_side(item[1].cp, white))
        for move, child in children:
            print(f"    {board.san(move):7} {show(child.cp)} at depth {child.depth}")
        if not children:
            print("    no position after it is in the database")


if __name__ == "__main__":
    main()
