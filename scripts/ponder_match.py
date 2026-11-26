#!/usr/bin/env python3
"""Plays two versions of the engine against each other while both think on the opponent's time.

    tools/lichess/venv/bin/python scripts/ponder_match.py --engine target/tune/release/tabx \\
        --old-option tm_ponder_scale=100 --cores 16-27

fastchess, which plays every other test here (scripts/sprt.sh), never lets an
engine think on the opponent's time, so it cannot measure anything that only
matters when the engine does: how much longer a move may take because the
expected reply will cost little, for one. On Lichess the bot ponders in every
game. This script plays the games the way lichess-bot does, through the same
library (python-chess): after its move an engine is told to ponder on the
reply it expects, and gets "ponderhit" if that reply comes and a new search
if it does not.

The two sides are called "new" and "old", and the result is given from the
new side's point of view. They can be two binaries (--engine, --old-engine)
or one binary with different options (--new-option, --old-option; a build
with "--features tune" has every search and clock parameter as an option).
Every opening is played twice, once with each side as White.

Each game keeps two cores busy all the time, because the side that is not on
move is pondering. With --cores every engine gets a core of its own (and the
default number of games at once is half the number of cores); keep the
machine's other work off those cores, as for any test on the clock.

The bot also plays its first moves from an opening book, and a book can be
tested here in the same way: --new-book and --old-book give each side a book
in Polyglot format, from which it takes its moves as lichess-bot does (for
its first --book-moves moves, as long as the position is in the book, a move
picked at random by its weight). Start such a test from the initial position
(--book with a file that holds that one position): the books bring the
variety. With --nodes the engines search a given number of nodes a move
instead of playing on the clock (a fifth more or less from move to move, or
games from the same book line would repeat each other move for move); then
nobody ponders, a game needs one core, and the result does not depend on
what else the machine is doing. That leaves out the time a book move saves,
so it is the cautious way to test a book.

--new-database and --old-database name the index of the Lichess evaluation
database (scripts/evaldb.py) for a side: where its book has no move, and
within --book-moves, it plays the move Stockfish's analysis there supports
(evaldb.database_choice).

Games end as in scripts/sprt.sh: by the rules, or early when both engines
have agreed for a while that the game is drawn (from move 40, 16 plies with
scores within 0.10) or won (6 plies with scores beyond 6.00). The test stops
when a sequential probability ratio test of --elo0 against --elo1 is decided
(logistic Elo, pentanomial variance, alpha 0.05, beta 0.1), or after --games.

Written to --out (a new folder under target/ponder by default): games.pgn,
and games.jsonl with the time taken for every move, whether the mover had
pondered on the right reply, and both clocks at the end.

Needs the python-chess package, which the Lichess setup installs.
"""
import argparse
import collections
import json
import math
import multiprocessing
import os
import queue
import random
import signal
import sys
import time

import chess
import chess.engine
import chess.pgn
import chess.polyglot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOWER = math.log(0.1 / 0.95)   # beta / (1 - alpha)
UPPER = math.log(0.9 / 0.05)   # (1 - beta) / alpha
DRAW_FROM_MOVE, DRAW_PLIES, DRAW_SCORE = 40, 16, 10
WIN_PLIES, WIN_SCORE = 6, 600


class Stalled(Exception):
    """An engine did not answer long after its clock had run out."""


def on_alarm(signum, frame):
    raise Stalled()


def read_position(line):
    """A board from a line of an opening file: a full FEN, or an EPD record."""
    fields = line.split()
    if len(fields) >= 6 and fields[4].isdigit() and fields[5].isdigit():
        return chess.Board(" ".join(fields[:6]))
    board = chess.Board()
    board.set_epd(line)
    return board


def play_game(paths, options, cores, line, base, increment, new_is_white, books=None, book_moves=0,
              nodes=None, rng=None, databases=None):
    """Plays one game and returns its record (a dict that goes into games.jsonl)."""
    names = {chess.WHITE: "new" if new_is_white else "old", chess.BLACK: "old" if new_is_white else "new"}
    books = books or {}
    databases = databases or {}
    from_book = {"new": 0, "old": 0}
    from_database = {"new": 0, "old": 0}
    board = read_position(line)
    start_fen = board.fen()
    remaining = {chess.WHITE: base, chess.BLACK: base}
    lowest = {chess.WHITE: base, chess.BLACK: base}
    # The reply each side is pondering on, and whether the move that came was that reply.
    expected = {chess.WHITE: None, chess.BLACK: None}
    guessed = {chess.WHITE: None, chess.BLACK: None}
    score = {chess.WHITE: None, chess.BLACK: None}   # each engine's last score, from White's side
    times, hits = [], []
    quiet = decisive = 0
    leader = None
    result = termination = None
    game_key = object()
    engines = {}
    try:
        for colour, core in zip((chess.WHITE, chess.BLACK), cores or (None, None)):
            command = [paths[names[colour]]]
            if core is not None:
                # Before the engine starts, so that every thread it makes stays on the core.
                command = ["taskset", "-c", str(core)] + command
            engines[colour] = chess.engine.SimpleEngine.popen_uci(command)
            engines[colour].configure(options[names[colour]])
        while result is None:
            side = board.turn
            if board.is_checkmate():
                result, termination = ("0-1" if side == chess.WHITE else "1-0"), "checkmate"
                break
            if (board.is_stalemate() or board.is_insufficient_material() or board.is_fifty_moves()
                    or board.is_repetition(3)):
                result, termination = "1/2-1/2", "draw by rule"
                break
            book = books.get(names[side])
            database = databases.get(names[side])
            if (book is not None or database is not None) and board.fullmove_number <= book_moves:
                ready = None
                if book is not None:
                    try:
                        ready = book.weighted_choice(board, random=rng).move
                        from_book[names[side]] += 1
                    except IndexError:
                        pass
                if ready is None and database is not None:
                    import evaldb
                    choice = evaldb.database_choice(board.copy(stack=False), database)
                    if choice is not None:
                        ready = choice[0]
                        from_database[names[side]] += 1
                if ready is not None and ready in board.legal_moves:
                    # A book move costs no time, and the engine is not asked: it has
                    # no reply in mind and gives no score.
                    if expected[not side] is not None:
                        guessed[not side] = ready == expected[not side]
                    board.push(ready)
                    expected[side] = guessed[side] = None
                    times.append(0.0)
                    hits.append(None)
                    continue
            if nodes:
                # A search of exactly so many nodes always finds the same move, and two
                # games from the same book line would be one game played twice.
                limit = chess.engine.Limit(nodes=round(nodes * rng.uniform(0.8, 1.2)) if rng else nodes)
                may_ponder = False
                signal.setitimer(signal.ITIMER_REAL, 300)
            else:
                limit = chess.engine.Limit(white_clock=remaining[chess.WHITE], black_clock=remaining[chess.BLACK],
                                           white_inc=increment, black_inc=increment)
                # Like lichess-bot: no pondering after a side's first move.
                may_ponder = len(board.move_stack) >= 2
                signal.setitimer(signal.ITIMER_REAL, remaining[side] + 30)
            start = time.perf_counter()
            played = engines[side].play(board, limit, game=game_key, ponder=may_ponder,
                                        info=chess.engine.INFO_SCORE)
            used = time.perf_counter() - start
            signal.setitimer(signal.ITIMER_REAL, 0)
            times.append(round(used, 4))
            hits.append(guessed[side])
            if not nodes:
                remaining[side] -= used
                lowest[side] = min(lowest[side], remaining[side])
                if remaining[side] < 0:
                    result, termination = ("0-1" if side == chess.WHITE else "1-0"), "time forfeit"
                    break
                remaining[side] += increment
            if played.move is None or played.move not in board.legal_moves:
                result, termination = ("0-1" if side == chess.WHITE else "1-0"), f"illegal move {played.move}"
                break
            if expected[not side] is not None:
                guessed[not side] = played.move == expected[not side]
            board.push(played.move)
            expected[side] = played.ponder if may_ponder else None
            guessed[side] = None

            # A move played at once after a right guess may come without a new
            # score: the engine's last one stands.
            if played.info.get("score") is not None:
                score[side] = played.info["score"].white().score(mate_score=100000)
            now = score[side]
            quiet = quiet + 1 if now is not None and abs(now) <= DRAW_SCORE and board.fullmove_number >= DRAW_FROM_MOVE else 0
            if now is not None and abs(now) >= WIN_SCORE:
                decisive = decisive + 1 if leader == (now > 0) else 1
                leader = now > 0
            else:
                decisive, leader = 0, None
            if quiet >= DRAW_PLIES:
                result, termination = "1/2-1/2", "adjudication"
            elif decisive >= WIN_PLIES:
                result, termination = ("1-0" if leader else "0-1"), "adjudication"
    except Stalled:
        result, termination = "*", "an engine stopped answering"
    except chess.engine.EngineError as error:
        result, termination = "*", f"engine error: {error}"
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        for engine in engines.values():
            try:
                engine.quit()
            except Exception:
                engine.close()

    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = "ponder match"
    game.headers["White"], game.headers["Black"] = names[chess.WHITE], names[chess.BLACK]
    game.headers["Result"] = result
    game.headers["TimeControl"] = f"{nodes} nodes a move" if nodes else f"{base:g}+{increment:g}"
    game.headers["Termination"] = termination
    by_name = {names[colour]: colour for colour in (chess.WHITE, chess.BLACK)}
    return {
        "white": names[chess.WHITE], "fen": start_fen, "result": result, "termination": termination,
        "plies": len(board.move_stack), "times": times, "hits": hits, "from_book": from_book,
        "from_database": from_database,
        "left": {name: round(remaining[colour], 3) for name, colour in by_name.items()},
        "lowest": {name: round(lowest[colour], 3) for name, colour in by_name.items()},
        "pgn": str(game),
    }


def worker(slot, cores, paths, options, openings, base, increment, next_pair, last_pair, stop, results,
           book_paths, book_moves, nodes, seed, database_paths):
    """Plays pairs of games until there are no more to play."""
    parent = os.getppid()
    books = {name: chess.polyglot.open_reader(path) for name, path in book_paths.items()}
    databases = {}
    if database_paths:
        import evaldb   # only for --new-database / --old-database
        databases = {name: evaldb.EvalIndex(path, remember=200_000) for name, path in database_paths.items()}
    signal.signal(signal.SIGINT, signal.SIG_IGN)   # the main process decides when to stop
    # Told to end at once: leave through play_game, which shuts its engines down.
    signal.signal(signal.SIGTERM, lambda signum, frame: sys.exit(1))
    signal.signal(signal.SIGALRM, on_alarm)
    if cores:
        os.sched_setaffinity(0, set(cores))
    while not stop.is_set() and os.getppid() == parent:
        with next_pair.get_lock():
            pair = next_pair.value
            if pair >= last_pair:
                break
            next_pair.value += 1
        line = openings[pair % len(openings)]
        for index, new_is_white in enumerate((True, False)):
            # The engine with White gets the first core, so each side has each core once.
            # The book moves of a game depend on the seed and the game alone, not on which worker plays it.
            rng = random.Random(seed * 1_000_003 + 2 * pair + index)
            record = play_game(paths, options, cores, line, base, increment, new_is_white,
                               books, book_moves, nodes, rng, databases)
            record.update(pair=pair, game=index)
            results.put(record)
    results.put({"done": slot})


def elo(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return -400 * math.log10(1 / score - 1)


def statistics(pairs, elo0, elo1):
    """Score, Elo with its 95% margin, and the test's log-likelihood ratio, from the pairs' results.

    pairs[i] is the number of pairs in which the new side made i half points.
    """
    count = sum(pairs)
    if count == 0:
        return None
    mean = sum(i * n for i, n in enumerate(pairs)) / (4 * count)
    variance = sum(n * (i / 4 - mean) ** 2 for i, n in enumerate(pairs)) / count
    if variance == 0 or not 0 < mean < 1:
        return mean, elo(mean), float("inf"), 0.0
    margin = 1.96 * math.sqrt(variance / count) * 400 / (math.log(10) * mean * (1 - mean))
    low, high = 1 / (1 + 10 ** (-elo0 / 400)), 1 / (1 + 10 ** (-elo1 / 400))
    llr = count * (high - low) * (2 * mean - low - high) / (2 * variance)
    return mean, elo(mean), margin, llr


class Tally:
    """What has been played so far."""

    def __init__(self, args):
        self.args = args
        self.open = {}                      # pair -> its first record
        self.pairs = [0] * 5
        self.games = collections.Counter()  # wins, draws, losses of the new side
        self.forfeits = collections.Counter()
        self.void = 0
        self.guesses = collections.Counter()
        self.right = collections.Counter()
        self.left = collections.defaultdict(float)
        self.endings = collections.Counter()
        self.from_book = collections.Counter()
        self.from_database = collections.Counter()

    @staticmethod
    def points(record):
        """Half points of the new side in one game, or None for a game without a result."""
        if record["result"] == "1/2-1/2":
            return 1
        if record["result"] not in ("1-0", "0-1"):
            return None
        return 2 if (record["result"] == "1-0") == (record["white"] == "new") else 0

    def add(self, record):
        """Counts a game; returns True when it completes a pair."""
        first = self.open.pop(record["pair"], None)
        if first is None:
            self.open[record["pair"]] = record
            return False
        both = [self.points(first), self.points(record)]
        if None in both:
            self.void += 1
            return True
        self.pairs[sum(both)] += 1
        for game, points in zip((first, record), both):
            self.games[points] += 1
            self.endings[game["termination"]] += 1
            if game["termination"] == "time forfeit":
                loser_is_white = game["result"] == "0-1"
                self.forfeits["new" if (game["white"] == "new") == loser_is_white else "old"] += 1
            # hits[0] belongs to the side that moved first, which is Black in some openings.
            movers = [game["white"], "old" if game["white"] == "new" else "new"]
            if game["fen"].split()[1] == "b":
                movers.reverse()
            for ply, hit in enumerate(game["hits"]):
                if hit is not None:
                    self.guesses[movers[ply % 2]] += 1
                    self.right[movers[ply % 2]] += hit
            for name in ("new", "old"):
                self.left[name] += game["left"][name]
                self.from_book[name] += game.get("from_book", {}).get(name, 0)
                self.from_database[name] += game.get("from_database", {}).get(name, 0)
        return True

    def line(self):
        stats = statistics(self.pairs, self.args.elo0, self.args.elo1)
        count = sum(self.pairs)
        if stats is None:
            return "no pair finished"
        mean, value, margin, llr = stats
        games = 2 * count
        hit = {name: (100 * self.right[name] / self.guesses[name] if self.guesses[name] else 0.0)
               for name in ("new", "old")}
        clock = ("" if self.args.nodes else
                 f", lost on time: new {self.forfeits['new']}, old {self.forfeits['old']}; "
                 f"right guesses: new {hit['new']:.0f}%, old {hit['old']:.0f}%; "
                 f"clock left at the end: new {self.left['new'] / games:.2f}s, old {self.left['old'] / games:.2f}s")
        return (f"{games} games: +{self.games[2]} ={self.games[1]} -{self.games[0]} ({100 * mean:.2f}%), "
                f"Elo {value:+.1f} +/- {margin:.1f}, LLR {llr:.2f} ({LOWER:.2f}, {UPPER:.2f}) "
                f"[{self.args.elo0:g}, {self.args.elo1:g}], pairs {self.pairs}" + clock
                + (f"; book moves a game: new {self.from_book['new'] / games:.1f}, old {self.from_book['old'] / games:.1f}"
                   if self.from_book else "")
                + (f"; moves from the database a game: new {self.from_database['new'] / games:.1f}, "
                   f"old {self.from_database['old'] / games:.1f}" if self.from_database else "")
                + (f"; pairs without a result: {self.void}" if self.void else ""))


def parse_options(pairs):
    options = {}
    for pair in pairs:
        name, _, value = pair.partition("=")
        if not name or not value:
            sys.exit(f'An option is written name=value, not "{pair}".')
        options[name] = int(value) if value.lstrip("-").isdigit() else value
    return options


def parse_cores(text):
    cores = []
    for part in text.split(","):
        first, _, last = part.partition("-")
        cores.extend(range(int(first), int(last or first) + 1))
    return cores


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engine", default=os.path.join(root, "target/release/tabx"),
                        help="the new side's engine [target/release/tabx]")
    parser.add_argument("--old-engine", help="the old side's engine [the same as --engine]")
    parser.add_argument("--option", action="append", default=[], metavar="NAME=VALUE",
                        help="a UCI option for both sides (can be repeated)")
    parser.add_argument("--new-option", action="append", default=[], metavar="NAME=VALUE",
                        help="a UCI option for the new side only")
    parser.add_argument("--old-option", action="append", default=[], metavar="NAME=VALUE",
                        help="a UCI option for the old side only")
    parser.add_argument("--tc", default="8+0.08", help="seconds per side plus seconds per move [8+0.08]")
    parser.add_argument("--book", default=os.path.join(root, "tools/books/start_6ply.epd"),
                        help="opening positions, one FEN or EPD record per line [tools/books/start_6ply.epd]")
    parser.add_argument("--new-book", help="opening book of the new side (Polyglot format)")
    parser.add_argument("--old-book", help="opening book of the old side (Polyglot format)")
    parser.add_argument("--book-moves", type=int, default=8,
                        help="moves a side may take from its book, like max_depth in the bot's settings [8]")
    parser.add_argument("--new-database", metavar="INDEX",
                        help="index of the Lichess evaluation database: the new side asks it where its book has no move")
    parser.add_argument("--old-database", metavar="INDEX", help="the same for the old side")
    parser.add_argument("--nodes", type=int,
                        help="search this many nodes a move instead of playing on the clock (no pondering)")
    parser.add_argument("--games", type=int, default=6000, help="play at most this many games [6000]")
    parser.add_argument("--elo0", type=float, default=0.0, help="the test's lower Elo [0]")
    parser.add_argument("--elo1", type=float, default=5.0, help="the test's upper Elo [5]")
    parser.add_argument("--no-sprt", action="store_true", help="play all the games whatever the test says")
    parser.add_argument("--cores", help='cores to play on, such as "16-27": two for each game')
    parser.add_argument("--concurrency", type=int, help="games at once [half the number of --cores, else 1]")
    parser.add_argument("--hash", type=int, default=16, help="hash table size in MB [16]")
    parser.add_argument("--seed", type=int, default=20261124, help="for the order of the openings")
    parser.add_argument("--report", type=int, default=100, help="print the score after every so many games [100]")
    parser.add_argument("--out", help="folder for the games [a new one under target/ponder]")
    args = parser.parse_args()

    paths = {"new": os.path.abspath(args.engine), "old": os.path.abspath(args.old_engine or args.engine)}
    for path in paths.values():
        if not os.access(path, os.X_OK):
            sys.exit(f"{path} is not an engine binary.")
    shared = {"Threads": 1, "Hash": args.hash, **parse_options(args.option)}
    options = {"new": {**shared, **parse_options(args.new_option)},
               "old": {**shared, **parse_options(args.old_option)}}
    if paths["new"] == paths["old"] and options["new"] == options["old"]:
        print("Both sides are the same engine with the same options: this only tests the script.")
    base, _, increment = args.tc.partition("+")
    base, increment = float(base), float(increment or 0)
    with open(args.book) as stream:
        openings = [line.strip() for line in stream if line.strip()]
    random.Random(args.seed).shuffle(openings)
    cores = parse_cores(args.cores) if args.cores else []
    # On the clock the side that is not on move ponders: two cores a game. With a
    # fixed number of nodes only one engine works at a time.
    per_game = 1 if args.nodes else 2
    concurrency = args.concurrency or max(1, len(cores) // per_game)
    if cores and len(cores) < per_game * concurrency:
        sys.exit(f"{concurrency} games at once need {per_game * concurrency} cores, --cores names {len(cores)}.")
    book_paths = {name: os.path.abspath(path) for name, path in (("new", args.new_book), ("old", args.old_book)) if path}
    database_paths = {name: os.path.abspath(path)
                      for name, path in (("new", args.new_database), ("old", args.old_database)) if path}
    for path in list(book_paths.values()) + list(database_paths.values()):
        if not os.path.isfile(path):
            sys.exit(f"{path}: no such file.")
    out = args.out or os.path.join(root, "target/ponder", time.strftime("run-%Y%m%d-%H%M%S"))
    os.makedirs(out, exist_ok=True)
    terms = f"{args.nodes} nodes a move" if args.nodes else f"tc {args.tc}, both sides pondering"
    print(f"new: {paths['new']} {options['new']}\nold: {paths['old']} {options['old']}\n"
          + "".join(f"{name}'s book: {path}, for {args.book_moves} moves\n" for name, path in book_paths.items())
          + "".join(f"{name} asks the database {path} after its book, for the same {args.book_moves} moves\n"
                    for name, path in database_paths.items())
          + f"{terms}, {concurrency} games at once"
          + (f" on cores {args.cores}" if cores else "")
          + f", {os.path.basename(args.book)}, at most {args.games} games; games in {out}", flush=True)

    stop = multiprocessing.Event()
    next_pair = multiprocessing.Value("i", 0)
    results = multiprocessing.Queue()
    workers = []
    for slot in range(concurrency):
        if not cores:
            pair_of_cores = None
        elif args.nodes:
            pair_of_cores = [cores[slot], cores[slot]]
        else:
            pair_of_cores = cores[2 * slot:2 * slot + 2]
        process = multiprocessing.Process(
            target=worker, args=(slot, pair_of_cores, paths, options, openings, base, increment,
                                 next_pair, args.games // 2, stop, results,
                                 book_paths, args.book_moves, args.nodes, args.seed, database_paths))
        process.start()
        workers.append(process)

    def ask_to_stop(signum, frame):
        if stop.is_set():
            raise KeyboardInterrupt
        print("Stopping after the games in progress (again to stop at once).", flush=True)
        stop.set()
    signal.signal(signal.SIGINT, ask_to_stop)
    signal.signal(signal.SIGTERM, ask_to_stop)

    tally = Tally(args)
    running = concurrency
    verdict = None
    reported = 0
    try:
        with open(os.path.join(out, "games.pgn"), "a") as pgn, open(os.path.join(out, "games.jsonl"), "a") as log:
            while running:
                try:
                    record = results.get(timeout=5)
                except queue.Empty:
                    if not any(process.is_alive() for process in workers):
                        break
                    continue
                if "done" in record:
                    running -= 1
                    continue
                print(record.pop("pgn"), file=pgn, end="\n\n", flush=True)
                print(json.dumps(record), file=log, flush=True)
                if not tally.add(record):
                    continue
                games = 2 * sum(tally.pairs)
                stats = statistics(tally.pairs, args.elo0, args.elo1)
                if games - reported >= args.report:
                    reported = games
                    print(tally.line(), flush=True)
                if stats and not args.no_sprt and verdict is None and not stop.is_set():
                    if stats[3] >= UPPER:
                        verdict = f"the new side is better: more than {args.elo0:g} Elo (H1 accepted)"
                    elif stats[3] <= LOWER:
                        verdict = f"the new side is not better by {args.elo1:g} Elo (H0 accepted)"
                    if verdict:
                        stop.set()
    except KeyboardInterrupt:
        print("Stopped at once: the games in progress are lost.", flush=True)
        for process in workers:
            process.terminate()
    for process in workers:
        process.join(timeout=120)
    print(tally.line())
    print(f"endings: {dict(tally.endings)}")
    print(verdict or "The test was not decided.")


if __name__ == "__main__":
    main()
