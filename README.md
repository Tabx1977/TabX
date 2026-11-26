# TabX

A UCI chess engine written in Rust: bitboards, an alpha-beta search with the
usual modern pruning, and a neural network evaluation (NNUE). The network is
trained only on TabX's own games.

TabX has no graphical interface. Load it in any chess GUI that speaks UCI
(Cute Chess, En Croissant, Nibbler, Arena, BanksiaGUI, ...) or drive it from
the command line. Standard chess and Chess960 are both supported.

The engine is young and still getting stronger; see "The network" below for
where it stands.

## Getting it

[Releases](../../releases) come with ready-made binaries for Windows and
Linux, in two builds each:

| Build | For |
|-------|-----|
| `x86-64-v3` | any x86-64 CPU with AVX2 (Intel since 2013, AMD since 2015): the fast one |
| `x86-64-v2` | older x86-64 CPUs (SSE4.2 and POPCNT) |

To build it yourself you need a recent stable Rust (1.88 or newer; `rustup`
installs it):

```
cargo build --release
```

The binary is `target/release/tabx`. Such a build is made for the CPU it is
built on (see `.cargo/config.toml`); for a binary that runs on other
machines, set the target CPU level instead:

```
RUSTFLAGS="-C target-cpu=x86-64-v3" cargo build --release
```

## Using it

```
target/release/tabx                  # UCI mode: reads commands from stdin
target/release/tabx bench            # fixed-depth benchmark, prints the network, nodes and speed
target/release/tabx perft 6          # move generation test from the start position
target/release/tabx perft 5 "<fen>"  # ... or from any position
```

UCI options:

| Option         | Default | Meaning                                                      |
|----------------|---------|--------------------------------------------------------------|
| `Hash`         | 16      | Transposition table size in MB                               |
| `Threads`      | 1       | Search threads                                               |
| `MultiPV`      | 1       | Number of best lines to analyse and report                   |
| `Ponder`       | false   | Set by GUIs that let the engine think on the opponent's time (`go ponder`). The engine ponders either way; with the option on it allows a move a quarter more time, because the moves it guessed right cost little on its own clock |
| `Move Overhead`| 10      | Milliseconds kept in reserve on every move for GUI or network lag |
| `UCI_Chess960` | false   | Chess960 castling notation (Chess960 positions always work)  |
| `Contempt`     | 0       | Centipawns by which the engine thinks a draw worse than a level game: with a positive value it plays on where it could repeat moves |
| `UCI_Opponent` | empty   | Set by GUIs: the opponent's title, rating, kind and name. With `Own Rating` it sets the contempt |
| `Own Rating`   | 0       | The engine's own rating on the opponent's scale. If both ratings are known, the contempt grows by 15 centipawns for every 100 points the engine is ahead, up to 40, and a draw is welcome by up to 15 if it is behind |

Besides the standard UCI commands the engine understands `d` (show the
board), `eval` (static evaluation), `bench [depth]` and `perft <depth>`.

## How it works

| Part | File | Notes |
|------|------|-------|
| Board | `src/position.rs`, `src/types.rs` | Bitboards plus a square-to-piece array. Moves are made on a copy of the position (copy-make), so there is no unmake. Zobrist hashes are updated incrementally. |
| Attack tables | `build.rs`, `src/attacks.rs` | Magic bitboards for sliding pieces. All tables are generated at build time and embedded in the binary. |
| Move generation | `src/movegen.rs` | Generates only legal moves, using pin and check masks. Chess960 castling is supported throughout. |
| Search | `src/search.rs` | Iterative deepening, aspiration windows, principal variation search, null move pruning, reverse futility pruning, razoring, late move reductions (for quiet moves and for captures) and late move pruning, futility and history pruning, static exchange pruning, singular extensions, internal iterative reductions, quiescence search. |
| Move ordering | `src/movepick.rs`, `src/history.rs` | Staged: hash move, winning captures, killers, quiet moves by history, losing captures. History tables: butterfly (kept apart by whether the move's two squares are attacked), pawn structure, capture, continuation, and correction history for the evaluation. |
| Transposition table | `src/tt.rs` | Lockless, shared between threads, three 10-byte entries per 32-byte bucket. |
| Evaluation | `src/nnue.rs`, `nets/` | NNUE with incrementally updated accumulators and AVX2 inference. |
| Threads | `src/search.rs`, `src/uci.rs` | Lazy SMP: every thread searches the same position and shares the transposition table. |
| Time management | `src/uci.rs`, `src/search.rs` | A move gets a share of the clock plus the coming increments (or an even share of the time to the next control), a smaller one in the first moves of a game and a larger one when the engine thinks on the opponent's time. The search stops early when the best move is settled and takes longer when it keeps changing. |

## The network

`nets/gen4.bin`: 768 inputs, mirrored so that the viewer's king is always
on files a to d, a hidden layer of 256 for each side, and 8 output buckets by
the number of pieces on the board.

It was trained on 74 million positions from TabX's own games, played with a
small fixed number of nodes per move, each labelled with the engine's search
score and the game's result. The first round of games was played with a
hand-written evaluation (the PeSTO piece-square tables); every later round
with the network trained on the rounds before it. Data from other engines
has never been used for it, and the next networks are made the same way,
from more and more of TabX's own games.

How a network is made (the commands assume the opening files and tools
named below are in `tools/`):

1. Generate games. Each record is a position, the search score and the final
   game result (the format is documented in `src/datagen.rs`).

   ```
   target/release/tabx datagen --out data/gen.bin --threads 20 --positions 100000000 --nodes 5000 \
       --book tools/books/8moves_plies8-16.epd --book tools/books/UHO_Lichess_4852_v1.epd
   ```

   With `--book` the games start from the positions in those files (one FEN
   per line) plus up to two random moves; without it from 8 or 9 random
   moves; with `--dfrc` from double Fischer random positions.
   `scripts/make_start_book.py` turns a PGN opening collection into such a
   file.

2. Train on a GPU with the `trainer/` crate, which uses
   [bullet](https://github.com/jw1912/bullet):

   ```
   ARCH=bucketed scripts/train.sh mynet data/gen.bin --hidden 256 --layout single \
       --wdl 0.3 --superbatches 50 --batches 450
   ```

   This shuffles the data, trains, and writes `nets/mynet.bin`. One batch is
   16384 positions, so `superbatches x batches x 16384` divided by the number
   of positions is the number of passes over the data.

3. Check the result on positions the network has not seen
   (`scripts/valloss.py --mirrored nets/mynet.bin data/heldout.bin`, lower is
   better), then build the engine with it and test it in games:

   ```
   TABX_NET=nets/mynet.bin TABX_NET_HIDDEN=256 TABX_NET_LAYOUT=single cargo build --release
   ```

   `build.rs` compiles in the network named by `TABX_NET` (by default
   `nets/gen4.bin`), with the hidden layer width and king bucket layout
   given; if they do not describe the file, the build fails.

The engine also has commands to read the Lichess database of position
evaluations (`tabx import-evals`, `tabx index-evals`, and
`scripts/evaldb.py`). They were used for experiments; TabX's network is not
trained on that data.

## Testing

```
cargo test                 # correctness tests (about 10 seconds)
```

The tests cover move generation (perft on the standard positions, and
comparison against the independent `shakmaty` library on tens of thousands of
random standard and Chess960 positions), incremental hashing, static exchange
evaluation, the move picker, the network's incremental updates, the division
of the clock, thinking on the opponent's time and the value of a draw (the
last two talk to the built engine the way a GUI does).

A change to the search or evaluation is kept only if it wins a statistical
self-play test (SPRT). `scripts/sprt.sh` plays two versions until the result
is clear:

```
scripts/sprt.sh                    # working tree vs HEAD
scripts/sprt.sh main               # working tree vs a branch, tag or commit
scripts/sprt.sh feature main       # two refs (or two engine binaries)
ELO1=10 TC=5+0.05 scripts/sprt.sh  # options are environment variables, see the script
```

It needs [fastchess](https://github.com/Disservin/fastchess) and an opening
book, by default `tools/fastchess/fastchess` and
`tools/books/UHO_Lichess_4852_v1.epd` (Stefan Pohl's "Unbalanced Human
Openings"). A test at a fixed number of nodes (`NODES=...`) is quick but
blind to speed, and its verdict does not always hold on the clock, so
changes are confirmed on the clock as well.

Self-play cannot see a weakness that both versions share, so
`scripts/spar.sh` also plays versions against an outside opponent, Stockfish
held to a small number of nodes per move, with the same openings for every
version. `scripts/rate.sh` gives a rough absolute rating against Stockfish's
calibrated strength levels. `scripts/spsa.py` tunes the search parameters
(`src/params.rs`) by self-play, and `scripts/ponder_match.py` plays games in
which both sides think on the opponent's time, which fastchess does not.

`tabx bench` prints a node count that changes whenever the search behaves
differently; commit messages quote it (`Bench: N`).

## Licence

Free software under the GNU General Public License, version 3 or (at your
option) any later version. The full text is in `LICENSE`.
