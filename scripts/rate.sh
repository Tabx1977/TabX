#!/usr/bin/env bash
# Estimates an absolute rating by playing Stockfish at limited strength.
#
#   scripts/rate.sh [engine]
#
# [engine] is the binary to rate (default: a fresh release build of the
# working tree).
#
# Stockfish can be told to play at a chosen rating (UCI_LimitStrength with
# UCI_Elo, 1320 to 3190). Its developers calibrated that scale at 120s+1s
# against the CCRL 40/4 rating list. The engine plays GAMES games against
# each rating in LEVELS; scripts/rating.py then fits the rating that best
# explains all the scores, and prints it again after every level.
#
# Read the result as a rough estimate, not as a place on a rating list:
#   * a handicapped Stockfish does not play like an ordinary engine of that
#     rating (it searches at full strength and then sometimes picks a worse
#     move on purpose);
#   * the scale is defined at 120s+1s, so a different TC measures something
#     slightly different;
#   * the +/- printed covers the luck of the games only;
#   * 3190 is the top of the scale. An engine that beats that level clearly
#     needs other opponents.
# Levels far from the engine's strength cost time and add little. The
# defaults are the two levels nearest to where the engine was last measured.
# If it has changed a lot since, find the right levels
# with a short wide pass first:
#   LEVELS="2300 2600 2900 3190" GAMES=16 TC=10+0.1 scripts/rate.sh
#
# Environment overrides (defaults in brackets):
#   STOCKFISH [tools/stockfish/stockfish/stockfish-linux-x86-64-universal]
#   LEVELS ["3050 3190"]  GAMES [40] per level  TC [120+1]
#   CONC [8]  HASH [256]  BOOK [tools/books/8moves_v3.pgn]  FASTCHESS
#   OUT [a new directory under target/rating]  name an earlier run's
#       directory to add more games or levels to it
#
# Games are not adjudicated as draws: a handicapped opponent can still lose a
# level position, and cutting those games short would understate the engine.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTCHESS="${FASTCHESS:-$ROOT/tools/fastchess/fastchess}"
STOCKFISH="${STOCKFISH:-$ROOT/tools/stockfish/stockfish/stockfish-linux-x86-64-universal}"
BOOK="${BOOK:-$ROOT/tools/books/8moves_v3.pgn}"
LEVELS="${LEVELS:-3050 3190}"
GAMES="${GAMES:-40}"
TC="${TC:-120+1}"
CONC="${CONC:-8}"
HASH="${HASH:-256}"
WORK="$ROOT/target/rating"

for file in "$FASTCHESS" "$STOCKFISH" "$BOOK"; do
    if [[ ! -e "$file" ]]; then
        echo "missing: $file" >&2
        exit 1
    fi
done
# The matches run from their own directory, so relative paths must not be kept.
FASTCHESS="$(realpath "$FASTCHESS")"
STOCKFISH="$(realpath "$STOCKFISH")"
BOOK="$(realpath "$BOOK")"

mkdir -p "$WORK"
if [[ -n "${OUT:-}" ]]; then
    mkdir -p "$OUT"
    RUN_DIR="$(realpath "$OUT")"
else
    RUN_DIR="$(mktemp -d "$WORK/run-XXXXXX")"
fi
if [[ $# -ge 1 ]]; then
    ENGINE="$(realpath "$1")"
else
    (cd "$ROOT" && cargo build --release --quiet)
    ENGINE="$RUN_DIR/tabx"
    cp "$ROOT/target/release/tabx" "$ENGINE"
fi

if [[ "$BOOK" == *.pgn ]]; then
    OPENINGS=(file="$BOOK" format=pgn order=random plies=16)
else
    OPENINGS=(file="$BOOK" format=epd order=random)
fi

# fastchess comments on every move where the principal variation an engine
# printed does not fit the move it played. A handicapped Stockfish does that
# on purpose, hundreds of times per match, so those remarks are kept out of
# the terminal. The complete output is in match-<level>.log.
NOISE='^(Warning; (Bestmove does not match|PV continues after)|Info; |Position; |Moves; )'

cd "$RUN_DIR"
echo "engine:    $ENGINE"
echo "reference: $STOCKFISH, limited to $LEVELS"
echo "$GAMES games per level at tc $TC, $CONC games at once, everything saved in $RUN_DIR"

for level in $LEVELS; do
    echo
    echo "=== against Stockfish limited to $level ==="
    "$FASTCHESS" \
        -engine cmd="$ENGINE" name=tabx \
        -engine cmd="$STOCKFISH" name="stockfish-$level" \
            option.UCI_LimitStrength=true option.UCI_Elo="$level" \
        -each tc="$TC" option.Hash="$HASH" option.Threads=1 \
        -openings "${OPENINGS[@]}" \
        -rounds $((GAMES / 2)) -repeat -concurrency "$CONC" \
        -resign movecount=3 score=600 twosided=true \
        -recover -ratinginterval 0 -autosaveinterval 0 \
        -pgnout file="games-$level.pgn" | tee "match-$level.log" \
        | grep --line-buffered -vE "$NOISE" || true

    # fastchess counts wins and losses for the first engine, the one being rated.
    result="$(grep -oE 'Games: [0-9]+, Wins: [0-9]+, Losses: [0-9]+, Draws: [0-9]+' "match-$level.log" | tail -1 || true)"
    if [[ -z "$result" ]]; then
        echo "no result found in $RUN_DIR/match-$level.log" >&2
        exit 1
    fi
    echo "$level $(echo "$result" | tr -dc '0-9 ' | awk '{print $2, $3, $4}')" >> results.txt

    echo
    python3 "$ROOT/scripts/rating.py" results.txt
done
