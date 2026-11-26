#!/usr/bin/env bash
# SPRT self-play test between two versions of the engine.
#
#   scripts/sprt.sh                 working tree ("new") vs HEAD ("old")
#   scripts/sprt.sh <old>           working tree vs <old>
#   scripts/sprt.sh <new> <old>     <new> vs <old>
#
# <new> and <old> are each either an engine binary or a git ref (branch, tag,
# commit), which is built automatically.
#
# The test stops by itself once the result is statistically clear:
#   "H1 accepted" = new is better by about ELO1 or more -> keep the change
#   "H0 accepted" = new is not better than ELO0         -> reject the change
#
# Environment overrides (defaults in brackets):
#   TC [8+0.08]  CONC [16]  ELO0 [0]  ELO1 [5]  HASH [16]  THREADS [1]
#   ROUNDS [40000]  BOOK [tools/books/UHO_Lichess_4852_v1.epd]  FASTCHESS
#   NODES [unset]  play with a fixed node budget per move instead of a clock.
#
# NODES=50000 is a quick screen for changes to search logic or evaluation: a
# game takes a few seconds and the result does not depend on how busy the
# machine is. It cannot see changes in speed or time management, so confirm
# anything important on the clock as well.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTCHESS="${FASTCHESS:-$ROOT/tools/fastchess/fastchess}"
BOOK="${BOOK:-$ROOT/tools/books/UHO_Lichess_4852_v1.epd}"
TC="${TC:-8+0.08}"
CONC="${CONC:-16}"
ELO0="${ELO0:-0}"
ELO1="${ELO1:-5}"
HASH="${HASH:-16}"
THREADS="${THREADS:-1}"
ROUNDS="${ROUNDS:-40000}"
WORK="$ROOT/target/sprt"
mkdir -p "$WORK"
for file in "$FASTCHESS" "$BOOK"; do
    if [[ ! -f "$file" ]]; then
        echo "not found: $file" >&2
        exit 1
    fi
done
# The match runs in its own folder, so relative paths would stop working there.
FASTCHESS="$(realpath "$FASTCHESS")"
BOOK="$(realpath "$BOOK")"

# Builds the engine at a git ref and prints the path of the binary.
build_ref() {
    local sha out src
    sha="$(git -C "$ROOT" rev-parse --short --verify --quiet "$1^{commit}")" || {
        echo "$1 is neither an engine binary nor a git ref" >&2
        return 1
    }
    out="$WORK/tabx-$sha"
    if [[ ! -x "$out" ]]; then
        src="$WORK/src-$sha"
        rm -rf "$src"
        mkdir -p "$src"
        git -C "$ROOT" archive "$1" | tar -x -C "$src"
        # Files come out of the archive dated with their commit, and cargo
        # goes by dates: after building a newer commit it would take an older
        # one for unchanged and leave the binary it has. So the files are
        # dated now, and the old binary is removed: if nothing gets built,
        # the copy fails instead of passing off the other commit's engine.
        find "$src" -type f -exec touch {} +
        rm -f "$WORK/build/release/tabx" "$WORK/build/release/patina"
        (cd "$src" && cargo build --release --quiet --target-dir "$WORK/build") >&2
        # Commits from before the engine was named TabX build a binary called patina.
        if [[ -x "$WORK/build/release/tabx" ]]; then
            cp "$WORK/build/release/tabx" "$out"
        else
            cp "$WORK/build/release/patina" "$out"
        fi
        rm -rf "$src"
    fi
    echo "$out"
}

# Resolves an argument to a binary: an existing executable is used as is,
# anything else is treated as a git ref and built.
resolve() {
    if [[ -f "$1" && -x "$1" ]]; then
        realpath "$1"
    else
        build_ref "$1"
    fi
}

if [[ $# -eq 2 ]]; then
    NEW="$(resolve "$1")"
    OLD="$(resolve "$2")"
else
    (cd "$ROOT" && cargo build --release --quiet)
    NEW="$WORK/tabx-dev-$$"
    cp "$ROOT/target/release/tabx" "$NEW"
    OLD="$(resolve "${1:-HEAD}")"
fi

if cmp -s "$NEW" "$OLD"; then
    echo "new and old are the same build: nothing to test ($NEW, $OLD)" >&2
    exit 1
fi

if [[ -n "${NODES:-}" ]]; then
    LIMIT=(tc=inf nodes="$NODES")
    LIMIT_TEXT="$NODES nodes per move"
else
    LIMIT=(tc="$TC")
    LIMIT_TEXT="tc $TC"
fi

RUN_DIR="$(mktemp -d "$WORK/run-XXXXXX")"
cd "$RUN_DIR"
echo "new: $NEW"
echo "old: $OLD"
echo "$LIMIT_TEXT, $CONC games at once, SPRT [$ELO0, $ELO1], games saved in $RUN_DIR/games.pgn"

exec "$FASTCHESS" \
    -engine cmd="$NEW" name=new \
    -engine cmd="$OLD" name=old \
    -each "${LIMIT[@]}" option.Hash="$HASH" option.Threads="$THREADS" \
    -openings file="$BOOK" format=epd order=random \
    -rounds "$ROUNDS" -repeat -concurrency "$CONC" \
    -sprt elo0="$ELO0" elo1="$ELO1" alpha=0.05 beta=0.1 \
    -draw movenumber=40 movecount=8 score=10 \
    -resign movecount=3 score=600 twosided=true \
    -recover -ratinginterval 100 -autosaveinterval 0 \
    -pgnout file=games.pgn
