#!/usr/bin/env bash
# Plays versions of the engine against Stockfish, as an outside opponent.
#
#   scripts/spar.sh <engine>...
#
# Each <engine> is an engine binary or a git ref (branch, tag, commit), which
# is built automatically.
#
# Self-play (scripts/sprt.sh) compares two versions of this engine with each
# other, and cannot see a weakness that both share: on Lichess the engine
# kept losing from positions it thought were fine, and no self-play test had
# shown it. Here every version plays the same games against Stockfish (the
# same openings, each with both colours), and the scores are compared.
#
# Both sides search a fixed number of nodes per move, so the result does not
# depend on how busy the machine is: NODES for the engine, SF_NODES for
# Stockfish. Stockfish judges positions far better and needs far fewer nodes
# for the same strength, so with few nodes it is an opponent that sees less
# tactics than the engine but understands the position better, which is the
# kind of opponent the engine's evaluation has to be measured against. Set
# SF_NODES so that the engine scores somewhere between 30% and 70%: a change
# shows best there.
#
# A fixed node count hides differences in speed. A version that is slower
# per node (a larger network) has to be confirmed on the clock as well:
# TC=8+0.08 SF_TC=1+0.01 plays on the clock instead (then keep the machine
# otherwise idle, and CONC below the number of free cores).
#
# Environment overrides (defaults in brackets):
#   GAMES [1000] per engine   NODES [50000]   SF_NODES [3000]
#   TC, SF_TC [unset]         CONC [12]       HASH [16]
#   BOOK [tools/books/UHO_Lichess_4852_v1.epd]
#   STOCKFISH [tools/stockfish/stockfish/stockfish-linux-x86-64-universal]
#   FASTCHESS [tools/fastchess/fastchess]
#   OUT [a new directory under target/spar]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FASTCHESS="${FASTCHESS:-$ROOT/tools/fastchess/fastchess}"
STOCKFISH="${STOCKFISH:-$ROOT/tools/stockfish/stockfish/stockfish-linux-x86-64-universal}"
BOOK="${BOOK:-$ROOT/tools/books/UHO_Lichess_4852_v1.epd}"
GAMES="${GAMES:-1000}"
NODES="${NODES:-50000}"
SF_NODES="${SF_NODES:-3000}"
CONC="${CONC:-12}"
HASH="${HASH:-16}"
WORK="$ROOT/target/spar"

if [[ $# -eq 0 ]]; then
    awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "${BASH_SOURCE[0]}"
    exit 1
fi
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

# Builds the engine at a git ref and prints the path of the binary.
build_ref() {
    local sha out src
    sha="$(git -C "$ROOT" rev-parse --short --verify --quiet "$1^{commit}")" || return 1
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

if [[ -n "${TC:-}" ]]; then
    OURS=(tc="$TC")
    THEIRS=(tc="${SF_TC:?set SF_TC as well as TC}")
    TERMS="$TC against Stockfish at $SF_TC"
else
    OURS=(tc=inf nodes="$NODES")
    THEIRS=(tc=inf nodes="$SF_NODES")
    TERMS="$NODES nodes a move against Stockfish at $SF_NODES"
fi
case "$BOOK" in
    *.pgn) FORMAT=pgn ;;
    *) FORMAT=epd ;;
esac

# Find or build every engine before anything is played: the matches run in
# their own directory, where a relative path would no longer be found.
BINARIES=()
LABELS=()
for engine in "$@"; do
    if [[ -f "$engine" && -x "$engine" ]]; then
        BINARIES+=("$(realpath "$engine")")
    elif binary="$(build_ref "$engine")"; then
        BINARIES+=("$binary")
    else
        echo "$engine is neither an engine binary nor a git ref" >&2
        exit 1
    fi
    LABELS+=("$(basename "$engine")")
done
for first in "${!BINARIES[@]}"; do
    for second in "${!BINARIES[@]}"; do
        if (( first < second )) && cmp -s "${BINARIES[$first]}" "${BINARIES[$second]}"; then
            echo "${LABELS[$first]} and ${LABELS[$second]} are the same build: one of them is enough" >&2
            exit 1
        fi
    done
done

OUT="${OUT:-$(mktemp -d "$WORK/run-XXXXXX")}"
mkdir -p "$OUT"
OUT="$(realpath "$OUT")"
echo "$GAMES games each, $TERMS, $CONC at once, 1 thread, $HASH MB, $(basename "$BOOK"); games in $OUT"

cd "$OUT"
for index in "${!BINARIES[@]}"; do
    binary="${BINARIES[$index]}"
    label="${LABELS[$index]}"
    # The same seed for every engine: the same openings in the same order.
    "$FASTCHESS" \
        -engine cmd="$binary" name="$label" "${OURS[@]}" \
        -engine cmd="$STOCKFISH" name=stockfish "${THEIRS[@]}" \
        -each option.Hash="$HASH" option.Threads=1 \
        -openings file="$BOOK" format="$FORMAT" order=random -srand 20261124 \
        -rounds $((GAMES / 2)) -repeat -concurrency "$CONC" \
        -resign movecount=3 score=600 twosided=true \
        -recover -ratinginterval 0 -autosaveinterval 0 \
        -pgnout file="games-$label.pgn" > "match-$label.log" 2>&1 || true
    # fastchess counts wins and losses for the first engine, the one being tested.
    result="$(grep -E '^(Elo|Games|Ptnml)' "match-$label.log" | tail -n 3 | tr '\n' ' ' || true)"
    forfeits="$(grep -c 'Termination "time forfeit"' "games-$label.pgn" 2> /dev/null || true)"
    echo "$label: ${result:-no result, see $OUT/match-$label.log}| lost or won on time: ${forfeits:-0}"
done
