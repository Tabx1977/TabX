#!/usr/bin/env bash
# Runs several SPRT tests one after another and logs a summary of each.
#
#   scripts/queue.sh "<new> <old> [elo1 [elo0]]" ...
#
# Each argument is one test; <new> and <old> are engine binaries or git refs
# (see sprt.sh). Summaries are appended to engines/results.log. The time
# control and other settings come from the same environment variables as
# sprt.sh, e.g. TC=5+0.05 scripts/queue.sh "feature main 5".
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG="$ROOT/engines/results.log"
for spec in "$@"; do
    read -r new old elo1 elo0 <<<"$spec"
    out="$(ELO1="${elo1:-5}" ELO0="${elo0:-0}" "$ROOT/scripts/sprt.sh" "$new" "$old" 2>&1 | sed 's/\x1b\[[0-9;]*m//g')"
    {
        echo "=== $(basename "$new") vs $(basename "$old")  [${elo0:-0}, ${elo1:-5}]  ${NODES:+$NODES nodes  }$(date '+%F %T')"
        echo "$out" | grep -E '^(Elo|Games|Ptnml|LLR|SPRT)' | tail -5
        echo "$out" | grep -ciE 'illegal|disconnect|stall|on time|timeout|crash' | sed 's/^/problem lines: /'
    } >>"$LOG"
done
echo "queue finished"
tail -40 "$LOG"
