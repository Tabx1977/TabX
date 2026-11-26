#!/usr/bin/env bash
# Trains an evaluation network from self-play data.
#
#   scripts/train.sh <name> <data.bin> [trainer options...]
#
# Example:
#   scripts/train.sh gen1 data/gen1.bin --superbatches 40 --batches 1500 --wdl 0.5
#
# Steps: shuffle the data, train on the GPU, copy the quantised network to
# nets/<name>.bin. Needs the CUDA runtime libraries under tools/cuda (or set
# CUDA_PATH yourself) and bullet's utilities under tools/bullet.
#
# ARCH selects the network architecture (a binary in trainer/src/bin):
#   simple    (768 -> 256)x2 -> 1                      [default]
#   bucketed  (768x4 mirrored -> 512)x2 -> 8 buckets
# The constants in src/nnue.rs must describe the same architecture.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NAME="${1:?usage: scripts/train.sh <name> <data.bin> [trainer options...]}"
DATA="$(realpath "${2:?missing data file}")"
shift 2

export CUDA_PATH="${CUDA_PATH:-$ROOT/tools/cuda}"
export LD_LIBRARY_PATH="$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
UTILS="$ROOT/tools/bullet/target/release/bullet-utils"

SHUFFLED="${DATA%.bin}.shuffled.bin"
if [[ ! -f "$SHUFFLED" || "$DATA" -nt "$SHUFFLED" ]]; then
    echo "shuffling $DATA"
    "$UTILS" shuffle --input "$DATA" --output "$SHUFFLED" --mem-used-mb 16384
fi

cd "$ROOT/trainer"
cargo run --release --bin "${ARCH:-simple}" -- --data "$SHUFFLED" --id "$NAME" "$@"

LATEST="$(ls -d "checkpoints/$NAME"-* | sort -t- -k2 -n | tail -1)"
mkdir -p "$ROOT/nets"
cp "$LATEST/quantised.bin" "$ROOT/nets/$NAME.bin"
echo "saved $ROOT/nets/$NAME.bin (from $LATEST)"
