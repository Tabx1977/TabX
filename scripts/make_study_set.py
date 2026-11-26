#!/usr/bin/env python3
"""Builds a training set that leaves a fixed block of games out for validation.

    scripts/make_study_set.py <out-prefix> <holdout-start> <holdout-count> <main.bin> [more.bin ...]

Writes <out-prefix>.hold.bin (records [start, start+count) of <main.bin>) and
<out-prefix>.train.shuffled.bin (everything else in <main.bin>, plus all of
the other files, shuffled with bullet-utils). Records are 32 bytes; a partial
trailing record is ignored. The held-out records are whole games as long as
<main.bin> is still in the order the generator wrote it.
"""
import os
import subprocess
import sys

RECORD = 32
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def copy(src, dst, start, count):
    """Appends `count` records of `src`, starting at record `start`, to the open file `dst`."""
    with open(src, "rb") as f:
        f.seek(start * RECORD)
        left = count * RECORD
        while left > 0:
            chunk = f.read(min(left, 1 << 24))
            if not chunk:
                break
            dst.write(chunk)
            left -= len(chunk)


def main():
    out, start, count, main_file, extras = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4], sys.argv[5:]
    total = os.path.getsize(main_file) // RECORD
    if start + count > total:
        sys.exit(f"{main_file} has only {total} records")
    train = out + ".train.bin"
    with open(train, "wb") as dst:
        copy(main_file, dst, 0, start)
        copy(main_file, dst, start + count, total - start - count)
        for extra in extras:
            copy(extra, dst, 0, os.path.getsize(extra) // RECORD)
    with open(out + ".hold.bin", "wb") as dst:
        copy(main_file, dst, start, count)
    shuffled = out + ".train.shuffled.bin"
    subprocess.run([os.path.join(ROOT, "tools/bullet/target/release/bullet-utils"), "shuffle",
                    "--input", train, "--output", shuffled, "--mem-used-mb", "16384"],
                   check=True, stdout=subprocess.DEVNULL)
    os.remove(train)
    print(f"{os.path.getsize(shuffled) // RECORD} training positions, {count} held out")


if __name__ == "__main__":
    main()
