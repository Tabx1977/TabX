#!/usr/bin/env python3
"""Joins the shards that "tabx import-evals --shards N" wrote into one shuffled file.

    scripts/join_shards.py data/lichess-lines6 data/lichess-all6.train.bin

The importer deals its records out at random among the files
data/lichess-lines6.000, .001 and so on. Each of them is read into memory,
shuffled and added to the output, and removed once it is there, so that the
disk never holds much more than one copy of the data. The result is as
well shuffled as one big shuffle would make it: which shard a record is in
was drawn at random, and so is its place in the shard.

Needs numpy.
"""
import glob
import os
import sys

import numpy as np

RECORD = 32

prefix, out = sys.argv[1], sys.argv[2]
shards = sorted(glob.glob(prefix + ".[0-9][0-9][0-9]"))
if not shards:
    sys.exit(f"no files {prefix}.000, ...")
if os.path.exists(out):
    sys.exit(f"{out} is there already")
rng = np.random.default_rng(20261125)
total = 0
with open(out + ".part", "wb") as stream:
    for number, path in enumerate(shards):
        size = os.path.getsize(path)
        if size % RECORD:
            sys.exit(f"{path} is not a whole number of records")
        records = np.fromfile(path, dtype=np.dtype((np.void, RECORD)))
        rng.shuffle(records)
        records.tofile(stream)
        stream.flush()
        os.fsync(stream.fileno())
        total += len(records)
        os.remove(path)
        print(f"{number + 1} of {len(shards)} shards joined, {total} records", flush=True)
os.replace(out + ".part", out)
print(f"{out}: {total} records")
