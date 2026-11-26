#!/usr/bin/env python3
"""Held-out loss of a trained network on datagen records.

usage: scripts/valloss.py [--mirrored | --four] <net.bin> <heldout.bin> [max_positions]

Without options the network is the plain (768 -> H)x2 -> 1 layout. With
--mirrored it is the mirrored layout with one set of input weights and 8
output buckets; with --four the mirrored layout with four king buckets.

Prints the mean squared error of the network's win-probability prediction
against (a) the game result, (b) the search score, and (c) their 50/50 blend.
Lower is better. Use positions the network was not trained on.
"""
import sys
import numpy as np

QA, QB, SCALE = 255, 64, 400
REC = np.dtype([("occ", "u1", 8), ("pcs", "u1", 16), ("score", "<i2"), ("result", "u1"),
                ("ksq", "u1"), ("opp_ksq", "u1"), ("extra", "u1", 3)])


OUT_BUCKETS = 8
# King bucket per king square (files a-d, rank 1 first) for the "four" layout,
# expanded to all 64 squares with files e-h mirrored. Must match the trainer.
LAYOUT_FOUR = [0, 0, 1, 1, 2, 2, 2, 2] + [3] * 24
KING_BUCKET_FOUR = np.array([LAYOUT_FOUR[(sq // 8) * 4 + min(sq % 8, 7 - sq % 8)] for sq in range(64)])


def load_net(path, mirrored, in_buckets):
    """Returns hidden size, first-layer weights and bias, and per-bucket output weights and biases."""
    raw = np.fromfile(path, dtype="<i2")
    buckets = OUT_BUCKETS if mirrored else 1
    inputs = 768 * in_buckets
    per_hidden, extra = inputs + 1 + 2 * buckets, buckets
    hidden = (raw.size - extra) // per_hidden
    while (per_hidden * hidden + extra + 31) // 32 * 32 != raw.size:
        hidden -= 1
        assert hidden > 0, "file size does not match the expected layout"
    w0 = raw[: inputs * hidden].reshape(inputs, hidden).astype(np.float32)
    b0 = raw[inputs * hidden: (inputs + 1) * hidden].astype(np.float32)
    w1 = raw[(inputs + 1) * hidden: (inputs + 1 + 2 * buckets) * hidden].reshape(buckets, 2 * hidden).astype(np.float32)
    b1 = raw[(inputs + 1 + 2 * buckets) * hidden: (inputs + 1 + 2 * buckets) * hidden + buckets].astype(np.float32)
    return hidden, w0, b0, w1, b1


def features(records, mirrored, in_buckets):
    """Dense 0/1 input matrices for the side to move and the other side."""
    n = len(records)
    bits = np.unpackbits(records["occ"], axis=1, bitorder="little").astype(bool)  # [n, 64]
    order = np.cumsum(bits, axis=1) - 1                                            # nibble index per square
    nibbles = np.empty((n, 32), dtype=np.uint8)
    nibbles[:, 0::2] = records["pcs"] & 15
    nibbles[:, 1::2] = records["pcs"] >> 4
    rows, squares = np.nonzero(bits)
    piece = nibbles[rows, order[rows, squares]]
    # Widen before multiplying: these are uint8 and 64 * 5 does not fit.
    theirs, ptype = (piece >> 3).astype(np.int64), (piece & 7).astype(np.int64)
    stm = np.zeros((n, 768 * in_buckets), dtype=np.float32)
    ntm = np.zeros((n, 768 * in_buckets), dtype=np.float32)
    stm_index = np.where(theirs == 1, 384, 0) + 64 * ptype + squares
    ntm_index = np.where(theirs == 1, 0, 384) + 64 * ptype + (squares ^ 56)
    if mirrored:
        # Each side's view is mirrored when its own king is on files e-h.
        stm_index ^= np.where(records["ksq"] % 8 > 3, 7, 0)[rows]
        ntm_index ^= np.where(records["opp_ksq"] % 8 > 3, 7, 0)[rows]
    if in_buckets > 1:
        stm_index += 768 * KING_BUCKET_FOUR[records["ksq"]][rows]
        ntm_index += 768 * KING_BUCKET_FOUR[records["opp_ksq"]][rows]
    stm[rows, stm_index] = 1
    ntm[rows, ntm_index] = 1
    return stm, ntm


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    in_buckets = 4 if "--four" in sys.argv else 1
    mirrored = "--mirrored" in sys.argv or in_buckets > 1
    net_path, data_path = args[0], args[1]
    limit = int(args[2]) if len(args) > 2 else 400_000
    hidden, w0, b0, w1, b1 = load_net(net_path, mirrored, in_buckets)
    data = np.fromfile(data_path, dtype=REC)[:limit]
    preds = []
    for start in range(0, len(data), 10_000):
        chunk = data[start:start + 10_000]
        stm, ntm = features(chunk, mirrored, in_buckets)
        a = np.clip(stm @ w0 + b0, 0, QA)
        b = np.clip(ntm @ w0 + b0, 0, QA)
        if mirrored:
            pieces = np.unpackbits(chunk["occ"], axis=1).sum(axis=1)
            bucket = (pieces - 2) // 4
        else:
            bucket = np.zeros(len(chunk), dtype=np.int64)
        hidden_layer = np.concatenate([a * a, b * b], axis=1)
        out = np.einsum("ij,ij->i", hidden_layer, w1[bucket]) / QA + b1[bucket]
        preds.append(1 / (1 + np.exp(-(out * SCALE / (QA * QB)) / SCALE)))
    pred = np.concatenate(preds)
    result = data["result"].astype(np.float32) / 2
    score = 1 / (1 + np.exp(-data["score"].astype(np.float32) / SCALE))
    print(f"{net_path}: hidden {hidden}, {len(data)} positions | "
          f"mse vs result {np.mean((pred - result) ** 2):.5f} | "
          f"vs score {np.mean((pred - score) ** 2):.5f} | "
          f"vs blend {np.mean((pred - (result + score) / 2) ** 2):.5f}")


if __name__ == "__main__":
    main()
