#!/usr/bin/env python3
"""Tunes the engine's search parameters by self-play (SPSA).

    scripts/spsa.py [--iterations N] [--pairs N] [--tc 5+0.05 | --nodes N]
                    [--concurrency N] [--state FILE] [--only a,b,...] [--skip a,b,...]

With --nodes the games use a fixed node budget per move instead of a clock.
That is much faster and unaffected by other load on the machine, but it
cannot tune anything time-related (skip the tm_* parameters), and values
found at a small budget should be confirmed on the clock.

Each iteration nudges every parameter up or down by a small random amount,
plays a few game pairs between the "up" and "down" versions, and moves the
parameters a little towards the side that scored better. Thousands of
iterations are needed for the values to settle; progress is saved to the
state file after every iteration, so the run can be stopped and resumed.

Needs a tuning build of the engine (built automatically) and fastchess with
an opening book under tools/, like scripts/sprt.sh.

When it finishes (or at any point, from the state file) copy the printed
values into src/params.rs and confirm the gain with scripts/sprt.sh.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTCHESS = os.path.join(ROOT, "tools/fastchess/fastchess")
BOOK = os.path.join(ROOT, "tools/books/UHO_Lichess_4852_v1.epd")
ENGINE = os.path.join(ROOT, "target/tune/release/tabx")

# Standard SPSA decay exponents and the end-of-run learning rate (as a
# fraction of the squared step), following common engine-testing practice.
ALPHA, GAMMA, R_END = 0.602, 0.101, 0.002


def build_engine():
    subprocess.run(["cargo", "build", "--release", "--features", "tune", "--target-dir", "target/tune"],
                   cwd=ROOT, check=True)


def read_params():
    out = subprocess.run([ENGINE], input="params\nquit\n", capture_output=True, text=True).stdout
    params = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) == 5 and parts[0] != "info":
            name, default, lo, hi, step = parts[0], int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])
            params[name] = {"value": float(default), "min": lo, "max": hi, "step": step}
    if not params:
        sys.exit("the engine listed no parameters; is it a tuning build?")
    return params


def play(plus, minus, args, seed):
    """Plays game pairs between two parameter sets; returns (wins, losses, draws) for `plus`."""
    def engine(name, values):
        options = [f"option.{k}={v}" for k, v in values.items()]
        return ["-engine", f"cmd={ENGINE}", f"name={name}"] + options

    cmd = ([FASTCHESS] + engine("plus", plus) + engine("minus", minus) +
           ["-each"] + (["tc=inf", f"nodes={args.nodes}"] if args.nodes else [f"tc={args.tc}"]) +
           ["option.Hash=16",
            "-openings", f"file={BOOK}", "format=epd", "order=random", "-srand", str(seed),
            "-rounds", str(args.pairs), "-repeat", "-concurrency", str(args.concurrency),
            "-draw", "movenumber=40", "movecount=8", "score=10",
            "-resign", "movecount=3", "score=600", "twosided=true",
            "-recover", "-autosaveinterval", "0", "-ratinginterval", "0"])
    out = subprocess.run(cmd, capture_output=True, text=True, cwd="/tmp").stdout
    match = re.findall(r"Games: \d+, Wins: (\d+), Losses: (\d+), Draws: (\d+)", out)
    if not match:
        sys.exit("could not read the match result from fastchess:\n" + out[-2000:])
    wins, losses, draws = map(int, match[-1])
    return wins, losses, draws


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--pairs", type=int, default=8, help="game pairs per iteration")
    parser.add_argument("--tc", default="5+0.05")
    parser.add_argument("--nodes", type=int, default=0, help="fixed nodes per move instead of a clock")
    parser.add_argument("--skip", default="", help="comma-separated name prefixes to leave untouched")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--state", default=os.path.join(ROOT, "engines/spsa-state.json"))
    parser.add_argument("--only", default="", help="comma-separated parameter names to tune (default: all)")
    args = parser.parse_args()

    build_engine()
    params = read_params()
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - set(params)
        if unknown:
            sys.exit(f"unknown parameters: {', '.join(sorted(unknown))}")
        params = {k: v for k, v in params.items() if k in wanted}
    if args.skip:
        prefixes = tuple(args.skip.split(","))
        params = {k: v for k, v in params.items() if not k.startswith(prefixes)}

    start = 1
    if os.path.exists(args.state):
        state = json.load(open(args.state))
        for name, value in state["values"].items():
            if name in params:
                params[name]["value"] = value
        start = state["iteration"] + 1
        print(f"resuming from iteration {start} ({args.state})")

    total = args.iterations
    stability = 0.1 * total
    rng = random.Random()
    for k in range(start, total + 1):
        plus, minus, signs, steps = {}, {}, {}, {}
        for name, p in params.items():
            c_k = p["step"] * total ** GAMMA / k ** GAMMA
            sign = rng.choice((-1, 1))
            clamp = lambda v: max(p["min"], min(p["max"], v))
            plus[name] = round(clamp(p["value"] + c_k * sign))
            minus[name] = round(clamp(p["value"] - c_k * sign))
            signs[name], steps[name] = sign, c_k

        wins, losses, draws = play(plus, minus, args, rng.randrange(1 << 30))

        for name, p in params.items():
            a_end = R_END * p["step"] ** 2
            a_k = a_end * (stability + total) ** ALPHA / (stability + k) ** ALPHA
            p["value"] += a_k / steps[name] * (wins - losses) * signs[name]
            p["value"] = max(p["min"], min(p["max"], p["value"]))

        json.dump({"iteration": k, "values": {n: p["value"] for n, p in params.items()}},
                  open(args.state, "w"), indent=1)
        if k % 10 == 0 or k == total:
            print(f"iteration {k}/{total}  last batch +{wins} -{losses} ={draws}")
            print("  " + "  ".join(f"{n}={p['value']:.1f}" for n, p in params.items()))
            sys.stdout.flush()

    print("\nfinal values (rounded):")
    for name, p in params.items():
        print(f"    {name} = {round(p['value'])}")


if __name__ == "__main__":
    main()
