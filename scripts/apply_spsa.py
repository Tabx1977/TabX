#!/usr/bin/env python3
"""Writes tuned values from an SPSA state file into src/params.rs.

    scripts/apply_spsa.py [state.json]        (default: engines/spsa-state.json)

Only the defaults change; the allowed range and step of each parameter stay
as they are. Rebuild and confirm the result with scripts/sprt.sh afterwards.
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
state_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "engines/spsa-state.json")
values = json.load(open(state_path))["values"]
path = os.path.join(ROOT, "src/params.rs")
source = open(path).read()
for name, value in values.items():
    pattern = re.compile(rf"^(\s*{re.escape(name)} = )(-?\d+)(,)", re.MULTILINE)
    match = pattern.search(source)
    if not match:
        sys.exit(f"parameter {name} not found in src/params.rs")
    new = round(value)
    if int(match.group(2)) != new:
        print(f"{name}: {match.group(2)} -> {new}")
    source = pattern.sub(rf"\g<1>{new}\g<3>", source, count=1)
open(path, "w").write(source)
