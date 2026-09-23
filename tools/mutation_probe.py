#!/usr/bin/env python3
"""Mutation probe — can the test suite actually *see* a broken defence rule?

The parity claim ("the port fires exactly when `INDICATOR.txt` says
`DEFENCE CONFIRMED`") is only worth as much as the tests behind it.  A gate the
suite never exercises is a gate that can be deleted in silence: the engine still
runs, still emits `confirmed` events, and every test still passes — they just
stop meaning anything for that gate.

This tool deletes each gate **one at a time** from the `defence` expression in
`precision_tap/engine.py` (and checks it is put back afterwards) and runs the
suite.  A gate whose removal the suite does not notice is reported as an
escape:

    python tools/mutation_probe.py              # all mutants, pytest -x
    python tools/mutation_probe.py --only clv   # one of them
    python tools/mutation_probe.py --keep-going # full run per mutant

Exit status is 0 only when every mutant was caught, so it can gate a release.
Run it from the repository root; the suite takes ~10 s per mutant.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENGINE = ROOT / "precision_tap" / "engine.py"

#: name -> (exact source fragment in the `defence` condition, replacement)
MUTANTS = {
    # barstate.isconfirmed — a forming bar may never confirm
    "closed_bar": ("if confirmed and pending", "if pending"),
    # pending = state == 1 and tapBar >= 0 and bar_index - tapBar <= confirmBars
    "pending_window": ("(i - z.tap_bars[-1]) <= p.confirm_bars", "True"),
    "close_gt_open": ("and close_i > open_i", ""),
    "clv": ("clv_i >= p.confirm_clv", "True"),
    "rvol": ("rvol_i >= p.confirm_rvol", "True"),
    "close_gt_top": ("and close_i > z.top", ""),
    "micro_bos": ("and (math.isfinite(bosv) and close_i > bosv)", ""),
}


def run_suite(keep_going: bool) -> subprocess.CompletedProcess:
    args = [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"]
    if not keep_going:
        args += ["-x"]
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def first_failure(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("FAILED") or line.startswith("ERROR"):
            return line.strip()
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", action="append", choices=sorted(MUTANTS), default=None,
                    help="run just these mutants (repeatable)")
    ap.add_argument("--keep-going", action="store_true",
                    help="do not stop pytest at the first failure")
    args = ap.parse_args(argv)

    original = ENGINE.read_text()
    names = args.only or list(MUTANTS)
    escapes = []
    try:
        for name in names:
            needle, replacement = MUTANTS[name]
            if needle not in original:
                print(f"{name:16s} SKIPPED — {needle!r} is not in engine.py (was the rule reworded?)")
                escapes.append(name)
                continue
            ENGINE.write_text(original.replace(needle, replacement, 1))
            try:
                r = run_suite(args.keep_going)
            finally:
                ENGINE.write_text(original)
            if r.returncode == 0:
                print(f"{name:16s} ESCAPED — the suite passes with this gate deleted")
                escapes.append(name)
            else:
                why = first_failure(r.stdout) or (r.stdout.strip().splitlines() or [""])[-1]
                print(f"{name:16s} caught   {why[:110]}")
    finally:
        ENGINE.write_text(original)

    if ENGINE.read_text() != original:                        # pragma: no cover
        print("engine.py was not restored!", file=sys.stderr)
        return 2
    print(f"\n{len(names) - len(escapes)}/{len(names)} mutants caught")
    if escapes:
        print("escaped: " + ", ".join(escapes))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
