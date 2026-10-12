"""Willy_Chess benchmark: runs the suites with an adapter and compares against a baseline.

    python -m bench.run                                   # reference, all suites, comparison with baseline.json
    python -m bench.run --adapter bench/adapters/mine.py  # your implementation
    python -m bench.run --quick --suite board mapping     # quick partial run (1/5 of the scenes)
    python -m bench.run --write-baseline bench/baseline_local.json   # create the baseline on your machine

Exit code 1 if a count metric is significantly worse than the baseline (one-sided Fisher test, p < 0.05).
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import platform
import sys
import time
from datetime import date
from pathlib import Path

import chess
import cv2 as cv
import numpy as np
import scipy
from scipy.stats import beta, fisher_exact

from bench.suites import SUITES

HERE = Path(__file__).resolve().parent
LOWER_IS_BETTER = {"shifted", "flagged", "silently wrong", "miscalibrated"}


def load_adapter(spec: str):
    if spec in ("reference", "original"):
        return importlib.import_module(f"bench.adapters.{spec}")
    path = Path(spec).resolve()
    mod_spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    return mod


def clopper_pearson(k, n, conf=0.95):
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else beta.ppf(a, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - a, k + 1, n - k)
    return lo, hi


def worse(metric, k, n, k0, n0):
    """One-sided Fisher test: is the adapter significantly worse than the baseline?"""
    if n == 0 or n0 == 0:
        return False, 1.0
    if metric in LOWER_IS_BETTER:
        table = [[k, n - k], [k0, n0 - k0]]
    else:
        table = [[n - k, k], [n0 - k0, k0]]           # failures of the adapter against those of the baseline
    p = fisher_exact(table, alternative="greater")[1]
    return p < 0.05, p


def versions():
    return {"python": platform.python_version(), "numpy": np.__version__, "opencv": cv.__version__,
            "scipy": scipy.__version__, "chess": chess.__version__}


def run(adapter, names, quick, dump):
    out = {}
    for name in names:
        fn, n, needs = SUITES[name]
        if not hasattr(adapter, needs):
            print(f"\n{name}: skipped, adapter has no `{needs}`")
            continue
        n_run = max(1, n // 5) if quick else n
        t0 = time.time()
        try:
            rows, fp = fn(adapter, n_run, dump / name if dump else None)
        except NotImplementedError:
            print(f"\n{name}: skipped, `{needs}` is not implemented yet")
            continue
        out[name] = {"fingerprint": fp, "quick": quick, "seconds": round(time.time() - t0, 1),
                     "rows": {r.name: {"counts": r.counts, "values": {m: float(np.mean(v)) for m, v in r.values.items()},
                                       "max": {m: float(np.max(v)) for m, v in r.values.items()},
                                       "failures": r.failures} for r in rows}}
    return out


def report(results, base, base_name):
    regress = False
    for suite, res in results.items():
        b = (base or {}).get(suite)
        same = b is not None and b.get("fingerprint") == res["fingerprint"] and b.get("quick") == res["quick"]
        tag = "" if b is None else ("  [inputs identical]" if same else "  [inputs differ: statistical comparison only]")
        print(f"\n== {suite}  ({res['seconds']} s){tag}")
        for row, r in res["rows"].items():
            br = (b or {}).get("rows", {}).get(row)
            for m, (k, n) in r["counts"].items():
                lo, hi = clopper_pearson(k, n)
                line = f"  {row:24s} {m:16s} {k:5d}/{n:<5d} {100 * k / max(n, 1):6.2f} %  [{100 * lo:5.1f}; {100 * hi:5.1f}]"
                if br and m in br["counts"]:
                    k0, n0 = br["counts"][m]
                    bad, p = worse(m, k, n, k0, n0)
                    regress |= bad
                    line += f"   {base_name}: {100 * k0 / max(n0, 1):6.2f} %  " + ("WORSE (p = %.3f)" % p if bad else "ok")
                print(line)
            for m, v in r["values"].items():
                line = f"  {row:24s} {m:16s} {v:13.3f}   (max {r['max'][m]:.3f})"
                if br and m in br["values"]:
                    line += f"   {base_name}: {br['values'][m]:.3f}"
                print(line)
    return regress


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="reference", help="reference, original or path to a .py file")
    ap.add_argument("--suite", nargs="*", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--quick", action="store_true", help="1/5 of the scenes, same seeds (subset of the full run)")
    ap.add_argument("--baseline", default=str(HERE / "baseline.json"))
    ap.add_argument("--against", default="reference", help="Baseline entry to compare against")
    ap.add_argument("--write-baseline", metavar="PATH", help="Save the result as the adapter's baseline entry")
    ap.add_argument("--out", metavar="PATH", help="Save the result as JSON")
    ap.add_argument("--dump-failures", metavar="DIR", help="Store failed scenes as PNG + NPZ")
    a = ap.parse_args(argv)

    adapter = load_adapter(a.adapter)
    name = getattr(adapter, "NAME", Path(a.adapter).stem)
    print(f"Adapter: {name}   Suites: {', '.join(a.suite)}   {'quick' if a.quick else 'full'}")
    results = run(adapter, a.suite, a.quick, Path(a.dump_failures) if a.dump_failures else None)

    base = None
    bpath = Path(a.baseline)
    if bpath.exists():
        data = json.loads(bpath.read_text(encoding="utf-8"))
        entry = data.get("adapters", {}).get(a.against)
        if entry is not None:
            base = entry["quick" if a.quick else "full"] if ("quick" in entry or "full" in entry) else None
            if base is None:
                print(f"Baseline {bpath.name} has no {'quick' if a.quick else 'full'} run for '{a.against}'.")
    regress = report(results, base, a.against)

    doc = {"meta": versions() | {"date": str(date.today()), "adapter": name}, "suites": results}
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    if a.write_baseline:
        p = Path(a.write_baseline)
        data = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"adapters": {}}
        entry = data["adapters"].setdefault(name, {})
        entry.setdefault("quick" if a.quick else "full", {}).update(results)
        data.setdefault("meta", {})[name] = doc["meta"]
        p.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nBaseline written: {p}")
    if regress:
        print("\nResult: at least one metric is significantly worse than the baseline.")
    return 1 if regress else 0


if __name__ == "__main__":
    sys.exit(main())
