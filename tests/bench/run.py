"""Willy_Chess-Benchmark: führt die Suiten mit einem Adapter aus und vergleicht mit einer Baseline.

    python -m bench.run                                   # Referenz, alle Suiten, Vergleich mit baseline.json
    python -m bench.run --adapter bench/adapters/mine.py  # deine Implementierung
    python -m bench.run --quick --suite brett zuordnung   # schneller Teillauf (1/5 der Szenen)
    python -m bench.run --write-baseline bench/baseline_local.json   # Baseline auf deinem Rechner erzeugen

Exit-Code 1, wenn eine Zählmetrik signifikant schlechter ist als die Baseline (einseitiger Fisher-Test, p < 0,05).
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
LOWER_IS_BETTER = {"verschoben", "markiert", "still falsch", "falsch eingemessen"}


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
    """Einseitiger Fisher-Test: ist der Adapter signifikant schlechter als die Baseline?"""
    if n == 0 or n0 == 0:
        return False, 1.0
    if metric in LOWER_IS_BETTER:
        table = [[k, n - k], [k0, n0 - k0]]
    else:
        table = [[n - k, k], [n0 - k0, k0]]           # Misserfolge des Adapters gegen die der Baseline
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
            print(f"\n{name}: übersprungen, Adapter hat kein `{needs}`")
            continue
        n_run = max(1, n // 5) if quick else n
        t0 = time.time()
        try:
            rows, fp = fn(adapter, n_run, dump / name if dump else None)
        except NotImplementedError:
            print(f"\n{name}: übersprungen, `{needs}` ist noch nicht implementiert")
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
        tag = "" if b is None else ("  [Eingaben identisch]" if same else "  [Eingaben weichen ab: nur statistischer Vergleich]")
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
                    line += f"   {base_name}: {100 * k0 / max(n0, 1):6.2f} %  " + ("SCHLECHTER (p = %.3f)" % p if bad else "ok")
                print(line)
            for m, v in r["values"].items():
                line = f"  {row:24s} {m:16s} {v:13.3f}   (max {r['max'][m]:.3f})"
                if br and m in br["values"]:
                    line += f"   {base_name}: {br['values'][m]:.3f}"
                print(line)
    return regress


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="reference", help="reference, original oder Pfad zu einer .py-Datei")
    ap.add_argument("--suite", nargs="*", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--quick", action="store_true", help="1/5 der Szenen, gleiche Seeds (Teilmenge des vollen Laufs)")
    ap.add_argument("--baseline", default=str(HERE / "baseline.json"))
    ap.add_argument("--against", default="reference", help="Eintrag in der Baseline, gegen den verglichen wird")
    ap.add_argument("--write-baseline", metavar="PATH", help="Ergebnis als Baseline-Eintrag des Adapters speichern")
    ap.add_argument("--out", metavar="PATH", help="Ergebnis als JSON speichern")
    ap.add_argument("--dump-failures", metavar="DIR", help="Fehlgeschlagene Szenen als PNG + NPZ ablegen")
    a = ap.parse_args(argv)

    adapter = load_adapter(a.adapter)
    name = getattr(adapter, "NAME", Path(a.adapter).stem)
    print(f"Adapter: {name}   Suiten: {', '.join(a.suite)}   {'quick' if a.quick else 'voll'}")
    results = run(adapter, a.suite, a.quick, Path(a.dump_failures) if a.dump_failures else None)

    base = None
    bpath = Path(a.baseline)
    if bpath.exists():
        data = json.loads(bpath.read_text(encoding="utf-8"))
        entry = data.get("adapters", {}).get(a.against)
        if entry is not None:
            base = entry["quick" if a.quick else "full"] if ("quick" in entry or "full" in entry) else None
            if base is None:
                print(f"Baseline {bpath.name} hat für '{a.against}' keinen {'quick' if a.quick else 'vollen'} Lauf.")
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
        print(f"\nBaseline geschrieben: {p}")
    if regress:
        print("\nErgebnis: mindestens eine Metrik ist signifikant schlechter als die Baseline.")
    return 1 if regress else 0


if __name__ == "__main__":
    sys.exit(main())
