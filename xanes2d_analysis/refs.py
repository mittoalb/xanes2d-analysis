"""Reference spectra: XDI reader and a local copy of the IXAS XAS Data Library.

The IXAS XAS Data Library (https://xaslib.xrayabsorption.org, public domain)
holds curated, measured spectra of standards in XDI format. `download_ixas`
copies them into the local library (default ~/.xanes2d_analysis/references,
or $XANES2D_REFS), skipping files already there, and keeps an index.json
with element, edge, sample and source for each spectrum.

Command line:  xanes2d-refs download [--elements Ni Fe]   |  list [--element Ni]
               xanes2d-refs where
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from typing import Callable, Dict, List, Optional

import numpy as np

IXAS_URL = "https://xaslib.xrayabsorption.org"
USER_AGENT = "xanes2d-analysis (reference library download)"
HC_EV_A = 12398.4198                      # eV·Å


def library_dir() -> str:
    return os.path.expanduser(os.environ.get("XANES2D_REFS", "~/.xanes2d_analysis/references"))


def _index_path() -> str:
    return os.path.join(library_dir(), "index.json")


# ── XDI ───────────────────────────────────────────────────────────────────

def read_xdi(path: str) -> Dict:
    """Read an XDI file. Returns {'energy': eV, 'mu': absorption, 'mu_kind':
    how mu was obtained, 'columns': {name: array}, 'meta': {Key.sub: value}}.

    Absorption, in order of preference: a mu / normalized column given in
    the file; ln(i0/itrans) (transmission); ifluor/i0 (fluorescence);
    ln(itrans/irefer) (reference channel)."""
    meta, names, units, rows = {}, {}, {}, []
    with open(path, errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                body = s.lstrip("#").strip()
                m = re.match(r"([A-Za-z_][\w]*\.[\w]+)\s*:\s*(.*)", body)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    meta[key] = val
                    cm = re.match(r"Column\.(\d+)$", key)
                    if cm:
                        parts = val.split()
                        names[int(cm.group(1))] = parts[0].lower() if parts else ""
                        units[int(cm.group(1))] = parts[1] if len(parts) > 1 else ""
                continue
            try:
                rows.append([float(v) for v in s.replace(",", " ").split()])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no data in {os.path.basename(path)}")
    ncol = min(len(r) for r in rows)
    data = np.array([r[:ncol] for r in rows], dtype=float)
    cols = {names.get(i + 1, f"col{i + 1}"): data[:, i] for i in range(ncol)}
    cunits = {names.get(i + 1, f"col{i + 1}"): units.get(i + 1, "") for i in range(ncol)}

    # energy
    if "energy" in cols:
        e = cols["energy"].copy()
        if cunits.get("energy", "").lower() == "kev" or np.nanmax(e) < 100:
            e *= 1000.0
    elif "angle" in cols and "Mono.d_spacing" in meta:
        d = float(meta["Mono.d_spacing"].split()[0])
        ang = cols["angle"]
        if cunits.get("angle", "deg").lower().startswith("rad"):
            ang = np.degrees(ang)
        e = HC_EV_A / (2 * d * np.sin(np.radians(ang)))
    else:
        e = data[:, 0].copy()

    mu, kind = None, ""
    with np.errstate(divide="ignore", invalid="ignore"):
        for name in ("mutrans", "mufluor", "mu", "xmu", "normtrans", "normfluor", "norm"):
            if name in cols:
                mu, kind = cols[name], name
                break
        if mu is None and "i0" in cols and "itrans" in cols:
            mu, kind = np.log(cols["i0"] / cols["itrans"]), "ln(i0/itrans)"
        if mu is None and "i0" in cols and "ifluor" in cols:
            mu, kind = cols["ifluor"] / cols["i0"], "ifluor/i0"
        if mu is None and "itrans" in cols and "irefer" in cols:
            mu, kind = np.log(cols["itrans"] / cols["irefer"]), "ln(itrans/irefer)"
    if mu is None:
        if ncol < 2:
            raise ValueError("no absorption column found")
        mu, kind = data[:, 1], f"column 2 ({names.get(2, '?')})"
    order = np.argsort(e)
    ok = np.isfinite(mu[order]) & np.isfinite(e[order])
    return {"energy": e[order][ok], "mu": np.asarray(mu)[order][ok], "mu_kind": kind,
            "columns": cols, "meta": meta}


def is_xdi(path: str) -> bool:
    try:
        with open(path, errors="replace") as fh:
            return fh.readline().lstrip("# ").upper().startswith("XDI")
    except OSError:
        return False


# ── library index ─────────────────────────────────────────────────────────

def load_index() -> List[Dict]:
    try:
        with open(_index_path()) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return []


def _save_index(entries: List[Dict]) -> None:
    os.makedirs(library_dir(), exist_ok=True)
    tmp = _index_path() + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(entries, fh, indent=1)
    os.replace(tmp, _index_path())


def _entry_for(path: str, source: str, url: str) -> Dict:
    meta = read_xdi(path)["meta"]
    return {
        "file": os.path.relpath(path, library_dir()),
        "element": meta.get("Element.symbol", ""),
        "edge": meta.get("Element.edge", ""),
        "sample": meta.get("Sample.name", "") or meta.get("Sample.formula", ""),
        "formula": meta.get("Sample.formula", ""),
        "beamline": meta.get("Beamline.name", ""),
        "source": source,
        "url": url,
    }


def entries(element: Optional[str] = None) -> List[Dict]:
    out = load_index()
    if element:
        out = [e for e in out if e["element"].lower() == element.lower()]
    return sorted(out, key=lambda e: (e["element"], e["edge"], e["sample"].lower()))


def entry_path(entry: Dict) -> str:
    return os.path.join(library_dir(), entry["file"])


# ── download (IXAS XAS Data Library) ──────────────────────────────────────

def _get(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def ixas_elements() -> List[str]:
    html = _get(f"{IXAS_URL}/elem/").decode("utf-8", "replace")
    return sorted({m for m in re.findall(r'href="/elem/([A-Z][a-z]?)"', html)})


def download_ixas(elements: Optional[List[str]] = None,
                  log: Callable[[str], None] = print,
                  progress: Optional[Callable[[int, int], None]] = None,
                  stop: Optional[Callable[[], bool]] = None,
                  delay: float = 0.15) -> int:
    """Copy IXAS library spectra into the local library. Files already
    present are skipped. Returns the number of new files."""
    root = os.path.join(library_dir(), "ixas")
    index = {e["file"]: e for e in load_index()}
    els = elements or ixas_elements()
    log(f"IXAS XAS Data Library: {len(els)} element(s)")
    todo = []
    for el in els:
        html = _get(f"{IXAS_URL}/elem/{el}").decode("utf-8", "replace")
        ids = sorted({int(i) for i in re.findall(r'href="/spectrum/(\d+)/"', html)})
        todo += [(el, i) for i in ids]
        time.sleep(delay)
    new = 0
    for k, (el, sid) in enumerate(todo, 1):
        if stop and stop():
            raise InterruptedError
        if progress:
            progress(k, len(todo))
        have = [f for f in index if f.startswith(f"ixas/{el}/{sid}_")]
        if have and os.path.exists(os.path.join(library_dir(), have[0])):
            continue
        try:
            page = _get(f"{IXAS_URL}/spectrum/{sid}/").decode("utf-8", "replace")
            m = re.search(r'href="(/rawfile/\d+/[^"]+)"', page)
            if not m:
                log(f"  {el} #{sid}: no file link, skipped")
                continue
            url = IXAS_URL + m.group(1)
            name = os.path.basename(m.group(1))
            path = os.path.join(root, el, f"{sid}_{name}")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(_get(url))
            entry = _entry_for(path, "IXAS XAS Data Library", url)
            index[entry["file"]] = entry
            new += 1
            if new % 20 == 0:
                _save_index(list(index.values()))
        except Exception as e:                # keep going; report
            log(f"  {el} #{sid}: {type(e).__name__}: {e}")
        time.sleep(delay)
    _save_index(list(index.values()))
    log(f"done: {new} new spectra, {len(index)} in the library ({library_dir()})")
    return new


# ── command line ──────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="xanes2d-refs",
                                 description="Local library of reference XANES spectra")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download", help="download / update from the IXAS XAS Data Library")
    d.add_argument("--elements", nargs="*", help="only these elements (default: all)")
    ls = sub.add_parser("list", help="list the spectra in the local library")
    ls.add_argument("--element")
    sub.add_parser("where", help="print the library folder")
    a = ap.parse_args(argv)
    if a.cmd == "download":
        download_ixas(a.elements)
    elif a.cmd == "list":
        for e in entries(a.element):
            print(f"{e['element']:>3} {e['edge']:<3} {e['sample'][:40]:<40} {e['beamline'][:20]:<20} {e['file']}")
    else:
        print(library_dir())
    return 0


if __name__ == "__main__":
    sys.exit(main())
