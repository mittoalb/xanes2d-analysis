# xanes2d-analysis

**User guide:** section "xanes2d-analysis" of the 32-ID TXM Software User
Guide (https://github.com/mittoalb/32id-gui-docs, `docs/xanes2d_analysis/`).

Analysis of XANES 2D scans from the 32-ID TXM (files written by the XANES 2D
GUI and by pystream's DataMap).

1. **Open** a XANES 2D HDF5 file. Every image is turned into absorption,
   −ln(I/I0), using its own reference image (also when references were
   taken only every N energies). Optional 2×/4× binning and dark level.
2. **Register** the images: sub-pixel phase correlation of every image to a
   reference image (or chained through neighbouring energies), optionally
   only inside a region you place on a sharp feature. You see the shifts
   before applying them.
3. **Spectra**: click a point (N×N average) or add rectangle / ellipse /
   polygon regions; each has its own spectrum, updated when you move it.
4. **Normalize** (pre-edge line, edge step from the post-edge line) with
   ranges you drag on the plot.
5. **Compare** with a reference spectrum, with an energy shift and the
   R-factor against the selected spectrum. **Load reference…** opens the
   local reference library (see below), or any file: XDI, or two columns
   (eV or keV, text or .npy — e.g. the XANES GUI `Calibrated/` curves).
6. **Save / load shifts** (CSV, full-resolution pixels vs energy) to reuse
   one scan's registration on other scans that drift the same way.
7. **Edge-energy and edge-step maps** (per-pixel half-step energy / edge
   jump, with masking), and a scale bar.
8. **Export** spectra to CSV (with ROI definitions, registration and
   normalization settings in the header) and the aligned stack to HDF5.

## Install and run

```bash
pip install .
xanes2d-analysis                       # or: xanes2d-analysis scan_001.h5 --bin 2
python -m xanes2d_analysis scan_001.h5
```

## Reference spectra library

Measured spectra of standards from the **IXAS XAS Data Library**
(https://xaslib.xrayabsorption.org, public domain): 277 spectra, K edges
S–Cd and L edges of Ce, Eu, W, Pt, Au, Pb (about 6 MB).

- The app downloads the library in the background **on its first start**
  (when the local library is empty). Set `XANES2D_NO_AUTO_DOWNLOAD=1` to
  turn this off.
- Or download / update it any time: `xanes2d-refs download`
  (`--elements Ni Fe` for some only), or **Load reference… → Download /
  update library**. Files already present are skipped.
- `xanes2d-refs list [--element Ni]` lists it; `xanes2d-refs where` shows
  the folder: `~/.xanes2d_analysis/references` (set `XANES2D_REFS` to use
  another, e.g. a shared one for the beamline).

`pip install` itself does not download anything (pip builds wheels and skips
custom install steps); the first start or `xanes2d-refs download` does.

Tests: `pip install .[test] && pytest tests`.
