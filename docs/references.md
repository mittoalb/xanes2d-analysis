# Reference spectra

Compare your spectra with measured spectra of standards (metal foils,
oxides, salts).

## The reference library

xanes2d-analysis keeps a local copy of the **IXAS XAS Data Library**
(<https://xaslib.xrayabsorption.org>, public domain): 277 measured spectra,
K edges of S, V, Cr, Mn, Fe, Co, Ni, Cu, Zn, As, Se, Sr, Mo, Cd and L
edges of Ce, Eu, W, Pt, Au, Pb (about 6 MB).

- On its **first start** the program downloads the library in the
  background (about 2 minutes) while you work; the status bar shows the
  progress. Set the environment variable `XANES2D_NO_AUTO_DOWNLOAD=1` to
  turn this off.
- To download or update later: **Load reference… → Download / update
  library**, or in a terminal `xanes2d-refs download`
  (`--elements Ni Fe` for some only). Spectra already there are skipped.
- `xanes2d-refs list --element Ni` lists what you have;
  `xanes2d-refs where` prints the folder (`~/.xanes2d_analysis/references`;
  set `XANES2D_REFS` to use another, for example a shared folder on a
  computer without internet).

## Loading a reference

Click **Load reference…**:

```{figure} images/reference_library.png
:alt: Reference spectra library dialog
:width: 85%
```

1. Choose the **Element**. With **Only spectra covering this scan's
   energies** ticked, only spectra spanning your scan's energy range are
   listed.
2. Select a spectrum (sample, edge, beamline, file) and click
   **Load selected**, or double-click it.

**From file…** loads any other reference: XDI files (as from the IXAS
library or Athena/Larch exports) or a plain two-column file (energy,
absorption; text or `.npy`, energies in eV or keV — for example the XANES
GUI `Calibrated/` curves).

## Comparing

The reference is drawn as a white dashed line and its name appears in the
**Reference spectrum** box.

- **Energy shift**: moves the reference in energy. Energy calibration
  differs between beamlines by a few eV; set the shift so a standard you
  measured yourself (ideally a metal foil) lines up with the library one.
- Tick **Normalize (edge step)** so both are normalised the same way.
- **R-factor** (below the Spectra list): Σ(spectrum − reference)² /
  Σ spectrum² for the selected spectrum, over the energies both cover.
  Smaller is more similar.
- **Remove** takes the reference off the plot.

:::{tip}
Spectra of standards measured on 32-ID with XANES 2D (same resolution and
calibration) are the best references. Save them with
**Export spectra…** and load them with **From file…**.
:::
