# Opening a scan

## Starting the program

xanes2d-analysis is a separate program. Start it from a terminal in the
environment where it is installed:

```bash
xanes2d-analysis                          # then Open…
xanes2d-analysis my_scan_001.h5 --bin 2   # open a file straight away
```

It is installed with `pip install .` in the `xanes2d-analysis` folder.

## Which files it opens

Files written by **XANES 2D** (one file per scan, `…_001.h5`, `…_002.h5`)
and by DataMap's 2D XANES action. A file must contain the sample images
and the reference images; otherwise the program says it is not a XANES 2D
file.

Scans taken with **Reference every N energies** in XANES 2D are fine: each
image is matched to the reference recorded for it. A scan that was stopped
early opens with the energies that were completed.

## Toolbar settings for loading

| Control | What it does |
|---|---|
| **Binning** | Averages 2×2 or 4×4 pixels while loading (1 = full resolution). Default 2. Binning makes loading and registration faster, needs less memory and reduces noise per pixel. A scan of about 100 energies at full camera resolution needs about 3 GB of memory. |
| **Dark** | Camera dark level in counts, subtracted from images and references. XANES 2D files contain no dark images, so leave 0 unless you know the offset. |
| **Reload** | Opens the same file again with the current **Binning** / **Dark** (this also undoes any registration). |

The status bar shows what was loaded: number of energies, energy range,
image size and the number of reference images.

## Flat-field correction

Each image is divided by **its own reference image** while the file is
loaded, before anything else. The reference image records the beam
profile on the camera, which stays put while the sample drifts, so this
has to happen before the images are aligned.

The **Flat-field correction** checkbox (toolbar) switches what you see and
use:

| | On (default) | Off |
|---|---|---|
| Image | absorption −ln(I/I0), or transmission I/I0 | the raw images (minus dark): −ln(I), or counts |
| Spectra | mean of −ln(I/I0) | mean of −ln(I) |
| What **Register…** compares | the corrected images | the raw images |

Registration shifts are applied to both, so you can switch at any time.
Turn it off only to check the raw data, for example when the references
look wrong.

## The image

- **Energy** slider: steps through the energies; the label shows the
  energy and its number.
- The drop-down below the image chooses **Absorption** or
  **Transmission**, or one of the maps (see {doc}`maps`).
- Drag the bar on the histogram (right of the image) to change the
  contrast.
- Hover over the image: the status bar shows row, column and value.
- Mouse wheel zooms; drag the background to pan.

## Scale bar

Tick **Scale bar** under the image; **…** opens its settings:

| Setting | Meaning |
|---|---|
| **Pixel size (full resolution)** | Size of one pixel of the file at the sample, in nm. Default 16 nm. It is multiplied by the **Binning** you loaded with. |
| **Length** | Bar length and unit (nm, µm, mm). |
| **Colour**, **Thickness**, **Text size**, **Position** | Appearance and corner. |

The settings are remembered.

:::{note}
XANES 2D files store a pixel size recorded at scan time
(`32id:TXMOptics:ImagePixelSize`). It is shown in the settings for
reference only and **not used**, because it is not always right. Enter
the pixel size you trust.
:::
