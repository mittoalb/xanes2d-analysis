# xanes2d-analysis: analysing XANES 2D scans

**xanes2d-analysis** opens the files written by **XANES 2D** (and by
DataMap's 2D XANES action) and turns them into spectra and maps:

1. every image is divided by its own reference image (flat-field
   correction);
2. the images are **registered** (aligned) so the sample is in the same
   place at every energy;
3. you click a point or draw regions and get their **spectra**;
4. you look at **edge-energy** and **edge-step maps**;
5. you **compare** with reference spectra of standards and **export** the
   results.

```{figure} images/main_window.png
:alt: The xanes2d-analysis window with an image, two spectra and a reference
:width: 100%

The window: the image at the selected energy (left), with a region (orange)
and a clicked point (green); their normalised spectra and a Ni foil
reference from the library (right). Synthetic demonstration data.
```

## Quick start

1. Start it: `xanes2d-analysis`, or `xanes2d-analysis my_scan_001.h5`.
2. **Open…** the XANES 2D file. Keep **Binning** at 2 for large images.
3. **Register…** → **Compute shifts** → check the plot → **Apply**.
4. Optionally **Find sample** and tick **Subtract background** (see
   {doc}`background`).
5. Click the sample, or use **+ Rectangle** / **+ Ellipse** / **+ Polygon**.
6. Tick **Normalize (edge step)** and drag the blue (pre-edge) and red
   (post-edge) ranges on the plot.
7. **Load reference…** to compare, and **Export spectra…** to save.

```{toctree}
:maxdepth: 2

opening
registration
background
sam3
spectra
maps
references
export
troubleshooting
```
