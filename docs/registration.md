# Registering the images

During a XANES 2D scan the sample image moves a little from energy to
energy (zone-plate moves, stage drift). Registration finds that movement
and shifts every image back, so a pixel is the same sample spot at all
energies. Without it, spectra of small features mix neighbouring
material, and maps are blurred at edges.

## Register…

1. Optionally move the **Energy** slider to the image you want as the
   reference (one with a clear, sharp sample, often above the edge).
2. Click **Register…**:

   ```{figure} images/register_dialog.png
   :alt: Register images dialog
   :width: 60%
   ```

   | Setting | Meaning |
   |---|---|
   | **Reference image** | The image the others are aligned to (number and energy). Starts at the one you are looking at. |
   | **Compare** | **Each image to the reference** (default), or **Each image to its neighbour (chained)** — see below. |
   | **Noise filter** | How strongly pixel noise is ignored (default 1.5 px; higher for noisy images, 0 = off). |
   | **Sub-pixel factor** | Precision: 1/this value of a pixel (default 20 → 0.05 px). |
   | **Only inside the registration region** | Use only the dashed box (see below). |

3. **Compute shifts**. A plot shows the shift found for every energy:

   ```{figure} images/shifts_preview.png
   :alt: Registration shifts preview
   :width: 80%
   ```

4. **Apply** shifts all images; **Discard** leaves them as they were.

The status bar then says which image was used and the largest shift.
You can register again; the new shifts are added to the previous ones.
**Reload** goes back to the unshifted images.

### Reading the shift plot

A smooth curve, or small scatter around one, is a good result. A single
point far from its neighbours usually means a wrong match for that image:
try a registration region, a stronger noise filter, or the chained mode.

## Registration region

Tick **Registration region** under the image: a dashed yellow box appears.
Drag and resize it over a sharp feature that is visible at **all**
energies (particle edges, a crack, a marker), then tick
**Only inside the registration region** in the dialog. This is faster
and avoids areas that change strongly with energy or have no structure.

## Chained mode

**Each image to its neighbour (chained)** compares every image with the
next one towards the reference image and adds up the shifts. Use it when
the image looks very different on the two sides of the edge, so that a
direct comparison with one reference image fails. Small errors add up
along the chain, so the direct mode is better when it works.

## Reusing shifts on other scans

When several scans drift in the same way, you can register one and reuse
its shifts:

1. Register a scan, then **Save shifts…** (toolbar). The CSV file holds,
   for each energy, the shift in **full-resolution pixels**, plus the
   source scan in its header.
2. Open another scan and click **Load shifts…** with that file. The
   shifts are matched to this scan's energies (interpolated when the
   energy points differ) and converted to the **Binning** you loaded
   with, so a file saved at binning 2 works at binning 1 or 4.
3. The usual shift plot appears; **Apply** or **Discard**.

If the new scan has energies outside the saved range, the nearest saved
value is used and the status bar says how many.

## How it works

For each image:

1. empty pixels are filled, the mean is subtracted and a smooth window
   fades the borders (so the image edges are not mistaken for structure);
2. the image and the reference image are compared by **phase
   correlation**: in the Fourier domain only the phase of every spatial
   frequency is kept, which gives a sharp peak at their offset and makes
   it insensitive to the overall change of contrast across the edge;
3. a band-pass weight ignores pixel noise and slow background (beam
   profile, overall absorption change) — this is what keeps low-contrast
   images below the edge working;
4. around the peak the correlation is recomputed on a 20× finer grid
   (upsampled DFT, Guizar-Sicairos et al. 2008), giving the shift to
   1/20 pixel;
5. on **Apply**, each image is shifted with linear interpolation.

Shifts are translations only (no rotation or magnification change).
