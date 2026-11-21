# Troubleshooting

**"… is not a XANES 2D file"**
: The file has no sample and reference images in the XANES 2D layout.
  Open the file written by XANES 2D (or DataMap's 2D XANES action), not
  a TomoScan file or an aligned stack.

**Loading is slow or the computer runs out of memory**
: Use **Binning** 2 or 4. A scan of about 100 energies at full resolution
  needs about 3 GB.

**The image looks wrong or has strange stripes / negative absorption**
: Untick **Flat-field correction** to look at the raw images and check
  that the references were taken with the beam passing beside the sample.

**Registration gives a jump at a few energies**
: Place the **Registration region** on a sharp feature visible at all
  energies, raise the **Noise filter**, or use the chained mode. Check
  the shift plot before clicking **Apply**.

**Loaded shifts do not fit**
: Shifts can only be reused for scans that drift the same way (same
  energy range and set-up). Discard them and register this scan.

**The spectrum of a region does not change when I move it**
: It updates when you release the mouse. If the region is outside the
  image, its spectrum is empty.

**The edge energy map is mostly blank**
: Lower **Mask step <**, check that the blue and red ranges lie below and
  above the edge, and that the scan really covers the edge.

**Find sample includes background, or misses part of the sample**
: With **Threshold**, raise or lower the **Threshold**. Check that the red
  post-edge range lies above the edge. For difficult samples use **SAM 3**
  (see {doc}`sam3`).

**SAM 3 does not work**
: See the table at the end of {doc}`sam3`.

**The reference library is empty**
: Click **Load reference… → Download / update library** (needs internet),
  or run `xanes2d-refs download`. On a computer without internet, copy
  the library folder from another one and point `XANES2D_REFS` to it.

**The map or drop-down options are missing**
: Restart the program after updating it.
