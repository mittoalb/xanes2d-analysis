# Background removal

The **Background** row under the image finds the sample, then optionally
subtracts the background level at every energy and hides everything that
is not sample.

```{figure} images/background.png
:alt: Sample found and outlined, background subtracted and hidden
:width: 100%

**Find sample** with **Threshold**: the sample outline in cyan, the
background subtracted and hidden. Synthetic data.
```

## Finding the sample

1. Register the images first (the sample must not move between energies).
2. Make sure the red **post-edge** range on the plot lies above the edge:
   the sample is found in the **mean of the post-edge images**, where it
   absorbs most and has the highest contrast. (The range is used even when
   **Normalize** is not ticked.)
3. Choose how to segment it, then click **Find sample**:

   | Method | How it works |
   |---|---|
   | **Threshold** (default) | Pixels above a threshold (lightly smoothed) are sample. The threshold is set automatically (Otsu's method) and shown in **Threshold**; change it and the mask follows at once. Small specks are removed and holes filled. |
   | **SAM 3** | Meta's Segment Anything Model 3 finds the objects described by the text prompt next to it (default `particle`; try `cell`, `grain`, `crystal`, …). All objects scoring at least the set score are combined into the sample. Needs a SAM 3 set-up: see {doc}`sam3`. |

4. Check the cyan **Outline** on the image. The grey text next to the row
   says how the sample was found, how much of the image it covers and how
   many background pixels there are.

## What you can do with it

| Checkbox | Effect |
|---|---|
| **Subtract background** | At every energy, the median absorption of the background is subtracted from the image, the spectra and the maps, so empty areas become 0. Background pixels are taken a few pixels away from the sample edge. |
| **Hide background** | Everything outside the sample is left blank in the image and in the maps. |
| **Outline** | Shows / hides the sample outline. |

**Why subtract the background?** The beam intensity changes a little
between each image and its reference. That shows up as an offset that is
the same for every pixel of that energy, and it makes spectra wavy.
Subtracting the background level, measured in empty areas of the same
image, removes it. With **Flat-field correction** off, the empty areas act
as the reference (I0).

## When the mask is updated

- After **registration** (the sample moved): with **Threshold** the
  sample is found again with the same threshold; with **SAM 3** the mask
  is cleared and the status bar asks you to press **Find sample** again.
- After switching **Flat-field correction**: with **Threshold** a new
  threshold is set; a SAM 3 mask is kept (only the background levels are
  recomputed).
- **Reload** or opening another file clears it.

## In exported files

With **Subtract background** ticked, **Export spectra…** adds a
`background` column (the level subtracted at each energy) and a header
line saying how the sample was found (threshold value or SAM 3 prompt).
