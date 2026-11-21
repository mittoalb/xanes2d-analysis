# Edge-energy and edge-step maps

The maps colour every pixel by a property of its own spectrum. Make them
after registration: they are computed from the aligned images, and are
recomputed automatically when you register, switch the flat-field
correction or move the pre/post-edge ranges.

```{figure} images/edge_energy_map.png
:alt: Edge energy map with a scale bar
:width: 100%

Edge energy map: the two particles have edges about 12 eV apart and show
in different colours (scale in eV on the right). Synthetic data.
```

## Showing a map

In the drop-down below the image choose:

- **Edge energy map**: each pixel coloured by the energy where its
  spectrum reaches half of its edge step. Shifts of this energy usually
  follow oxidation state or phase (for most transition metals, a higher
  edge energy means more oxidised).
- **Edge step map**: the size of the edge jump, which shows where the
  element is and how much of it.

```{figure} images/edge_step_map.png
:alt: Edge step map
:width: 60%

Edge step map of the same sample.
```

While a map is shown, the energy slider is inactive and the pre/post-edge
ranges are shown on the plot (drag them to change the maps). Clicking
still gives spectra. Choose **Absorption** again to return to the images.

| Control | Meaning |
|---|---|
| Colour drop-down (**turbo**, viridis, …) | Colour scale. |
| **Mask step < … % of max** | Edge energy map only: pixels whose edge step is smaller than this fraction of the largest step are left blank — there is too little of the element for a meaningful edge. |
| Histogram (right) | Drag to set the colour range. |
| Hover | The status bar shows the pixel's edge energy or edge step. |

## How it is computed

For every pixel:

1. a straight line is fitted to its spectrum in the **blue** (pre-edge)
   range and another in the **red** (post-edge) range;
2. **edge step** = distance between the two lines in the middle of the
   gap between the ranges (this is the edge step map);
3. the spectrum is normalised (pre-edge line subtracted, divided by the
   edge step) and lightly smoothed along energy (3-point average);
4. **edge energy** = the energy where it first crosses 0.5 between the
   end of the blue range and the start of the red one, interpolated
   between the two energy points around the crossing (so it is not
   limited to the energy step).

Pixels without a crossing, or masked by the edge-step threshold, are
blank.

## Things to keep in mind

- Place the ranges well away from the edge; they matter more than
  anything else.
- A strong white line or noisy pixels can move the half-step energy
  slightly; compare regions, not single pixels, when differences are
  small (binning helps).
- A scan that starts above the edge has no pre-edge range, so its maps
  are not meaningful.
