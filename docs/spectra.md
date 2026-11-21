# Spectra from points and regions

## A point

Click the image. The spectrum of that spot appears on the plot as
**Point**, averaged over a small square around it (**Point average**,
default 3 px = 3×3 pixels). Clicking elsewhere moves the point.

To keep it and add another, click **Keep point**: it becomes
**Point 1**, **Point 2**, … and the next click starts a new one.

## Regions

**+ Rectangle**, **+ Ellipse** and **+ Polygon** add a region in the middle
of the image, each with its own colour. Drag it to move it and drag its
handles to change its shape; its spectrum (the mean over all pixels
inside) updates when you release the mouse. For a polygon, drag the
corner points.

## The Spectra list

| What you do | Effect |
|---|---|
| Untick an entry | Hides its curve and its marker or region on the image. |
| Double-click its name | Rename it (used in the legend and in exported files). |
| Select it, **Remove selected** | Deletes the spectrum and its region / point. |
| Select it | The R-factor against the reference and E0 refer to this one. |

## Normalization

Tick **Normalize (edge step)**. Two shaded ranges appear on the plot:

- **blue** = pre-edge range: a straight line fitted here is subtracted;
- **red** = post-edge range: the spectrum is divided by the edge step,
  the distance between the post-edge and pre-edge lines at the edge.

Drag the ranges and their edges so that the blue one lies below the edge
and the red one above it, away from the white line. All spectra, the
reference and the maps use the same ranges. Next to the checkbox the
edge position **E0** of the selected spectrum is shown (energy of the
steepest rise).

:::{tip}
If a scan starts at or above the edge, there is no real pre-edge range;
normalized spectra and maps from such a scan are not meaningful.
:::

## The energy line

The dashed vertical line on the plot marks the energy shown in the image.
