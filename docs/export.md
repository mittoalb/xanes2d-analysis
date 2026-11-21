# Saving results

## Export spectra…

Saves the ticked spectra to a CSV file (default name
`<scan>_spectra.csv`), one column per spectrum:

- `energy_eV`;
- each spectrum (absorption);
- with **Normalize** ticked, also each normalised spectrum;
- with a reference loaded, the reference at your scan's energies (shifted,
  and normalised when **Normalize** is ticked);
- with **Subtract background** ticked, `background`: the level subtracted
  at each energy.

Lines starting with `#` at the top record how the spectra were made: the
source file, binning and dark level, whether the flat-field correction
was on, the registration shifts for every energy, the normalization
ranges, how the sample was found and whether the background was
subtracted, the position and size of every region or point, and the
reference and its energy shift. The file opens in any spreadsheet,
Athena/Larch or Python.

## Save shifts…

Saves the registration shifts for reuse on other scans; see
[Reusing shifts on other scans](registration.md#reusing-shifts-on-other-scans). Columns:
`energy_eV`, `dy_px`, `dx_px`, in full-resolution pixels.

## Save aligned stack…

Saves the registered images to HDF5 (default `<scan>_aligned.h5`):

| Dataset | Content |
|---|---|
| `/exchange/data_od` | Flat-field corrected absorption −ln(I/I0), aligned |
| `/exchange/data_raw` | Raw images (minus dark), with the same shifts |
| `/exchange/energy` | Energies, eV |
| `/registration/shifts` | Shift of every image (rows, columns), in the binned pixels the stack was loaded with |

The file also records the source file, binning and dark level. Use it to
continue the analysis in other programs.
