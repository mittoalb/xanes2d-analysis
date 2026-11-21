import os
import sys

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from synth import make_file                                   # noqa: E402
from xanes2d_analysis import io, registration, spectra        # noqa: E402


@pytest.fixture(scope="module")
def scan_file(tmp_path_factory):
    p = tmp_path_factory.mktemp("d") / "scan_001.h5"
    truth = make_file(str(p))
    return str(p), truth


def test_load_uses_flat_index(scan_file):
    path, (energies, drifts, disc_a, disc_b) = scan_file
    s = io.load_scan(path)
    assert s.od.shape == (31, 120, 160)
    assert np.allclose(s.energy, energies)
    # background OD ~0.2 + texture: flat-field correction must be right
    # (a wrong flat would leave the beam profile in the image)
    bg = s.od[-1][~(disc_a | disc_b)]
    assert 0.1 < np.nanmedian(bg) < 0.8


def test_binning(scan_file):
    s = io.load_scan(scan_file[0], binning=2)
    assert s.od.shape == (31, 60, 80)


def test_phase_correlation_subpixel():
    rng = np.random.default_rng(0)
    img = ndimage.gaussian_filter(rng.random((128, 128)), 2)
    moved = ndimage.shift(img, (2.3, -1.7), order=3, mode="nearest")
    shift = registration.phase_correlation(img, moved, upsample=50)
    assert np.allclose(shift, (-2.3, 1.7), atol=0.1)


def test_register_recovers_drift(scan_file):
    path, (energies, drifts, *_) = scan_file
    s = io.load_scan(path)
    shifts = registration.register_stack(s.od, ref_index=s.n - 1, upsample=20)
    # applying `shifts` must undo the drift: shifts ~ -drift
    err = np.abs(shifts + drifts)
    assert err.max() < 0.35, err.max()
    registration.apply_shifts(s.od, shifts)


def test_spectra_after_registration(scan_file):
    path, (energies, drifts, disc_a, disc_b) = scan_file
    s = io.load_scan(path)
    registration.apply_shifts(s.od, registration.register_stack(s.od, s.n - 1))
    core_a = ndimage.binary_erosion(disc_a, iterations=6)
    core_b = ndimage.binary_erosion(disc_b, iterations=6)
    spec_a = np.nanmean(s.od[:, core_a], axis=1)
    spec_b = np.nanmean(s.od[:, core_b], axis=1)
    e0a = spectra.edge_energy(s.energy, spec_a)
    e0b = spectra.edge_energy(s.energy, spec_b)
    assert abs(e0a - 8333) <= 3 and abs(e0b - 8345) <= 3, (e0a, e0b)
    norm, e0, step = spectra.normalize(s.energy, spec_a, (8310, 8320), (8355, 8370))
    assert abs(np.nanmean(norm[s.energy > 8360]) - 1) < 0.1
    assert abs(np.nanmean(norm[s.energy < 8318])) < 0.05


def test_reference_loader_text_npy_kev(tmp_path):
    e_kev = np.linspace(8.3, 8.4, 11)
    mu = np.linspace(0, 1, 11)
    real = tmp_path / "a.npy"
    np.save(real, np.column_stack([e_kev, mu]))
    text = tmp_path / "b_calibrated.npy"            # text file with .npy name
    text.write_text("Energy Intensity\n" + "\n".join(f"{a} {b}" for a, b in zip(e_kev, mu)))
    csv = tmp_path / "c.csv"
    csv.write_text("# ref\nE,mu\n" + "\n".join(f"{a*1000},{b}" for a, b in zip(e_kev, mu)))
    for p in (real, text, csv):
        e, m = io.load_reference(str(p))
        assert np.allclose(e, e_kev * 1000) and np.allclose(m, mu)


def test_r_factor_zero_for_identical():
    e = np.linspace(0, 10, 50)
    assert spectra.r_factor(e, np.sin(e) + 2, e, np.sin(e) + 2) == pytest.approx(0)


def test_csv_export(tmp_path):
    p = tmp_path / "s.csv"
    spectra.write_csv(str(p), np.array([1.0, 2.0]), {"roi 1": [0.1, 0.2]}, ["src x.h5"])
    txt = p.read_text().splitlines()
    assert txt[0] == "# src x.h5" and txt[1] == "energy_eV,roi 1" and txt[2] == "1,0.1"


def test_register_neighbour_mode(scan_file):
    path, (energies, drifts, *_) = scan_file
    s = io.load_scan(path)
    shifts = registration.register_stack(s.od, ref_index=s.n // 2, mode="neighbour")
    # relative to the middle frame: shifts[i] ~ drift[ref] - drift[i]
    expected = drifts[s.n // 2] - drifts
    assert np.abs(shifts - expected).max() < 0.4


def test_each_image_divided_by_its_own_reference(scan_file):
    import h5py
    path = scan_file[0]
    s = io.load_scan(path)
    with h5py.File(path, "r") as f:
        flats = f["exchange/data_flat"][:].astype(np.float32)
        fidx = f["exchange/flat_index"][:]
    assert np.array_equal(s.flat_index, fidx)
    for i in (0, 1, 2, 3, 15, s.n - 1):
        expected = -np.log(s.raw[i] / flats[fidx[i]])
        assert np.allclose(s.od[i], expected, equal_nan=True, atol=1e-5)


def _xdi(tmp_path, name, cols, rows, extra=""):
    head = "# XDI/1.0\n" + "".join(f"# Column.{i+1}: {c}\n" for i, c in enumerate(cols))
    head += "# Element.symbol: Ni\n# Element.edge: K\n# Sample.name: test\n" + extra + "# ///\n#----\n"
    p = tmp_path / name
    p.write_text(head + "\n".join(" ".join(f"{v:.8g}" for v in r) for r in rows) + "\n")
    return str(p)


def test_xdi_reader_variants(tmp_path):
    from xanes2d_analysis import refs
    e = np.linspace(8300, 8400, 51)
    mu = 0.5 + 0.5 * np.tanh((e - 8333) / 3)
    i0 = np.full_like(e, 1e5)
    it = i0 * np.exp(-mu)
    # transmission from i0 / itrans
    d = refs.read_xdi(_xdi(tmp_path, "t.xdi", ["energy eV", "i0", "itrans"], zip(e, i0, it)))
    assert d["mu_kind"] == "ln(i0/itrans)" and np.allclose(d["mu"], mu, atol=1e-6)
    assert d["meta"]["Element.symbol"] == "Ni"
    # mu column wins, keV energies converted
    d = refs.read_xdi(_xdi(tmp_path, "m.xdi", ["energy keV", "mutrans"], zip(e / 1000, mu)))
    assert d["mu_kind"] == "mutrans" and np.allclose(d["energy"], e)
    # fluorescence
    d = refs.read_xdi(_xdi(tmp_path, "f.xdi", ["energy eV", "i0", "ifluor"], zip(e, i0, mu * 1e3)))
    assert d["mu_kind"] == "ifluor/i0" and np.allclose(d["mu"], mu * 1e-2)
    # angle column with d-spacing → energy
    dsp = 3.1355
    ang = np.degrees(np.arcsin(refs.HC_EV_A / (2 * dsp * e)))
    d = refs.read_xdi(_xdi(tmp_path, "a.xdi", ["angle deg", "mutrans"], zip(ang, mu),
                           extra=f"# Mono.d_spacing: {dsp}\n"))
    assert np.allclose(d["energy"], e, atol=1e-3)
    # io.load_reference recognises XDI
    e2, mu2 = io.load_reference(str(tmp_path / "t.xdi"))
    assert np.allclose(mu2, mu, atol=1e-6)


def test_edge_maps_separate_the_two_phases(scan_file):
    path, (energies, drifts, disc_a, disc_b) = scan_file
    s = io.load_scan(path)
    registration.apply_shifts(s.od, registration.register_stack(s.od, s.n - 1))
    step, e_half = spectra.edge_maps(s.energy, s.od, (8310, 8322), (8355, 8370))
    a = ndimage.binary_erosion(disc_a, iterations=6)
    b = ndimage.binary_erosion(disc_b, iterations=6)
    bg = ~ndimage.binary_dilation(disc_a | disc_b, iterations=4)
    assert abs(np.nanmedian(e_half[a]) - 8333) < 3
    assert abs(np.nanmedian(e_half[b]) - 8345) < 3
    assert np.nanmedian(step[a]) > 0.6 and abs(np.nanmedian(step[bg])) < 0.1


def test_shifts_roundtrip_binning_and_interpolation(tmp_path):
    e = np.array([8300.0, 8310.0, 8320.0, 8330.0])
    sh_bin2 = np.array([[0.5, -1.0], [1.0, 0.0], [0.0, 0.25], [0.0, 0.0]])
    p = tmp_path / "shifts.csv"
    io.save_shifts(str(p), e, sh_bin2, binning=2, header=["source: x.h5"])
    e2, full = io.load_shifts(str(p))
    assert np.allclose(e2, e) and np.allclose(full, sh_bin2 * 2)      # stored full-res
    # same energies, binning 4 → half of the binning-2 values
    s4, out = io.shifts_for_scan(e, 4, e2, full)
    assert out == 0 and np.allclose(s4, sh_bin2 / 2)
    # other energies: interpolated; outside the saved range → nearest value
    s1, out = io.shifts_for_scan(np.array([8305.0, 8340.0]), 1, e2, full)
    assert out == 1
    assert np.allclose(s1[0], [1.5, -1.0]) and np.allclose(s1[1], full[-1])


def test_segmentation_and_background_removal(tmp_path):
    from xanes2d_analysis import background as bgm
    p = tmp_path / "flux_001.h5"
    energies, drifts, disc_a, disc_b = make_file(str(p), flux_jitter=0.05, drift_px=0)
    s = io.load_scan(str(p))
    img = bgm.contrast_image(s.energy, s.od, (8355, 8370))
    mask = bgm.segment_sample(img, bgm.otsu_threshold(img))
    truth = disc_a | disc_b
    iou = (mask & truth).sum() / (mask | truth).sum()
    assert iou > 0.9, iou
    bg = bgm.background_pixels(mask)
    levels = bgm.background_levels(s.od, bg)
    # the per-energy flux jitter is an OD offset of -ln(flux): background
    # levels must follow it, and after subtraction the background is ~0
    corrected = s.od - levels[:, None, None]
    assert abs(np.nanmedian(corrected[:, bg])) < 0.01
    core = ndimage.binary_erosion(disc_a, iterations=6)
    raw_spec = np.nanmean(s.od[:, core], axis=1)
    cor_spec = np.nanmean(corrected[:, core], axis=1)
    # pre-edge part of phase A is flat in truth: jitter makes it wavy,
    # background removal flattens it
    pre = s.energy < 8322
    assert np.std(cor_spec[pre]) < 0.3 * np.std(raw_spec[pre])
