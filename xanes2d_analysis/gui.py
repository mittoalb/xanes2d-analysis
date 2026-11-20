"""XANES 2D analysis window: open a XANES 2D scan, register the images,
pick points / regions and look at, normalise, compare and export spectra."""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui, QtWidgets

from . import io, registration, spectra

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True)

COLORS = ["#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
          "#17becf", "#bcbd22", "#1f77b4", "#7f7f7f"]
XANES_GUI_CURVES = [os.path.expanduser(p) for p in (
    "~/Software/xanes_gui/Calibrated", "/home/beams/USERTXM/Software/xanes_gui/Calibrated",
    "/home/beams0/AMITTONE/Software/xanes_gui/Calibrated")]


# ── background work ───────────────────────────────────────────────────────

class Job(QtCore.QThread):
    """Run fn(progress, stop) in a thread; emits progress and the result."""
    progress = QtCore.pyqtSignal(int, int)
    result = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            self.result.emit(self.fn(self.progress.emit, lambda: self._stop))
        except InterruptedError:
            self.failed.emit("Cancelled.")
        except Exception as e:                        # shown to the user
            self.failed.emit(f"{type(e).__name__}: {e}")


# ── one spectrum on the plot ──────────────────────────────────────────────

class Spectrum:
    def __init__(self, name, color, roi=None, point=None, marker=None):
        self.name, self.color = name, color
        self.roi = roi                 # pg ROI (regions) or None
        self.point = point             # (row, col) for points
        self.marker = marker           # scatter item on the image for points
        self.values: Optional[np.ndarray] = None
        self.curve: Optional[pg.PlotDataItem] = None
        self.item: Optional[QtWidgets.QListWidgetItem] = None

    def describe(self) -> str:
        if self.point is not None:
            return f"{self.name}: point row {self.point[0]}, col {self.point[1]}"
        r = self.roi
        p, s = r.pos(), r.size()
        kind = type(r).__name__.replace("ROI", "").lower() or "region"
        if isinstance(r, pg.PolyLineROI):
            pts = [(round(h.pos().x() + p.x(), 1), round(h.pos().y() + p.y(), 1))
                   for h in r.getHandles()]
            return f"{self.name}: polygon (col,row) {pts}"
        return (f"{self.name}: {kind} col {p.x():.1f}, row {p.y():.1f}, "
                f"width {s.x():.1f}, height {s.y():.1f}")


# ── registration settings + result ────────────────────────────────────────

class RegisterDialog(QtWidgets.QDialog):
    def __init__(self, parent, scan: io.Scan, current: int, has_region: bool):
        super().__init__(parent)
        self.setWindowTitle("Register images")
        f = QtWidgets.QFormLayout(self)
        self.ref = QtWidgets.QSpinBox()
        self.ref.setRange(1, scan.n)
        self.ref.setValue(current + 1)
        self.ref_lbl = QtWidgets.QLabel()
        self.ref.valueChanged.connect(
            lambda v: self.ref_lbl.setText(f"{scan.energy[v - 1]:.2f} eV"))
        self.ref.valueChanged.emit(self.ref.value())
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self.ref)
        row.addWidget(self.ref_lbl)
        f.addRow("Reference image:", row)
        self.mode = QtWidgets.QComboBox()
        self.mode.addItems(["Each image to the reference",
                            "Each image to its neighbour (chained)"])
        self.mode.setToolTip("Chained: for scans whose images change a lot across "
                             "the edge — neighbouring energies look alike. Small "
                             "errors add up along the chain.")
        f.addRow("Compare:", self.mode)
        self.smooth = QtWidgets.QDoubleSpinBox()
        self.smooth.setRange(0, 10)
        self.smooth.setSingleStep(0.5)
        self.smooth.setValue(1.5)
        self.smooth.setSuffix(" px")
        self.smooth.setToolTip("Noise filter: higher for noisy images, 0 = none.")
        f.addRow("Noise filter:", self.smooth)
        self.upsample = QtWidgets.QSpinBox()
        self.upsample.setRange(1, 200)
        self.upsample.setValue(20)
        self.upsample.setToolTip("Sub-pixel precision = 1 / this value.")
        f.addRow("Sub-pixel factor:", self.upsample)
        self.use_region = QtWidgets.QCheckBox("Only inside the registration region")
        self.use_region.setChecked(has_region)
        self.use_region.setEnabled(has_region)
        if not has_region:
            self.use_region.setToolTip("Tick 'Registration region' in the main window "
                                       "and place it on a feature first.")
        f.addRow("", self.use_region)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.button(QtWidgets.QDialogButtonBox.Ok).setText("Compute shifts")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)


class ShiftsDialog(QtWidgets.QDialog):
    """Show the computed shifts; Apply or Discard."""
    def __init__(self, parent, energy, shifts):
        super().__init__(parent)
        self.setWindowTitle("Registration shifts")
        self.resize(560, 380)
        v = QtWidgets.QVBoxLayout(self)
        pw = pg.PlotWidget()
        pw.addLegend()
        pw.setLabel("bottom", "Energy", "eV")
        pw.getAxis("bottom").enableAutoSIPrefix(False)
        pw.setLabel("left", "Shift", "px")
        pw.plot(energy, shifts[:, 1], pen="#1f77b4", symbol="o", symbolSize=5,
                symbolBrush="#1f77b4", name="x (columns)")
        pw.plot(energy, shifts[:, 0], pen="#ff7f0e", symbol="o", symbolSize=5,
                symbolBrush="#ff7f0e", name="y (rows)")
        v.addWidget(pw)
        mx = np.abs(shifts).max() if len(shifts) else 0
        v.addWidget(QtWidgets.QLabel(
            f"Largest shift {mx:.2f} px. Jumps far from the rest usually mean a "
            "wrong match: try a registration region or the chained mode."))
        bb = QtWidgets.QDialogButtonBox()
        bb.addButton("Apply", QtWidgets.QDialogButtonBox.AcceptRole)
        bb.addButton("Discard", QtWidgets.QDialogButtonBox.RejectRole)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)


# ── main window ───────────────────────────────────────────────────────────

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("XANES 2D Analysis")
        self.resize(1500, 900)
        self.scan: Optional[io.Scan] = None
        self.spectra: List[Spectrum] = []
        self.point_spec: Optional[Spectrum] = None
        self.ref: Optional[tuple] = None            # (energy, mu, name)
        self.ref_curve = None
        self.job: Optional[Job] = None
        self._levels_set = False
        self._n_roi = 0
        self._build()
        self._set_enabled(False)

    # ── layout ──────────────────────────────────────────────────────────
    def _build(self):
        tb = self.addToolBar("Main")
        tb.setMovable(False)
        self.act_open = tb.addAction("Open…", self.open_dialog)
        tb.addWidget(QtWidgets.QLabel("  Binning "))
        self.binning = QtWidgets.QComboBox()
        self.binning.addItems(["1", "2", "4"])
        self.binning.setCurrentIndex(1)
        self.binning.setToolTip("Average N×N pixels when loading: faster, less memory, "
                                "less noise per pixel.")
        tb.addWidget(self.binning)
        tb.addWidget(QtWidgets.QLabel("  Dark "))
        self.dark = QtWidgets.QDoubleSpinBox()
        self.dark.setRange(0, 65535)
        self.dark.setDecimals(1)
        self.dark.setToolTip("Camera dark level (counts) subtracted from images and "
                             "references. XANES 2D files contain no dark images.")
        tb.addWidget(self.dark)
        self.act_reload = tb.addAction("Reload", self.reload)
        tb.addSeparator()
        self.flat_chk = QtWidgets.QCheckBox("Flat-field correction")
        self.flat_chk.setChecked(True)
        self.flat_chk.setToolTip(
            "On: every image is divided by its own reference image (done at load, "
            "before any registration), shown as −ln(I/I0).\n"
            "Off: the raw images (minus dark), shown as −ln(I) or counts.\n"
            "Registration shifts are applied to both, so you can switch at any time.")
        self.flat_chk.toggled.connect(self._toggle_flat)
        tb.addWidget(self.flat_chk)
        tb.addSeparator()
        self.act_register = tb.addAction("Register…", self.register_dialog)
        self.act_save = tb.addAction("Save aligned stack…", self.save_aligned)
        tb.addSeparator()
        self.act_ref = tb.addAction("Load reference…", self.load_reference_dialog)
        self.act_export = tb.addAction("Export spectra…", self.export_dialog)

        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.setCentralWidget(split)

        # image side
        left = QtWidgets.QWidget()
        lv = QtWidgets.QVBoxLayout(left)
        self.glw = pg.GraphicsLayoutWidget()
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked(True)
        self.vb.invertY(True)
        self.img = pg.ImageItem()
        self.vb.addItem(self.img)
        self.hist = pg.HistogramLUTItem()
        self.hist.setImageItem(self.img)
        self.glw.addItem(self.hist)
        self.img.scene().sigMouseClicked.connect(self._on_click)
        self.img.scene().sigMouseMoved.connect(self._on_hover)
        lv.addWidget(self.glw, 1)

        srow = QtWidgets.QHBoxLayout()
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._show_frame)
        self.e_lbl = QtWidgets.QLabel("—")
        self.e_lbl.setMinimumWidth(190)
        srow.addWidget(QtWidgets.QLabel("Energy"))
        srow.addWidget(self.slider, 1)
        srow.addWidget(self.e_lbl)
        lv.addLayout(srow)

        crow = QtWidgets.QHBoxLayout()
        self.display = QtWidgets.QComboBox()
        self.display.addItems(["Absorption (−ln I/I0)", "Transmission (I/I0)"])
        self.display.currentIndexChanged.connect(lambda _: self._show_frame(relevel=True))
        crow.addWidget(self.display)
        crow.addWidget(QtWidgets.QLabel("  Point average"))
        self.pt_size = QtWidgets.QSpinBox()
        self.pt_size.setRange(1, 51)
        self.pt_size.setSingleStep(2)
        self.pt_size.setValue(3)
        self.pt_size.setSuffix(" px")
        self.pt_size.setToolTip("A clicked point averages an N×N square around it.")
        self.pt_size.valueChanged.connect(lambda _: self._update_spectrum(self.point_spec))
        crow.addWidget(self.pt_size)
        crow.addStretch()
        self.reg_region_chk = QtWidgets.QCheckBox("Registration region")
        self.reg_region_chk.setToolTip("A dashed box: registration compares only what "
                                       "is inside it. Put it on a sharp feature that is "
                                       "visible at all energies.")
        self.reg_region_chk.toggled.connect(self._toggle_reg_region)
        crow.addWidget(self.reg_region_chk)
        lv.addLayout(crow)

        rrow = QtWidgets.QHBoxLayout()
        for text, kind in (("+ Rectangle", "rect"), ("+ Ellipse", "ellipse"),
                           ("+ Polygon", "polygon")):
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(lambda _, k=kind: self.add_roi(k))
            rrow.addWidget(b)
            setattr(self, f"btn_{kind}", b)
        self.btn_keep = QtWidgets.QPushButton("Keep point")
        self.btn_keep.setToolTip("Turn the clicked point into a permanent entry, so the "
                                 "next click adds a new one.")
        self.btn_keep.clicked.connect(self.keep_point)
        rrow.addWidget(self.btn_keep)
        lv.addLayout(rrow)
        hint = QtWidgets.QLabel("Click the image for the spectrum of a point. Add regions "
                                "with the buttons; drag them, their spectrum follows.")
        hint.setStyleSheet("color: #888;")
        lv.addWidget(hint)
        split.addWidget(left)

        # spectrum side
        right = QtWidgets.QWidget()
        rv = QtWidgets.QVBoxLayout(right)
        self.plot = pg.PlotWidget()
        self.plot.addLegend(offset=(10, 10))
        self.plot.setLabel("bottom", "Energy", "eV")
        self.plot.getAxis("bottom").enableAutoSIPrefix(False)    # stay in eV
        self.plot.setLabel("left", "Absorption −ln(I/I0)")
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.e_line = pg.InfiniteLine(angle=90, pen=pg.mkPen("#888", style=QtCore.Qt.DashLine))
        self.plot.addItem(self.e_line)
        self.pre_region = pg.LinearRegionItem(brush=(80, 80, 255, 40))
        self.post_region = pg.LinearRegionItem(brush=(255, 80, 80, 40))
        for r in (self.pre_region, self.post_region):
            r.sigRegionChangeFinished.connect(self._replot_all)
        rv.addWidget(self.plot, 3)

        nrow = QtWidgets.QHBoxLayout()
        self.norm_chk = QtWidgets.QCheckBox("Normalize (edge step)")
        self.norm_chk.setToolTip("Subtract a line fitted in the blue (pre-edge) range and "
                                 "divide by the edge step, using a line fitted in the red "
                                 "(post-edge) range. Drag the ranges on the plot.")
        self.norm_chk.toggled.connect(self._toggle_norm)
        nrow.addWidget(self.norm_chk)
        self.e0_lbl = QtWidgets.QLabel("")
        nrow.addWidget(self.e0_lbl)
        nrow.addStretch()
        rv.addLayout(nrow)

        refbox = QtWidgets.QGroupBox("Reference spectrum")
        rf = QtWidgets.QHBoxLayout(refbox)
        self.ref_lbl = QtWidgets.QLabel("none — Load reference…")
        rf.addWidget(self.ref_lbl, 1)
        rf.addWidget(QtWidgets.QLabel("Energy shift"))
        self.ref_shift = QtWidgets.QDoubleSpinBox()
        self.ref_shift.setRange(-500, 500)
        self.ref_shift.setDecimals(2)
        self.ref_shift.setSuffix(" eV")
        self.ref_shift.valueChanged.connect(lambda _: self._replot_all())
        rf.addWidget(self.ref_shift)
        self.ref_clear = QtWidgets.QPushButton("Remove")
        self.ref_clear.clicked.connect(self.clear_reference)
        rf.addWidget(self.ref_clear)
        rv.addWidget(refbox)

        listbox = QtWidgets.QGroupBox("Spectra")
        lb = QtWidgets.QVBoxLayout(listbox)
        self.list = QtWidgets.QListWidget()
        self.list.itemChanged.connect(self._on_item_changed)
        self.list.currentItemChanged.connect(lambda *_: self._update_rfactor())
        lb.addWidget(self.list)
        brow = QtWidgets.QHBoxLayout()
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self.remove_selected)
        brow.addWidget(self.btn_remove)
        brow.addStretch()
        self.rf_lbl = QtWidgets.QLabel("")
        brow.addWidget(self.rf_lbl)
        lb.addLayout(brow)
        rv.addWidget(listbox, 2)
        split.addWidget(right)
        split.setSizes([800, 700])

        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(260)
        self.progress.hide()
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.hide()
        self.cancel_btn.clicked.connect(lambda: self.job and self.job.stop())
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.cancel_btn)
        self.reg_roi = None

    def _set_enabled(self, on: bool):
        for w in (self.act_reload, self.act_register, self.act_save, self.act_export,
                  self.btn_rect, self.btn_ellipse, self.btn_polygon, self.btn_keep,
                  self.btn_remove, self.reg_region_chk, self.norm_chk, self.slider):
            w.setEnabled(on)

    # ── jobs ────────────────────────────────────────────────────────────
    def _run(self, label, fn, done):
        if self.job is not None and self.job.isRunning():
            return
        self.statusBar().showMessage(label)
        self.progress.setValue(0)
        self.progress.show()
        self.cancel_btn.show()
        self.job = Job(fn, self)
        self.job.progress.connect(lambda i, n: (self.progress.setMaximum(n),
                                                self.progress.setValue(i)))
        self.job.result.connect(done)
        self.job.failed.connect(lambda m: QtWidgets.QMessageBox.warning(self, "XANES 2D Analysis", m))
        self.job.finished.connect(lambda: (self.progress.hide(), self.cancel_btn.hide()))
        self.job.start()

    # ── loading ─────────────────────────────────────────────────────────
    def open_dialog(self):
        start = os.path.dirname(self.scan.path) if self.scan else os.getcwd()
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open XANES 2D file", start, "HDF5 (*.h5 *.hdf5 *.hdf);;All files (*)")
        if path:
            self.open_file(path)

    def open_file(self, path):
        b, dark = int(self.binning.currentText()), self.dark.value()
        t0 = time.time()

        def done(scan):
            self._set_scan(scan)
            self.statusBar().showMessage(
                f"{os.path.basename(path)}: {scan.n} energies "
                f"{scan.energy[0]:.1f}–{scan.energy[-1]:.1f} eV, images "
                f"{scan.od.shape[2]}×{scan.od.shape[1]} (binning {b}), "
                f"{scan.info.get('n_flats', '?')} references — {time.time() - t0:.1f} s")
        self._run(f"Loading {os.path.basename(path)}…",
                  lambda p, s: io.load_scan(path, b, dark, p, s), done)

    def reload(self):
        if self.scan:
            self.open_file(self.scan.path)

    def _set_scan(self, scan: io.Scan):
        self.scan = scan
        self.setWindowTitle(f"XANES 2D Analysis — {os.path.basename(scan.path)}")
        for sp in list(self.spectra):
            self._remove_spectrum(sp)
        self.point_spec = None
        self._n_roi = 0
        self._levels_set = False
        self.slider.blockSignals(True)
        self.slider.setRange(0, scan.n - 1)
        self.slider.setValue(scan.n - 1)
        self.slider.blockSignals(False)
        e = scan.energy
        span = e[-1] - e[0] if scan.n > 1 else 1.0
        self.pre_region.setRegion((e[0], e[0] + 0.2 * span))
        self.post_region.setRegion((e[-1] - 0.3 * span, e[-1]))
        self._set_enabled(True)
        self._show_frame(relevel=True)
        self.vb.autoRange()
        self._replot_all()

    # ── image ───────────────────────────────────────────────────────────
    def _flat_on(self) -> bool:
        return self.flat_chk.isChecked()

    def _frame(self, i):
        """Image i as displayed: flat-field corrected or raw, as absorption
        or transmission / intensity."""
        trans = self.display.currentIndex() == 1
        if self._flat_on():
            f = self.scan.od[i]
            return np.exp(-f) if trans else f
        r = self.scan.raw[i]
        if trans:
            return r
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log(r)

    def _toggle_flat(self, on):
        self.display.setItemText(0, "Absorption (−ln I/I0)" if on else "Absorption (−ln I), raw")
        self.display.setItemText(1, "Transmission (I/I0)" if on else "Intensity (counts), raw")
        if self.scan is None:
            return
        self._show_frame(relevel=True)
        for sp in self.spectra:
            self._update_spectrum(sp)
        self._set_plot_label()

    def _set_plot_label(self):
        base = "Absorption −ln(I/I0)" if self._flat_on() else "−ln(I), no flat-field correction"
        self.plot.setLabel("left", ("Normalized " + base[0].lower() + base[1:])
                           if self.norm_chk.isChecked() else base)

    def _show_frame(self, i=None, relevel=False):
        if self.scan is None:
            return
        i = self.slider.value() if i is None or isinstance(i, bool) else i
        frame = self._frame(i)
        if relevel or not self._levels_set:
            lo, hi = np.nanpercentile(frame, (1, 99))
            self.img.setImage(frame, levels=(lo, hi), autoLevels=False)
            self._levels_set = True
        else:
            self.img.setImage(frame, autoLevels=False)
        self.e_lbl.setText(f"{self.scan.energy[i]:.2f} eV  ({i + 1}/{self.scan.n})")
        self.e_line.setValue(self.scan.energy[i])

    def _on_hover(self, pos):
        if self.scan is None:
            return
        p = self.img.mapFromScene(pos)
        r, c = int(p.y()), int(p.x())
        h, w = self.scan.od.shape[1:]
        if 0 <= r < h and 0 <= c < w:
            v = self._frame(self.slider.value())[r, c]
            self.statusBar().showMessage(f"row {r}, col {c}: {v:.4f}")

    def _on_click(self, ev):
        if self.scan is None or ev.button() != QtCore.Qt.LeftButton or ev.double():
            return
        p = self.img.mapFromScene(ev.scenePos())
        r, c = int(p.y()), int(p.x())
        h, w = self.scan.od.shape[1:]
        if not (0 <= r < h and 0 <= c < w):
            return
        # clicks on a region / its handles belong to the region
        for it in self.img.scene().items(ev.scenePos()):
            if isinstance(it, (pg.ROI, pg.graphicsItems.ROI.Handle)):
                return
        if self.point_spec is None:
            marker = pg.ScatterPlotItem(size=12, pen=pg.mkPen("w", width=1.5),
                                        brush=None, symbol="+")
            self.vb.addItem(marker)
            self.point_spec = self._add_spectrum(Spectrum("Point", self._next_color(),
                                                          point=(r, c), marker=marker))
        sp = self.point_spec
        sp.point = (r, c)
        sp.marker.setData([c + 0.5], [r + 0.5])
        sp.marker.setPen(pg.mkPen(sp.color, width=2))
        self._update_spectrum(sp)

    def _toggle_reg_region(self, on):
        if on and self.scan is not None:
            h, w = self.scan.od.shape[1:]
            self.reg_roi = pg.RectROI([w * 0.25, h * 0.25], [w * 0.5, h * 0.5],
                                      pen=pg.mkPen("y", width=2, style=QtCore.Qt.DashLine))
            self.vb.addItem(self.reg_roi)
        elif self.reg_roi is not None:
            self.vb.removeItem(self.reg_roi)
            self.reg_roi = None

    def _reg_region(self):
        if self.reg_roi is None:
            return None
        h, w = self.scan.od.shape[1:]
        p, s = self.reg_roi.pos(), self.reg_roi.size()
        x0, y0 = int(max(0, p.x())), int(max(0, p.y()))
        x1, y1 = int(min(w, p.x() + s.x())), int(min(h, p.y() + s.y()))
        return (y0, y1, x0, x1) if (y1 - y0 > 8 and x1 - x0 > 8) else None

    # ── spectra ─────────────────────────────────────────────────────────
    def _next_color(self):
        used = {s.color for s in self.spectra}
        return next((c for c in COLORS if c not in used), COLORS[len(self.spectra) % len(COLORS)])

    def _add_spectrum(self, sp: Spectrum) -> Spectrum:
        sp.curve = self.plot.plot([], [], pen=pg.mkPen(sp.color, width=2), name=sp.name)
        sp.item = QtWidgets.QListWidgetItem(sp.name)
        sp.item.setFlags(sp.item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEditable)
        sp.item.setCheckState(QtCore.Qt.Checked)
        sp.item.setForeground(QtGui.QColor(sp.color))
        self.list.blockSignals(True)
        self.list.addItem(sp.item)
        self.list.blockSignals(False)
        self.list.setCurrentItem(sp.item)
        self.spectra.append(sp)
        return sp

    def add_roi(self, kind):
        if self.scan is None:
            return
        h, w = self.scan.od.shape[1:]
        self._n_roi += 1
        color = self._next_color()
        pen = pg.mkPen(color, width=2)
        cx, cy, s = w / 2, h / 2, min(h, w) / 8
        if kind == "rect":
            roi = pg.RectROI([cx - s, cy - s], [2 * s, 2 * s], pen=pen)
        elif kind == "ellipse":
            roi = pg.EllipseROI([cx - s, cy - s], [2 * s, 2 * s], pen=pen)
        else:
            roi = pg.PolyLineROI([[cx - s, cy - s], [cx + s, cy - s], [cx, cy + s]],
                                 closed=True, pen=pen)
        self.vb.addItem(roi)
        sp = self._add_spectrum(Spectrum(f"Region {self._n_roi}", color, roi=roi))
        roi.sigRegionChangeFinished.connect(lambda _r, sp=sp: self._update_spectrum(sp))
        self._update_spectrum(sp)

    def keep_point(self):
        if self.point_spec is None:
            return
        sp = self.point_spec
        n = sum(1 for s in self.spectra if s.point is not None)
        sp.name = f"Point {n}"
        sp.item.setText(sp.name)
        self._rename_curve(sp)
        self.point_spec = None

    def _rename_curve(self, sp):
        legend = self.plot.plotItem.legend
        if legend is not None:
            legend.removeItem(sp.curve)
            if sp.item.checkState() == QtCore.Qt.Checked:
                legend.addItem(sp.curve, sp.name)

    def _extract(self, sp: Spectrum) -> np.ndarray:
        """Absorption spectrum: mean over the point / region of -ln(I/I0)
        (flat-field corrected) or of -ln(I) (raw)."""
        flat = self._flat_on()
        stack = self.scan.od if flat else self.scan.raw
        to_abs = (lambda v: v) if flat else (lambda v: -np.log(v))
        n = stack.shape[0]
        if sp.point is not None:
            r, c = sp.point
            k = self.pt_size.value() // 2
            block = stack[:, max(0, r - k):r + k + 1, max(0, c - k):c + k + 1]
            return np.nanmean(to_abs(block.reshape(n, -1)), axis=1)
        region = sp.roi.getArrayRegion(stack, self.img, axes=(1, 2), order=0)
        mask = sp.roi.getArrayRegion(np.ones(stack.shape[1:], np.float32), self.img,
                                     axes=(0, 1), order=0) > 0.5
        if region.size == 0 or not mask.any():
            return np.full(n, np.nan)
        return np.nanmean(to_abs(region[:, mask]), axis=1)

    def _update_spectrum(self, sp: Optional[Spectrum]):
        if sp is None or self.scan is None:
            return
        with np.errstate(all="ignore"):
            sp.values = self._extract(sp)
        self._replot(sp)
        self._update_rfactor()

    def _shown(self, values):
        """(values to plot, E0 or None) — normalised if requested."""
        if values is None:
            return None, None
        if not self.norm_chk.isChecked():
            return values, None
        norm, e0, _ = spectra.normalize(self.scan.energy, values,
                                        self.pre_region.getRegion(),
                                        self.post_region.getRegion())
        return norm, e0

    def _replot(self, sp: Spectrum):
        y, e0 = self._shown(sp.values)
        if y is None:
            return
        sp.curve.setData(self.scan.energy, y, connect="finite")
        sp.curve.setVisible(sp.item.checkState() == QtCore.Qt.Checked)
        if e0 is not None and sp.item is self.list.currentItem():
            self.e0_lbl.setText(f"E0 ({sp.name}) = {e0:.1f} eV")

    def _ref_shown(self):
        """Reference energies (shifted) and values, cropped near the scan and
        normalised like the spectra when requested."""
        e_ref, mu_ref, _ = self.ref
        e_ref = e_ref + self.ref_shift.value()
        if self.scan is not None:
            e = self.scan.energy
            margin = max(20.0, 0.25 * (e[-1] - e[0]))
            m = (e_ref >= e[0] - margin) & (e_ref <= e[-1] + margin)
            if m.sum() >= 2:
                e_ref, mu_ref = e_ref[m], mu_ref[m]
        if self.norm_chk.isChecked():
            mu_ref, _, _ = spectra.normalize(e_ref, mu_ref, self.pre_region.getRegion(),
                                             self.post_region.getRegion())
        return e_ref, mu_ref

    def _replot_all(self):
        self.e0_lbl.setText("")
        for sp in self.spectra:
            self._replot(sp)
        if self.ref is not None:
            e, mu = self._ref_shown()
            if self.ref_curve is None:
                self.ref_curve = self.plot.plot([], [], name=f"Reference: {self.ref[2]}",
                                                pen=pg.mkPen("w", width=2, style=QtCore.Qt.DashLine))
            self.ref_curve.setData(e, mu)
        self._update_rfactor()

    def _toggle_norm(self, on):
        for r in (self.pre_region, self.post_region):
            (self.plot.addItem if on else self.plot.removeItem)(r)
        self._set_plot_label()
        self._replot_all()
        self.plot.enableAutoRange()

    def _on_item_changed(self, item):
        sp = next((s for s in self.spectra if s.item is item), None)
        if sp is None:
            return
        if item.text() != sp.name:
            sp.name = item.text()
            self._rename_curve(sp)
        visible = item.checkState() == QtCore.Qt.Checked
        sp.curve.setVisible(visible)
        if sp.roi is not None:
            sp.roi.setVisible(visible)
        if sp.marker is not None:
            sp.marker.setVisible(visible)

    def _remove_spectrum(self, sp: Spectrum):
        self.plot.removeItem(sp.curve)
        if sp.roi is not None:
            self.vb.removeItem(sp.roi)
        if sp.marker is not None:
            self.vb.removeItem(sp.marker)
        self.list.takeItem(self.list.row(sp.item))
        self.spectra.remove(sp)
        if sp is self.point_spec:
            self.point_spec = None

    def remove_selected(self):
        item = self.list.currentItem()
        sp = next((s for s in self.spectra if s.item is item), None)
        if sp:
            self._remove_spectrum(sp)
            self._update_rfactor()

    def _update_rfactor(self):
        item = self.list.currentItem()
        sp = next((s for s in self.spectra if s.item is item), None)
        if self.ref is None or sp is None or sp.values is None:
            self.rf_lbl.setText("")
            return
        y, _ = self._shown(sp.values)
        e_ref, mu_ref = self._ref_shown()
        rf = spectra.r_factor(self.scan.energy, y, e_ref, mu_ref)
        self.rf_lbl.setText(f"R-factor {sp.name} vs reference: {rf:.4g}")

    # ── reference ───────────────────────────────────────────────────────
    def load_reference_dialog(self):
        start = next((d for d in XANES_GUI_CURVES if os.path.isdir(d)), os.getcwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference spectrum", start,
            "Spectra (*.npy *.txt *.dat *.csv *.xmu *.nor);;All files (*)")
        if path:
            self.load_reference(path)

    def load_reference(self, path):
        try:
            e, mu = io.load_reference(path)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Reference", f"Cannot read {path}:\n{ex}")
            return
        self.clear_reference()
        self.ref = (e, mu, os.path.basename(path))
        self.ref_lbl.setText(f"{os.path.basename(path)}  ({e[0]:.0f}–{e[-1]:.0f} eV, {len(e)} points)")
        self._replot_all()

    def clear_reference(self):
        if self.ref_curve is not None:
            self.plot.removeItem(self.ref_curve)
            self.ref_curve = None
        self.ref = None
        self.ref_lbl.setText("none — Load reference…")
        self._update_rfactor()

    # ── registration ────────────────────────────────────────────────────
    def register_dialog(self):
        if self.scan is None:
            return
        region = self._reg_region()
        dlg = RegisterDialog(self, self.scan, self.slider.value(), region is not None)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        ref_i = dlg.ref.value() - 1
        mode = "neighbour" if dlg.mode.currentIndex() == 1 else "reference"
        region = region if dlg.use_region.isChecked() else None
        smooth, up = dlg.smooth.value(), dlg.upsample.value()
        od, raw = self.scan.od, self.scan.raw
        # Compare the images as shown (corrected or raw); the shifts are then
        # applied to both stacks so the toggle stays consistent.
        compare = od if self._flat_on() else raw

        def compute(p, s):
            return registration.register_stack(compare, ref_i, region, up, smooth, mode, p, s)

        def computed(shifts):
            if ShiftsDialog(self, self.scan.energy, shifts).exec_() != QtWidgets.QDialog.Accepted:
                self.statusBar().showMessage("Registration discarded.")
                return

            def apply(p, s):
                n = len(shifts)
                registration.apply_shifts(od, shifts, lambda i, _n: p(i, 2 * n))
                registration.apply_shifts(raw, shifts, lambda i, _n: p(n + i, 2 * n))
                return shifts

            def applied(sh):
                prev = self.scan.shifts
                self.scan.shifts = sh if prev is None else prev + sh
                self._show_frame()
                for sp in self.spectra:
                    self._update_spectrum(sp)
                self.statusBar().showMessage(
                    f"Registered to image {ref_i + 1} ({self.scan.energy[ref_i]:.1f} eV); "
                    f"largest shift {np.abs(sh).max():.2f} px.")
            self._run("Applying shifts…", apply, applied)
        self._run("Registering…", compute, computed)

    def save_aligned(self):
        if self.scan is None:
            return
        base = os.path.splitext(self.scan.path)[0] + "_aligned.h5"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save aligned stack", base, "HDF5 (*.h5)")
        if not path:
            return
        self._run("Saving…", lambda p, s: io.save_aligned(self.scan, path) or path,
                  lambda pth: self.statusBar().showMessage(f"Saved {pth}"))

    # ── export ──────────────────────────────────────────────────────────
    def export_dialog(self):
        shown = [s for s in self.spectra
                 if s.values is not None and s.item.checkState() == QtCore.Qt.Checked]
        if not shown:
            QtWidgets.QMessageBox.information(self, "Export", "No spectra to export: click "
                                              "the image or add a region first.")
            return
        base = os.path.splitext(self.scan.path)[0] + "_spectra.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export spectra", base,
                                                        "CSV (*.csv);;Text (*.txt)")
        if path:
            self.export(path, shown)

    def export(self, path, shown=None):
        shown = shown if shown is not None else [s for s in self.spectra if s.values is not None]
        e = self.scan.energy
        cols, header = {}, [f"source: {self.scan.path}",
                            f"binning: {self.scan.binning}, dark: {self.scan.dark}",
                            ("flat-field correction: on, each image divided by its own reference; "
                             "absorption = -ln(I/I0)") if self._flat_on() else
                            "flat-field correction: OFF (raw images); absorption = -ln(I)"]
        header.append("registration: " + ("applied, shifts (dy,dx px) per energy: " +
                      "; ".join(f"{a:.2f},{b:.2f}" for a, b in self.scan.shifts)
                      if self.scan.shifts is not None else "none"))
        norm = self.norm_chk.isChecked()
        if norm:
            header.append(f"normalization: pre-edge {self.pre_region.getRegion()}, "
                          f"post-edge {self.post_region.getRegion()} eV")
        for sp in shown:
            header.append(sp.describe())
            cols[sp.name] = sp.values
            if norm:
                cols[f"{sp.name} (normalized)"] = self._shown(sp.values)[0]
        if self.ref is not None:
            e_ref, mu_ref = self._ref_shown()
            header.append(f"reference: {self.ref[2]}, energy shift {self.ref_shift.value():.2f} eV"
                          + (" (normalized)" if norm else ""))
            cols["reference"] = np.interp(e, e_ref, mu_ref, left=np.nan, right=np.nan)
        spectra.write_csv(path, e, cols, header)
        self.statusBar().showMessage(f"Exported {len(shown)} spectra to {path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="XANES 2D analysis: registration and spectra")
    ap.add_argument("file", nargs="?", help="XANES 2D HDF5 file to open")
    ap.add_argument("--bin", type=int, choices=(1, 2, 4), default=None,
                    help="binning when loading (default 2)")
    args = ap.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    w = MainWindow()
    if args.bin:
        w.binning.setCurrentText(str(args.bin))
    w.show()
    if args.file:
        w.open_file(args.file)
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
