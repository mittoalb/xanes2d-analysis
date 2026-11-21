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

from . import background as bgm
from . import sam3_backend
from . import io, refs, registration, spectra

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


SCALEBAR_DEFAULTS = {"pixel_nm": 16.0, "length": 1.0, "unit": "µm", "color": "#ffffff",
                     "thickness": 6, "font": 12, "position": "Bottom right"}
POSITIONS = {"Bottom right": (-30, -30), "Bottom left": (30, -30),
             "Top right": (-30, 30), "Top left": (30, 30)}


class ScaleBarItem(pg.GraphicsWidgetAnchor, pg.GraphicsObject):
    """Scale bar anchored to a corner of the view; its length is a fixed
    distance in image pixels, so it follows zooming."""

    def __init__(self):
        pg.GraphicsObject.__init__(self)
        pg.GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.size_px, self.thickness, self.right, self.bottom = 100.0, 6, True, True
        self.bar = QtWidgets.QGraphicsRectItem(self)
        self.text = pg.TextItem(anchor=(0.5, 1))
        self.text.setParentItem(self)

    def boundingRect(self):
        return QtCore.QRectF()

    def attach(self, view):
        self.setParentItem(view)
        view.sigRangeChanged.connect(self.update_bar)

    def configure(self, size_px, label, color, thickness, font_pt, position):
        self.size_px, self.thickness = float(size_px), int(thickness)
        c = QtGui.QColor(color)
        self.bar.setPen(pg.mkPen(c))
        self.bar.setBrush(pg.mkBrush(c))
        self.text.setText(label)
        self.text.setColor(c)
        f = QtGui.QFont()
        f.setPointSize(int(font_pt))
        self.text.setFont(f)
        off = POSITIONS.get(position, (-30, -30))
        self.right, self.bottom = off[0] <= 0, off[1] <= 0
        a = (1 if self.right else 0, 1 if self.bottom else 0)
        if self.parentItem() is not None:
            self.anchor(itemPos=a, parentPos=a, offset=off)
        self.update_bar()

    def update_bar(self, *_):
        view = self.parentItem()
        if view is None:
            return
        p1 = view.mapFromViewToItem(self, QtCore.QPointF(0, 0))
        p2 = view.mapFromViewToItem(self, QtCore.QPointF(self.size_px, 0))
        w = abs((p2 - p1).x())
        x0 = -w if self.right else 0.0
        y0 = -self.thickness if self.bottom else 0.0
        self.bar.setRect(QtCore.QRectF(x0, y0, w, self.thickness))
        self.text.setAnchor((0.5, 1) if self.bottom else (0.5, 0))
        self.text.setPos(x0 + w / 2.0, y0 if self.bottom else y0 + self.thickness)


def _settings() -> QtCore.QSettings:
    return QtCore.QSettings("xanes2d-analysis", "xanes2d-analysis")


def load_scalebar_settings() -> dict:
    st, out = _settings(), dict(SCALEBAR_DEFAULTS)
    for k, v in SCALEBAR_DEFAULTS.items():
        val = st.value(f"scalebar/{k}", v)
        out[k] = type(v)(val) if not isinstance(v, str) else str(val)
    return out


class ScaleBarDialog(QtWidgets.QDialog):
    def __init__(self, parent, cfg: dict, binning: int, file_pixel_nm=None):
        super().__init__(parent)
        self.setWindowTitle("Scale bar")
        self.cfg = dict(cfg)
        f = QtWidgets.QFormLayout(self)
        self.pixel = QtWidgets.QDoubleSpinBox()
        self.pixel.setRange(0.01, 100000)
        self.pixel.setDecimals(3)
        self.pixel.setSuffix(" nm")
        self.pixel.setValue(cfg["pixel_nm"])
        self.eff = QtWidgets.QLabel()
        self.pixel.valueChanged.connect(lambda v: self.eff.setText(
            f"= {v * binning:g} nm per displayed pixel (binning {binning})"))
        self.pixel.valueChanged.emit(self.pixel.value())
        f.addRow("Pixel size (full resolution):", self.pixel)
        f.addRow("", self.eff)
        if file_pixel_nm:
            note = QtWidgets.QLabel(f"The file recorded {file_pixel_nm:g} nm "
                                    "(32id:TXMOptics:ImagePixelSize) — not used.")
            note.setStyleSheet("color: #888;")
            f.addRow("", note)
        row = QtWidgets.QHBoxLayout()
        self.length = QtWidgets.QDoubleSpinBox()
        self.length.setRange(0.001, 100000)
        self.length.setDecimals(3)
        self.length.setValue(cfg["length"])
        self.unit = QtWidgets.QComboBox()
        self.unit.addItems(["nm", "µm", "mm"])
        self.unit.setCurrentText(cfg["unit"])
        row.addWidget(self.length)
        row.addWidget(self.unit)
        f.addRow("Length:", row)
        self.color_btn = QtWidgets.QPushButton()
        self._set_color(cfg["color"])
        self.color_btn.clicked.connect(self._pick_color)
        f.addRow("Colour:", self.color_btn)
        self.thick = QtWidgets.QSpinBox()
        self.thick.setRange(1, 50)
        self.thick.setValue(cfg["thickness"])
        self.thick.setSuffix(" px")
        f.addRow("Thickness:", self.thick)
        self.font = QtWidgets.QSpinBox()
        self.font.setRange(6, 72)
        self.font.setValue(cfg["font"])
        self.font.setSuffix(" pt")
        f.addRow("Text size:", self.font)
        self.pos = QtWidgets.QComboBox()
        self.pos.addItems(list(POSITIONS))
        self.pos.setCurrentText(cfg["position"])
        f.addRow("Position:", self.pos)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok |
                                        QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _set_color(self, c):
        self.cfg["color"] = c
        self.color_btn.setText(c)
        self.color_btn.setStyleSheet(f"background: {c}; color: {'#000' if QtGui.QColor(c).lightness() > 128 else '#fff'};")

    def _pick_color(self):
        c = QtWidgets.QColorDialog.getColor(QtGui.QColor(self.cfg["color"]), self)
        if c.isValid():
            self._set_color(c.name())

    def values(self) -> dict:
        return {"pixel_nm": self.pixel.value(), "length": self.length.value(),
                "unit": self.unit.currentText(), "color": self.cfg["color"],
                "thickness": self.thick.value(), "font": self.font.value(),
                "position": self.pos.currentText()}


class Sam3Dialog(QtWidgets.QDialog):
    """Where SAM 3 lives and how its result is used."""
    def __init__(self, parent):
        super().__init__(parent)
        self.setWindowTitle("SAM 3 settings")
        st = _settings()
        f = QtWidgets.QFormLayout(self)
        row = QtWidgets.QHBoxLayout()
        self.python = QtWidgets.QLineEdit(st.value("sam3/python", "") or "")
        self.python.setPlaceholderText(sam3_backend.find_python() or "path to the SAM 3 environment's python")
        browse = QtWidgets.QPushButton("…")
        browse.clicked.connect(self._browse)
        row.addWidget(self.python, 1)
        row.addWidget(browse)
        f.addRow("SAM 3 Python:", row)
        self.score = QtWidgets.QDoubleSpinBox()
        self.score.setRange(0, 1)
        self.score.setSingleStep(0.05)
        self.score.setValue(float(st.value("sam3/min_score", 0.5)))
        f.addRow("Keep objects with score ≥", self.score)
        self.ckpt = QtWidgets.QLineEdit(st.value("sam3/checkpoint", "") or "")
        self.ckpt.setPlaceholderText("blank = download from Hugging Face (needs access)")
        f.addRow("Checkpoint (optional):", self.ckpt)
        self.timeout = QtWidgets.QSpinBox()
        self.timeout.setRange(10, 7200)
        self.timeout.setSuffix(" s")
        self.timeout.setValue(int(st.value("sam3/timeout", 600)))
        f.addRow("Timeout:", self.timeout)
        help_lbl = QtWidgets.QLabel(sam3_backend.SETUP_HELP)
        help_lbl.setStyleSheet("color: #888; font-family: monospace;")
        help_lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        f.addRow(help_lbl)
        bb = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        f.addRow(bb)

    def _browse(self):
        p, _ = QtWidgets.QFileDialog.getOpenFileName(self, "SAM 3 environment's python",
                                                     os.path.expanduser("~"))
        if p:
            self.python.setText(p)

    def _save(self):
        st = _settings()
        st.setValue("sam3/python", self.python.text().strip())
        st.setValue("sam3/min_score", self.score.value())
        st.setValue("sam3/checkpoint", self.ckpt.text().strip())
        st.setValue("sam3/timeout", self.timeout.value())
        self.accept()


class ReferenceDialog(QtWidgets.QDialog):
    """Pick a reference spectrum from the local library (IXAS XAS Data
    Library copy), download / update it, or open any file."""

    def __init__(self, parent, scan_range=None):
        super().__init__(parent)
        self.setWindowTitle("Reference spectra")
        self.resize(900, 560)
        self.scan_range = scan_range
        self.chosen: Optional[str] = None
        v = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Element"))
        self.element = QtWidgets.QComboBox()
        self.element.currentIndexChanged.connect(self._fill)
        top.addWidget(self.element)
        self.in_range = QtWidgets.QCheckBox("Only spectra covering this scan's energies")
        self.in_range.setChecked(scan_range is not None)
        self.in_range.setEnabled(scan_range is not None)
        self.in_range.toggled.connect(self._fill)
        top.addWidget(self.in_range)
        top.addStretch()
        v.addLayout(top)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Element", "Edge", "Sample", "Beamline", "File"])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.doubleClicked.connect(lambda _: self._accept_selected())
        v.addWidget(self.table, 1)
        self.info = QtWidgets.QLabel("")
        self.info.setWordWrap(True)
        v.addWidget(self.info)
        row = QtWidgets.QHBoxLayout()
        self.dl_btn = QtWidgets.QPushButton("Download / update library")
        self.dl_btn.setToolTip(f"Copy the IXAS XAS Data Library ({refs.IXAS_URL}, public domain) "
                               f"into {refs.library_dir()}. Files already there are skipped.")
        self.dl_btn.clicked.connect(self._download)
        row.addWidget(self.dl_btn)
        self.dl_prog = QtWidgets.QProgressBar()
        self.dl_prog.hide()
        row.addWidget(self.dl_prog, 1)
        row.addStretch()
        b_file = QtWidgets.QPushButton("From file…")
        b_file.clicked.connect(self._from_file)
        row.addWidget(b_file)
        b_ok = QtWidgets.QPushButton("Load selected")
        b_ok.setDefault(True)
        b_ok.clicked.connect(self._accept_selected)
        row.addWidget(b_ok)
        b_cancel = QtWidgets.QPushButton("Cancel")
        b_cancel.clicked.connect(self.reject)
        row.addWidget(b_cancel)
        v.addLayout(row)
        self._job = None
        self._reload()

    def _reload(self):
        self._entries = refs.entries()
        self._ranges = {}
        cur = self.element.currentText()
        self.element.blockSignals(True)
        self.element.clear()
        self.element.addItem("All")
        self.element.addItems(sorted({e["element"] for e in self._entries}))
        if cur:
            self.element.setCurrentText(cur)
        self.element.blockSignals(False)
        self._fill()

    def _range(self, e):
        key = e["file"]
        if key not in self._ranges:
            try:
                d = refs.read_xdi(refs.entry_path(e))
                self._ranges[key] = (d["energy"][0], d["energy"][-1])
            except Exception:
                self._ranges[key] = (np.nan, np.nan)
        return self._ranges[key]

    def _fill(self):
        el = self.element.currentText()
        rows = [e for e in self._entries if el in ("", "All") or e["element"] == el]
        if self.in_range.isChecked() and self.scan_range is not None:
            lo, hi = self.scan_range
            rows = [e for e in rows if self._range(e)[0] <= lo and self._range(e)[1] >= hi]
        self._rows = rows
        self.table.setRowCount(len(rows))
        for r, e in enumerate(rows):
            for c, txt in enumerate((e["element"], e["edge"], " ".join(e["sample"].split()),
                                     e["beamline"], os.path.basename(e["file"]))):
                self.table.setItem(r, c, QtWidgets.QTableWidgetItem(txt))
        self.table.resizeColumnsToContents()
        if not self._entries:
            self.info.setText("The library is empty. Press 'Download / update library' "
                              f"(about 6 MB from {refs.IXAS_URL}), or use 'From file…'.")
        else:
            self.info.setText(f"{len(rows)} of {len(self._entries)} spectra shown — local "
                              f"library {refs.library_dir()} (IXAS XAS Data Library, public domain).")

    def _accept_selected(self):
        r = self.table.currentRow()
        if 0 <= r < len(self._rows):
            self.chosen = refs.entry_path(self._rows[r])
            self.accept()

    def _from_file(self):
        start = next((d for d in XANES_GUI_CURVES if os.path.isdir(d)), os.getcwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load reference spectrum", start,
            "Spectra (*.xdi *.npy *.txt *.dat *.csv *.xmu *.nor);;All files (*)")
        if path:
            self.chosen = path
            self.accept()

    def _download(self):
        if self._job is not None and self._job.isRunning():
            return
        self.dl_btn.setEnabled(False)
        self.dl_prog.show()
        self._job = Job(lambda p, s: refs.download_ixas(log=lambda m: None, progress=p, stop=s), self)
        self._job.progress.connect(lambda i, n: (self.dl_prog.setMaximum(n), self.dl_prog.setValue(i)))
        self._job.result.connect(lambda n: self.info.setText(f"Downloaded {n} new spectra."))
        self._job.failed.connect(lambda m: self.info.setText(f"Download failed: {m}"))
        self._job.finished.connect(lambda: (self.dl_btn.setEnabled(True), self.dl_prog.hide(),
                                            self._reload()))
        self._job.start()

    def reject(self):
        if self._job is not None and self._job.isRunning():
            self._job.stop()
            self._job.wait(5000)
        super().reject()


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
        self._maps = None                          # (step, edge energy) cache
        self.sample_mask: Optional[np.ndarray] = None
        self._contrast = None                      # post-edge mean used to segment
        self._bg_cache = {}                        # flat-field on/off → (N,) levels
        self.scalebar = None
        self.sb_cfg = load_scalebar_settings()
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
        self.act_save_shifts = tb.addAction("Save shifts…", self.save_shifts_dialog)
        self.act_save_shifts.setToolTip("Save the registration shifts (full-resolution pixels "
                                        "vs energy) to reuse them on other scans")
        self.act_load_shifts = tb.addAction("Load shifts…", self.load_shifts_dialog)
        self.act_load_shifts.setToolTip("Apply shifts saved from another scan (matched by "
                                        "energy, converted to this binning)")
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
        self.display.addItems(["Absorption (−ln I/I0)", "Transmission (I/I0)",
                               "Edge energy map", "Edge step map"])
        self.display.setToolTip(
            "Edge energy map: each pixel coloured by the energy where its spectrum "
            "reaches half the edge step (pre/post-edge ranges from the plot).\n"
            "Edge step map: size of the edge jump — where the element is.\n"
            "Run them after registration.")
        self.display.currentIndexChanged.connect(self._display_changed)
        crow.addWidget(self.display)
        self.cmap = QtWidgets.QComboBox()
        self.cmap.addItems(["turbo", "viridis", "plasma", "inferno", "CET-R4"])
        self.cmap.setToolTip("Colour scale of the maps")
        self.cmap.currentIndexChanged.connect(lambda _: self._show_frame(relevel=True))
        crow.addWidget(self.cmap)
        self.mask_lbl = QtWidgets.QLabel("  Mask step <")
        crow.addWidget(self.mask_lbl)
        self.mask = QtWidgets.QDoubleSpinBox()
        self.mask.setRange(0, 100)
        self.mask.setValue(15)
        self.mask.setSuffix(" % of max")
        self.mask.setToolTip("Pixels whose edge step is below this fraction of the largest "
                             "step (99th percentile) are left blank in the edge-energy map.")
        self.mask.valueChanged.connect(lambda _: self._show_frame(relevel=True))
        crow.addWidget(self.mask)
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
        self.sb_chk = QtWidgets.QCheckBox("Scale bar")
        self.sb_chk.toggled.connect(lambda _: self._update_scalebar())
        crow.addWidget(self.sb_chk)
        self.sb_btn = QtWidgets.QToolButton()
        self.sb_btn.setText("…")
        self.sb_btn.setToolTip("Scale bar settings: pixel size, length, colour, size, position")
        self.sb_btn.clicked.connect(self.scalebar_dialog)
        crow.addWidget(self.sb_btn)
        lv.addLayout(crow)

        brow = QtWidgets.QHBoxLayout()
        brow.addWidget(QtWidgets.QLabel("Background:"))
        self.seg_method = QtWidgets.QComboBox()
        self.seg_method.addItems(["Threshold", "SAM 3"])
        self.seg_method.setToolTip("How Find sample segments the sample in the post-edge image:\n"
                                   "Threshold — automatic (Otsu) threshold, editable;\n"
                                   "SAM 3 — Meta's Segment Anything 3 with a text prompt "
                                   "(needs a SAM 3 environment, see …).")
        self.seg_method.currentIndexChanged.connect(lambda _: self._bg_controls())
        brow.addWidget(self.seg_method)
        self.sam_prompt = QtWidgets.QLineEdit("particle")
        self.sam_prompt.setMaximumWidth(140)
        self.sam_prompt.setToolTip("What SAM 3 should find (text prompt), e.g. particle, cell, grain")
        brow.addWidget(self.sam_prompt)
        self.sam_btn = QtWidgets.QToolButton()
        self.sam_btn.setText("…")
        self.sam_btn.setToolTip("SAM 3 settings (environment, score, checkpoint)")
        self.sam_btn.clicked.connect(lambda: Sam3Dialog(self).exec_())
        brow.addWidget(self.sam_btn)
        self.btn_find = QtWidgets.QPushButton("Find sample")
        self.btn_find.setToolTip("Segment the sample in the mean of the post-edge images (red "
                                 "range on the plot) with an automatic (Otsu) threshold.")
        self.btn_find.clicked.connect(self.find_sample)
        brow.addWidget(self.btn_find)
        brow.addWidget(QtWidgets.QLabel("Threshold"))
        self.thr = QtWidgets.QDoubleSpinBox()
        self.thr.setRange(-100, 100)
        self.thr.setDecimals(3)
        self.thr.setSingleStep(0.01)
        self.thr.setToolTip("Pixels above this (post-edge mean absorption, lightly smoothed) "
                            "are sample. Set automatically by Find sample.")
        self.thr.valueChanged.connect(lambda _: self._segment())
        brow.addWidget(self.thr)
        self.bg_sub = QtWidgets.QCheckBox("Subtract background")
        self.bg_sub.setToolTip("At every energy subtract the median absorption of the background "
                               "(away from the sample): empty areas become 0, and beam-intensity "
                               "changes between image and reference are removed.")
        self.bg_sub.toggled.connect(lambda _: self._bg_changed())
        brow.addWidget(self.bg_sub)
        self.bg_hide = QtWidgets.QCheckBox("Hide background")
        self.bg_hide.setToolTip("Blank everything outside the sample in the image and the maps.")
        self.bg_hide.toggled.connect(lambda _: self._bg_changed(spectra_too=False))
        brow.addWidget(self.bg_hide)
        self.bg_outline = QtWidgets.QCheckBox("Outline")
        self.bg_outline.setChecked(True)
        self.bg_outline.toggled.connect(lambda _: self._draw_outline())
        brow.addWidget(self.bg_outline)
        self.bg_info = QtWidgets.QLabel("")
        self.bg_info.setStyleSheet("color: #888;")
        brow.addWidget(self.bg_info)
        brow.addStretch()
        lv.addLayout(brow)
        self.outline_img = pg.ImageItem()
        self.outline_img.setZValue(5)
        self.vb.addItem(self.outline_img)

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
            r.sigRegionChangeFinished.connect(self._ranges_changed)
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
                  self.act_save_shifts, self.act_load_shifts, self.btn_find,
                  self.btn_rect, self.btn_ellipse, self.btn_polygon, self.btn_keep,
                  self.btn_remove, self.reg_region_chk, self.norm_chk, self.slider,
                  self.sb_chk, self.sb_btn):
            w.setEnabled(on)
        self._bg_controls()
        self._map_controls()

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
        self._maps = None
        self._reset_background()
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
        self._update_scalebar()
        self._replot_all()

    # ── image ───────────────────────────────────────────────────────────
    def _flat_on(self) -> bool:
        return self.flat_chk.isChecked()

    def _frame(self, i):
        """Image i as displayed: flat-field corrected or raw, as absorption
        or transmission / intensity."""
        trans = self.display.currentIndex() == 1
        b = self._bg_levels()
        b = 0.0 if b is None else float(b[i])
        with np.errstate(divide="ignore", invalid="ignore"):
            if self._flat_on():
                f = self.scan.od[i] - b
                out = np.exp(-f) if trans else f
            elif trans:
                out = self.scan.raw[i] * np.exp(b)      # I / I_background when subtracting
            else:
                out = -np.log(self.scan.raw[i]) - b
        if self.bg_hide.isChecked() and self.sample_mask is not None:
            out = np.where(self.sample_mask, out, np.nan)
        return out

    def _toggle_flat(self, on):
        self.display.setItemText(0, "Absorption (−ln I/I0)" if on else "Absorption (−ln I), raw")
        self.display.setItemText(1, "Transmission (I/I0)" if on else "Intensity (counts), raw")
        if self.scan is None:
            return
        self._maps = None
        if self.sample_mask is not None and not self._sam_mode():
            self.find_sample()           # other quantity → new threshold and mask
        self._show_frame(relevel=True)
        for sp in self.spectra:
            self._update_spectrum(sp)
        self._set_plot_label()

    def _set_plot_label(self):
        base = "Absorption −ln(I/I0)" if self._flat_on() else "−ln(I), no flat-field correction"
        self.plot.setLabel("left", ("Normalized " + base[0].lower() + base[1:])
                           if self.norm_chk.isChecked() else base)

    def _map_mode(self) -> int:
        """0 = image, 1 = edge-energy map, 2 = edge-step map."""
        return max(0, self.display.currentIndex() - 1) if self.display.currentIndex() >= 2 else 0

    def _map_controls(self):
        m = self._map_mode()
        self.cmap.setVisible(m > 0)
        self.mask_lbl.setVisible(m == 1)
        self.mask.setVisible(m == 1)
        self.slider.setEnabled(self.scan is not None and m == 0)

    def _display_changed(self, _=None):
        self._map_controls()
        # the maps use the pre/post-edge ranges: show them on the plot
        show = self._map_mode() > 0 or self.norm_chk.isChecked()
        for r in (self.pre_region, self.post_region):
            if show and r.scene() is None:
                self.plot.addItem(r)
            elif not show and r.scene() is not None:
                self.plot.removeItem(r)
        self._show_frame(relevel=True)

    def _ranges_changed(self):
        self._maps = None
        self._replot_all()
        if self._map_mode():
            self._show_frame(relevel=True)

    def _get_maps(self):
        if self._maps is None and self.scan is not None:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
            try:
                stack = self._active_abs_stack()
                b = self._bg_levels()
                if b is not None:
                    stack = stack - b[:, None, None]
                self._maps = spectra.edge_maps(self.scan.energy, stack,
                                               self.pre_region.getRegion(),
                                               self.post_region.getRegion())
            finally:
                QtWidgets.QApplication.restoreOverrideCursor()
        return self._maps

    def _map_image(self):
        step, e_half = self._get_maps()
        hide = self.bg_hide.isChecked() and self.sample_mask is not None
        if self._map_mode() == 2:
            return np.where(self.sample_mask, step, np.nan) if hide else step
        ref = np.nanpercentile(step, 99) if np.isfinite(step).any() else 0
        out = e_half.copy()
        out[~(step >= ref * self.mask.value() / 100.0)] = np.nan
        if hide:
            out[~self.sample_mask] = np.nan
        return out

    def _show_frame(self, i=None, relevel=False):
        if self.scan is None:
            return
        i = self.slider.value() if i is None or isinstance(i, bool) else i
        mode = self._map_mode()
        if mode:
            frame = self._map_image()
            try:
                self.hist.gradient.setColorMap(pg.colormap.get(self.cmap.currentText()))
            except Exception:
                self.hist.gradient.loadPreset("viridis")
            finite = frame[np.isfinite(frame)]
            lo, hi = (np.percentile(finite, (2, 98)) if finite.size else (0, 1))
            self.img.setImage(frame, levels=(lo, hi), autoLevels=False)
            self._levels_set = False
            what = "edge energy (eV)" if mode == 1 else "edge step"
            self.e_lbl.setText(f"map: {what}")
            self.e_line.hide()
            return
        if relevel or not self._levels_set:
            self.hist.gradient.loadPreset("grey")
        frame = self._frame(i)
        if relevel or not self._levels_set:
            lo, hi = np.nanpercentile(frame, (1, 99))
            self.img.setImage(frame, levels=(lo, hi), autoLevels=False)
            self._levels_set = True
        else:
            self.img.setImage(frame, autoLevels=False)
        self.e_lbl.setText(f"{self.scan.energy[i]:.2f} eV  ({i + 1}/{self.scan.n})")
        self.e_line.show()
        self.e_line.setValue(self.scan.energy[i])

    # ── background ──────────────────────────────────────────────────────
    def _sam_mode(self) -> bool:
        return self.seg_method.currentIndex() == 1

    def _bg_controls(self):
        has = self.sample_mask is not None
        for wdg in (self.bg_sub, self.bg_hide, self.bg_outline):
            wdg.setEnabled(has)
        sam = self._sam_mode()
        self.thr.setEnabled(has and not sam)
        self.sam_prompt.setVisible(sam)
        self.sam_btn.setVisible(sam)

    def _active_abs_stack(self):
        """The absorption stack currently in use: -ln(I/I0) or -ln(I)."""
        if self._flat_on():
            return self.scan.od
        with np.errstate(divide="ignore", invalid="ignore"):
            return -np.log(self.scan.raw)

    def find_sample(self):
        if self.scan is None:
            return
        self._contrast = bgm.contrast_image(self.scan.energy, self._active_abs_stack(),
                                            self.post_region.getRegion())
        if self._sam_mode():
            self._find_sample_sam3()
            return
        t = bgm.otsu_threshold(self._contrast)
        self.thr.blockSignals(True)
        self.thr.setValue(t)
        self.thr.blockSignals(False)
        self._segment()

    def _find_sample_sam3(self):
        st = _settings()
        img, prompt = self._contrast, self.sam_prompt.text().strip() or "particle"
        py = st.value("sam3/python", "") or None
        score = float(st.value("sam3/min_score", 0.5))
        ckpt = st.value("sam3/checkpoint", "") or None
        timeout = float(st.value("sam3/timeout", 600))

        def run(p, s):
            return sam3_backend.segment(img, prompt, py, score, timeout, ckpt)

        def done(res):
            mask, n, scores = res
            if not mask.any():
                QtWidgets.QMessageBox.information(
                    self, "SAM 3", f"SAM 3 found nothing for '{prompt}' with score ≥ {score:g} "
                    f"(scores: {', '.join(f'{x:.2f}' for x in scores) or 'none'}). Try another "
                    "prompt or a lower score in SAM 3 settings.")
                return
            self._set_mask(mask, f"SAM 3 '{prompt}': {n} object(s)")
        self._run(f"SAM 3: segmenting '{prompt}' (loading the model can take a minute)…",
                  run, done)

    def _segment(self):
        if self.scan is None or self._contrast is None or self._sam_mode():
            return
        self._set_mask(bgm.segment_sample(self._contrast, self.thr.value()), "threshold")

    def _set_mask(self, mask, how: str):
        self.sample_mask = mask
        self._bg_cache = {}
        frac = 100.0 * self.sample_mask.mean()
        nbg = int(bgm.background_pixels(self.sample_mask).sum())
        self.bg_info.setText(f"{how}: sample {frac:.0f}% of image, {nbg} background px")
        self._bg_controls()
        self._draw_outline()
        self._bg_changed()

    def _draw_outline(self):
        if self.sample_mask is None or not self.bg_outline.isChecked():
            self.outline_img.clear()
            return
        from scipy import ndimage
        m = self.sample_mask
        edge = m & ~ndimage.binary_erosion(m)
        rgba = np.zeros(m.shape + (4,), np.ubyte)
        rgba[edge] = (0, 255, 255, 255)
        self.outline_img.setImage(rgba, levels=(0, 255))

    def _bg_levels(self) -> Optional[np.ndarray]:
        """Per-energy background absorption when 'Subtract background' is on."""
        if self.sample_mask is None or not self.bg_sub.isChecked():
            return None
        key = self._flat_on()
        if key not in self._bg_cache:
            bg = bgm.background_pixels(self.sample_mask)
            self._bg_cache[key] = bgm.background_levels(self._active_abs_stack(), bg)
        return self._bg_cache[key]

    def _bg_changed(self, spectra_too=True):
        if self.scan is None:
            return
        self._maps = None
        self._show_frame(relevel=True)
        if spectra_too:
            for sp in self.spectra:
                self._update_spectrum(sp)

    def _reset_background(self, keep_threshold=False):
        """Mask and levels are stale (new scan / registration): drop them, or
        re-segment with the same threshold."""
        self._bg_cache = {}
        if keep_threshold and self.sample_mask is not None and self._sam_mode():
            keep_threshold = False
            self.statusBar().showMessage("Sample moved: press Find sample again for a new SAM 3 mask.", 8000)
        if keep_threshold and self.sample_mask is not None and self.scan is not None:
            self._contrast = bgm.contrast_image(self.scan.energy, self._active_abs_stack(),
                                                self.post_region.getRegion())
            self._segment()
            return
        self.sample_mask = None
        self._contrast = None
        self.outline_img.clear()
        self.bg_info.setText("")
        self._bg_controls()

    # ── scale bar ───────────────────────────────────────────────────────
    def scalebar_dialog(self):
        if self.scan is None:
            return
        dlg = ScaleBarDialog(self, self.sb_cfg, self.scan.binning,
                             self.scan.info.get("file_pixel_nm"))
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.sb_cfg = dlg.values()
            st = _settings()
            for k, v in self.sb_cfg.items():
                st.setValue(f"scalebar/{k}", v)
            self.sb_chk.setChecked(True)
            self._update_scalebar()

    def _update_scalebar(self):
        on = self.sb_chk.isChecked() and self.scan is not None
        if self.scalebar is None:
            if not on:
                return
            self.scalebar = ScaleBarItem()
            self.scalebar.attach(self.vb)
        self.scalebar.setVisible(on)
        if not on:
            return
        c = self.sb_cfg
        nm = c["length"] * {"nm": 1.0, "µm": 1e3, "mm": 1e6}[c["unit"]]
        px = nm / (c["pixel_nm"] * self.scan.binning)          # in displayed pixels
        self.scalebar.configure(px, f"{c['length']:g} {c['unit']}", c["color"],
                                c["thickness"], c["font"], c["position"])

    def _on_hover(self, pos):
        if self.scan is None:
            return
        p = self.img.mapFromScene(pos)
        r, c = int(p.y()), int(p.x())
        h, w = self.scan.od.shape[1:]
        if 0 <= r < h and 0 <= c < w:
            mode = self._map_mode()
            if mode:
                v = self._map_image()[r, c]
                txt = (f"edge energy {v:.1f} eV" if mode == 1 else f"edge step {v:.3f}") \
                    if np.isfinite(v) else "masked"
                self.statusBar().showMessage(f"row {r}, col {c}: {txt}")
            else:
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
            b = self._bg_levels()
            if b is not None:
                sp.values = sp.values - b
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
        show = on or self._map_mode() > 0
        for r in (self.pre_region, self.post_region):
            if show and r.scene() is None:
                self.plot.addItem(r)
            elif not show and r.scene() is not None:
                self.plot.removeItem(r)
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
        rng = (self.scan.energy[0], self.scan.energy[-1]) if self.scan is not None else None
        dlg = ReferenceDialog(self, rng)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.chosen:
            self.load_reference(dlg.chosen)

    def start_auto_download(self):
        """First start: fetch the reference library in the background if the
        local copy is empty (XANES2D_NO_AUTO_DOWNLOAD=1 turns this off)."""
        if refs.load_index() or os.environ.get("XANES2D_NO_AUTO_DOWNLOAD"):
            return
        self.dl_job = Job(lambda p, s: refs.download_ixas(log=lambda m: None, progress=p, stop=s), self)
        self.dl_job.progress.connect(lambda i, n: self.statusBar().showMessage(
            f"Downloading the reference spectra library (first start): {i}/{n}", 3000))
        self.dl_job.result.connect(lambda n: self.statusBar().showMessage(
            f"Reference library ready: {n} spectra in {refs.library_dir()}", 8000))
        self.dl_job.failed.connect(lambda m: self.statusBar().showMessage(
            f"Reference library download failed ({m}) — use Load reference… → Download later.", 8000))
        self.dl_job.start()

    def load_reference(self, path):
        try:
            e, mu = io.load_reference(path)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Reference", f"Cannot read {path}:\n{ex}")
            return
        self.clear_reference()
        name = os.path.basename(path)
        if refs.is_xdi(path):
            meta = refs.read_xdi(path)["meta"]
            sample = " ".join((meta.get("Sample.name") or meta.get("Sample.formula") or "").split())
            if sample:
                name = f"{meta.get('Element.symbol', '')} {meta.get('Element.edge', '')} {sample}".strip()
        self.ref = (e, mu, name)
        self.ref_lbl.setText(f"{name}  ({e[0]:.0f}–{e[-1]:.0f} eV, {len(e)} points)")
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
            self._preview_and_apply(
                shifts, f"Registered to image {ref_i + 1} ({self.scan.energy[ref_i]:.1f} eV)")
        self._run("Registering…", compute, computed)

    def _preview_and_apply(self, shifts, what: str):
        """Show the shifts; on Apply, shift both stacks (corrected and raw)."""
        if ShiftsDialog(self, self.scan.energy, shifts).exec_() != QtWidgets.QDialog.Accepted:
            self.statusBar().showMessage("Shifts discarded.")
            return
        od, raw = self.scan.od, self.scan.raw

        def apply(p, s):
            n = len(shifts)
            registration.apply_shifts(od, shifts, lambda i, _n: p(i, 2 * n))
            registration.apply_shifts(raw, shifts, lambda i, _n: p(n + i, 2 * n))
            return shifts

        def applied(sh):
            prev = self.scan.shifts
            self.scan.shifts = sh if prev is None else prev + sh
            self._maps = None
            self._reset_background(keep_threshold=True)   # sample moved: re-segment
            self._show_frame()
            for sp in self.spectra:
                self._update_spectrum(sp)
            self.statusBar().showMessage(f"{what}; largest shift {np.abs(sh).max():.2f} px.")
        self._run("Applying shifts…", apply, applied)

    def save_shifts_dialog(self):
        if self.scan is None or self.scan.shifts is None:
            QtWidgets.QMessageBox.information(self, "Save shifts",
                                              "No shifts yet: register the images first.")
            return
        base = os.path.splitext(self.scan.path)[0] + "_shifts.csv"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save shifts", base,
                                                        "CSV (*.csv);;Text (*.txt)")
        if not path:
            return
        io.save_shifts(path, self.scan.energy, self.scan.shifts, self.scan.binning,
                       [f"source: {self.scan.path}",
                        f"registered at binning {self.scan.binning} (values here are full resolution)"])
        self.statusBar().showMessage(f"Saved shifts to {path}")

    def load_shifts_dialog(self):
        if self.scan is None:
            return
        start = os.path.dirname(self.scan.path)
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Load shifts", start,
                                                        "CSV (*.csv *.txt);;All files (*)")
        if path:
            self.load_shifts(path)

    def load_shifts(self, path):
        try:
            e_saved, full = io.load_shifts(path)
        except Exception as ex:
            QtWidgets.QMessageBox.warning(self, "Load shifts", f"Cannot read {path}:\n{ex}")
            return
        shifts, outside = io.shifts_for_scan(self.scan.energy, self.scan.binning, e_saved, full)
        what = f"Applied shifts from {os.path.basename(path)}"
        if outside:
            what += f" ({outside} energies outside the saved range used the nearest value)"
        self._preview_and_apply(shifts, what)

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
        b = self._bg_levels()
        if b is not None:
            nbg = int(bgm.background_pixels(self.sample_mask).sum())
            how = (f"SAM 3 prompt '{self.sam_prompt.text()}'" if self._sam_mode()
                   else f"threshold {self.thr.value():.4g}")
            header.append(f"background: subtracted per energy (median of {nbg} background px; "
                          f"sample from {how} on the post-edge mean); "
                          "column 'background' = subtracted level")
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
        if b is not None:
            cols["background"] = b
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
    w.start_auto_download()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
