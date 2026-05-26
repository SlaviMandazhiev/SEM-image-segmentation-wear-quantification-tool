#!/usr/bin/env python3
"""
RSA_Seg_GUI.py  -  Graphical front-end for the SEM wear-analysis pipeline.

Usage:
    python RSA_Seg_GUI.py

Workflow:
  1. Open Image         -> displays original SEM image
  2. Set Scale          -> draw a line over the scale bar, enter its real length in um
  3. Set Crop Region    -> enter pixel ranges (optional; blank = full image)
  4. Run Analysis       -> pipeline runs on the cropped image with the calibrated scale
  5. Adjust Boundaries  -> drag sliders and click Apply to update the wear map
  6. Save Results       -> write all output files to a chosen folder
  "Back to Setup" in the top bar lets you recalibrate or re-crop at any time.
"""

import os
import sys
import tempfile
import importlib.util
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from typing import Optional

import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle as MplRectangle

# ---------------------------------------------------------------------------
# Load analysis module dynamically (filename has dots/dashes → not importable)
# ---------------------------------------------------------------------------
# When frozen by PyInstaller (--onefile), bundled files live in sys._MEIPASS
_HERE = (
    sys._MEIPASS
    if getattr(sys, "frozen", False)
    else os.path.dirname(os.path.abspath(__file__))
)
_ANALYSIS_FILE = os.path.join(_HERE, "RSA-Seg-adaptive-V2.8_1.py")

if not os.path.isfile(_ANALYSIS_FILE):
    raise FileNotFoundError(
        f"Analysis module not found:\n{_ANALYSIS_FILE}\n"
        "Ensure RSA-Seg-adaptive-V2.8_1.py is in the same folder as this script."
    )

_spec = importlib.util.spec_from_file_location("sem_analysis", _ANALYSIS_FILE)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

process_one_image                       = _mod.process_one_image
mask_generator                          = _mod.mask_generator
auswertung_file_creator                 = _mod.auswertung_file_creator
create_scale_plots_histogram            = _mod.create_scale_plots_histogram
get_intensity_boundaries_from_histogram = _mod.get_intensity_boundaries_from_histogram
flaeche_pixel                           = _mod.flaeche_pixel

_DEFAULT_FIXED_BOUNDARY = _mod.fixed_boundary
_DEFAULT_OFFSET_VALUE   = _mod.offset_value
_DEFAULT_ADJUST_FE      = _mod.adjust_fe
_DEFAULT_ADJUST_W       = _mod.adjust_w
_DEFAULT_SCALE_POWER    = 0.1


# ---------------------------------------------------------------------------
class _CompactToolbar(NavigationToolbar2Tk):
    """Toolbar with only Home / Back / Forward / Zoom-to-rectangle."""
    toolitems = [
        item for item in NavigationToolbar2Tk.toolitems
        if item[0] in ("Home", "Back", "Forward", "Zoom")
    ]

    def set_message(self, msg):
        pass  # suppress the coordinate readout label


# ---------------------------------------------------------------------------
class SEMAnalyzerApp:
    """Two-phase GUI: Setup (load/calibrate/crop) then Results (adjust/save)."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("SEM Wear Analyzer")
        self.root.minsize(980, 720)

        # ── pre-analysis state ──────────────────────────────────────────────
        self.image_path: Optional[str]              = None
        self.original_image: Optional[np.ndarray]  = None   # full loaded grayscale

        # Scale calibration
        self._pixel_area: float       = flaeche_pixel
        self._cal_cids: list          = []
        self._cal_start               = None
        self._cal_line_artist         = None
        self._cal_info_var            = tk.StringVar(value="Not calibrated: using built-in default")

        # Crop region (empty string = full image)
        self._cx0 = tk.StringVar()
        self._cx1 = tk.StringVar()
        self._cy0 = tk.StringVar()
        self._cy1 = tk.StringVar()
        self._crop_patch: Optional[MplRectangle] = None
        for v in (self._cx0, self._cx1, self._cy0, self._cy1):
            v.trace_add("write", lambda *_: self._update_crop_preview())

        # ── post-analysis state ────────────────────────────────────────────
        self.adjusted_image: Optional[np.ndarray]  = None
        self.solid_tool_mask: Optional[np.ndarray] = None
        self.hist_adjusted: Optional[np.ndarray]   = None
        self.pipeline_results: dict                = {}
        self.pixel_counts: dict                    = {}
        self.color_image: Optional[np.ndarray]     = None
        self.current_boundaries: list              = [7, 100, 200]

        # Mode: 3 = Coating+Adhesion+Substrate,  2 = Coating+Adhesion only
        self._mode_var = tk.IntVar(value=3)

        # Boundary slider IntVars
        self._bvar = [tk.IntVar(value=v) for v in [7, 100, 200]]

        self._build_ui()

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        # ── top bar ─────────────────────────────────────────────────────────
        top = ttk.Frame(self.root, padding=(8, 5))
        top.grid(row=0, column=0, sticky="ew")

        ttk.Button(top, text="Open Image...", command=self.open_image).pack(side="left")
        self._file_label = ttk.Label(top, text="No image selected.", foreground="gray")
        self._file_label.pack(side="left", padx=10)

        self._back_btn = ttk.Button(
            top, text="← Back to Setup", command=self._show_setup_phase, state="disabled"
        )
        self._back_btn.pack(side="right")

        # ── image panels (always visible) ────────────────────────────────────
        display = ttk.Frame(self.root)
        display.grid(row=1, column=0, sticky="nsew", padx=6, pady=4)
        display.columnconfigure(0, weight=1)
        display.columnconfigure(1, weight=1)
        display.rowconfigure(0, weight=1)

        self._fig_left, self._ax_left, self._canvas_left = self._make_image_panel(
            display, col=0, title="SEM Image"
        )
        self._fig_right, self._ax_right, self._canvas_right = self._make_image_panel(
            display, col=1, title="Wear Map"
        )

        # Zoom / pan toolbar for the left panel (row 1, col 0; row 1 has no weight → stays compact)
        self._toolbar = _CompactToolbar(self._canvas_left, display, pack_toolbar=False)
        self._toolbar.update()
        self._toolbar.grid(row=1, column=0, sticky="ew", padx=2, pady=(0, 2))

        # ── SETUP PHASE controls (row 2) ─────────────────────────────────────
        self._setup_frame = ttk.Frame(self.root)
        self._setup_frame.grid(row=2, column=0, sticky="ew")
        self._setup_frame.columnconfigure(0, weight=1)

        # Step 1 — Scale calibration
        cal = ttk.LabelFrame(
            self._setup_frame, text="Step 1: Scale Calibration", padding=(10, 6)
        )
        cal.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 3))
        cal.columnconfigure(1, weight=1)

        self._cal_btn = ttk.Button(
            cal, text="Draw Scale Line on Image",
            command=self._enter_cal_mode, state="disabled"
        )
        self._cal_btn.grid(row=0, column=0, padx=(0, 12), sticky="w")
        ttk.Label(cal, textvariable=self._cal_info_var, foreground="darkgreen").grid(
            row=0, column=1, sticky="w"
        )

        # Step 2 — Crop
        crop = ttk.LabelFrame(
            self._setup_frame,
            text="Step 2: Crop Region  (optional: leave blank to use full image)",
            padding=(10, 6)
        )
        crop.grid(row=1, column=0, sticky="ew", padx=6, pady=3)

        ttk.Label(crop, text="X  (columns):").grid(row=0, column=0, padx=(0, 4), sticky="e")
        ttk.Entry(crop, textvariable=self._cx0, width=7).grid(row=0, column=1)
        ttk.Label(crop, text=" to ").grid(row=0, column=2)
        ttk.Entry(crop, textvariable=self._cx1, width=7).grid(row=0, column=3, padx=(0, 20))

        ttk.Label(crop, text="Y  (rows):").grid(row=0, column=4, padx=(0, 4), sticky="e")
        ttk.Entry(crop, textvariable=self._cy0, width=7).grid(row=0, column=5)
        ttk.Label(crop, text=" to ").grid(row=0, column=6)
        ttk.Entry(crop, textvariable=self._cy1, width=7).grid(row=0, column=7)

        self._img_size_label = ttk.Label(crop, text="", foreground="gray")
        self._img_size_label.grid(row=0, column=8, padx=(16, 0), sticky="w")

        # Step 3 — Mode + Run
        run_bar = ttk.Frame(self._setup_frame, padding=(6, 4))
        run_bar.grid(row=2, column=0, sticky="ew")
        run_bar.columnconfigure(0, weight=1)

        mode_frame = ttk.LabelFrame(run_bar, text="Step 3: Segmentation Mode", padding=(6, 2))
        mode_frame.grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(
            mode_frame, text="3 Regions  (Coating + Adhesion + Substrate)",
            variable=self._mode_var, value=3
        ).pack(side="left", padx=6)
        ttk.Radiobutton(
            mode_frame, text="2 Regions  (Coating + Adhesion)",
            variable=self._mode_var, value=2
        ).pack(side="left", padx=6)

        self._run_btn = ttk.Button(
            run_bar, text="▶   Run Analysis", command=self._run_analysis, state="disabled"
        )
        self._run_btn.grid(row=0, column=1, padx=(12, 0), sticky="e")

        # ── RESULTS PHASE controls (row 2, hidden initially) ─────────────────
        self._results_frame = ttk.Frame(self.root)
        self._results_frame.grid(row=2, column=0, sticky="ew")
        self._results_frame.columnconfigure(0, weight=1)
        self._results_frame.grid_remove()

        # Boundary sliders
        ctrl = ttk.LabelFrame(
            self._results_frame, text="Intensity Boundaries", padding=(10, 6)
        )
        ctrl.grid(row=0, column=0, sticky="ew", padx=6, pady=(4, 3))
        ctrl.columnconfigure(1, weight=1)

        ttk.Label(ctrl, text="Background:", width=22, anchor="e").grid(
            row=0, column=0, padx=(0, 6), pady=2, sticky="e"
        )
        ttk.Scale(ctrl, from_=0, to=255, orient="horizontal",
                  variable=self._bvar[0]).grid(row=0, column=1, sticky="ew", pady=2)
        ttk.Label(ctrl, textvariable=self._bvar[0], width=4, anchor="w").grid(
            row=0, column=2, padx=(6, 0)
        )

        ttk.Label(ctrl, text="Coating / Adhesion:", width=22, anchor="e").grid(
            row=1, column=0, padx=(0, 6), pady=2, sticky="e"
        )
        ttk.Scale(ctrl, from_=0, to=255, orient="horizontal",
                  variable=self._bvar[1]).grid(row=1, column=1, sticky="ew", pady=2)
        ttk.Label(ctrl, textvariable=self._bvar[1], width=4, anchor="w").grid(
            row=1, column=2, padx=(6, 0)
        )

        self._b2_label = ttk.Label(ctrl, text="Adhesion / Substrate:", width=22, anchor="e")
        self._b2_label.grid(row=2, column=0, padx=(0, 6), pady=2, sticky="e")
        self._b2_slider = ttk.Scale(ctrl, from_=0, to=255, orient="horizontal",
                                    variable=self._bvar[2])
        self._b2_slider.grid(row=2, column=1, sticky="ew", pady=2)
        self._b2_value = ttk.Label(ctrl, textvariable=self._bvar[2], width=4, anchor="w")
        self._b2_value.grid(row=2, column=2, padx=(6, 0))

        ttk.Button(
            ctrl, text="Apply Boundaries", command=self._apply_boundaries
        ).grid(row=3, column=0, columnspan=3, pady=(10, 2))

        # Stats + save
        bottom = ttk.Frame(self._results_frame, padding=(8, 4))
        bottom.grid(row=1, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        self._stats_var = tk.StringVar(value="")
        ttk.Label(bottom, textvariable=self._stats_var, foreground="navy").grid(
            row=0, column=0, sticky="w"
        )
        self._save_btn = ttk.Button(
            bottom, text="Save Results...", command=self.save_results
        )
        self._save_btn.grid(row=0, column=1, sticky="e")

    @staticmethod
    def _make_image_panel(parent, col: int, title: str):
        fig    = Figure(figsize=(5, 4), dpi=96)
        ax     = fig.add_subplot(111)
        ax.set_title(title, fontsize=8, pad=4)
        ax.axis("off")
        fig.tight_layout(pad=1.2)
        canvas = FigureCanvasTkAgg(fig, master=parent)
        canvas.get_tk_widget().grid(row=0, column=col, sticky="nsew", padx=2)
        return fig, ax, canvas

    # -----------------------------------------------------------------------
    # Phase switching
    # -----------------------------------------------------------------------
    def _show_setup_phase(self) -> None:
        self._results_frame.grid_remove()
        self._setup_frame.grid()
        self._back_btn.config(state="disabled")
        # Restore original image (with any existing scale line) in left panel
        if self.original_image is not None:
            self._redraw_original_image()
            self._update_crop_preview()
        # Clear right panel
        self._ax_right.clear()
        self._ax_right.set_title("Wear Map", fontsize=8, pad=4)
        self._ax_right.axis("off")
        self._canvas_right.draw()

    def _show_results_phase(self) -> None:
        self._setup_frame.grid_remove()
        self._results_frame.grid()
        self._back_btn.config(state="normal")
        # Enforce b2 visibility based on current mode
        self._set_b2_visibility(self._mode_var.get() == 3)

    def _set_b2_visibility(self, visible: bool) -> None:
        if visible:
            self._b2_label.grid()
            self._b2_slider.grid()
            self._b2_value.grid()
        else:
            self._b2_label.grid_remove()
            self._b2_slider.grid_remove()
            self._b2_value.grid_remove()

    # -----------------------------------------------------------------------
    # Image loading
    # -----------------------------------------------------------------------
    def open_image(self) -> None:
        path = filedialog.askopenfilename(
            title="Select SEM Image",
            filetypes=[("JPEG images", "*.jpg *.jpeg"), ("All files", "*.*")],
        )
        if not path:
            return

        img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            messagebox.showerror("Error", f"Could not read image:\n{path}")
            return

        self.image_path    = path
        self.original_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w               = self.original_image.shape

        self._file_label.config(text=os.path.basename(path), foreground="black")
        self._img_size_label.config(text=f"Image size:  {w} × {h} px")

        # Clear the visual scale line (it was drawn on the previous image)
        # but keep _pixel_area and calibration label so the user doesn't have to recalibrate
        self._cal_line_artist = None
        self._crop_patch      = None
        for v in (self._cx0, self._cx1, self._cy0, self._cy1):
            v.set("")

        # Enable setup buttons
        self._cal_btn.config(state="normal")
        self._run_btn.config(state="normal")

        # Show the original image and switch to setup phase
        self._show_setup_phase()

    def _redraw_original_image(self) -> None:
        """Redraw the original grayscale image in the left panel (preserves artists)."""
        # Collect existing overlay artists (scale line + crop rect)
        overlays = []
        if self._cal_line_artist is not None:
            overlays.append(self._cal_line_artist)

        self._ax_left.clear()
        self._ax_left.set_title("SEM Image", fontsize=8, pad=4)
        self._ax_left.axis("off")
        self._ax_left.imshow(self.original_image, cmap="gray", vmin=0, vmax=255)

        # Re-add overlay artists
        for artist in overlays:
            self._ax_left.add_line(artist)

        self._canvas_left.draw_idle()
        self._toolbar.update()  # reset home view to the newly loaded image

    # -----------------------------------------------------------------------
    # Crop preview
    # -----------------------------------------------------------------------
    def _parse_crop(self):
        """Return (x0, y0, x1, y1) from entries, or None for any missing/invalid value."""
        if self.original_image is None:
            return None
        h, w = self.original_image.shape
        try:
            x0 = int(self._cx0.get()) if self._cx0.get().strip() else 0
            x1 = int(self._cx1.get()) if self._cx1.get().strip() else w
            y0 = int(self._cy0.get()) if self._cy0.get().strip() else 0
            y1 = int(self._cy1.get()) if self._cy1.get().strip() else h
        except ValueError:
            return None
        x0 = max(0, min(x0, w - 1))
        x1 = max(x0 + 1, min(x1, w))
        y0 = max(0, min(y0, h - 1))
        y1 = max(y0 + 1, min(y1, h))
        return x0, y0, x1, y1

    def _update_crop_preview(self) -> None:
        """Draw/update a green rectangle on the left panel showing the crop region."""
        if self.original_image is None:
            return
        # Remove old patch
        if self._crop_patch is not None:
            try:
                self._crop_patch.remove()
            except ValueError:
                pass
            self._crop_patch = None

        coords = self._parse_crop()
        if coords is not None:
            x0, y0, x1, y1 = coords
            h, w = self.original_image.shape
            # Only draw if it's a real sub-region
            if not (x0 == 0 and y0 == 0 and x1 == w and y1 == h):
                self._crop_patch = MplRectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    linewidth=2, edgecolor="lime", facecolor="none", linestyle="--"
                )
                self._ax_left.add_patch(self._crop_patch)

        self._canvas_left.draw_idle()

    # -----------------------------------------------------------------------
    # Scale calibration
    # -----------------------------------------------------------------------
    def _enter_cal_mode(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No Image", "Open an image first.")
            return

        # Remove previous line if any
        if self._cal_line_artist is not None:
            try:
                self._cal_line_artist.remove()
            except ValueError:
                pass
            self._cal_line_artist = None
            self._canvas_left.draw_idle()

        # Deactivate any active toolbar zoom/pan so it won't intercept mouse events
        mode_str = str(self._toolbar.mode).lower()
        if "zoom" in mode_str:
            self._toolbar.zoom()
        elif "pan" in mode_str:
            self._toolbar.pan()

        self._cal_start = None
        self._cal_btn.config(state="disabled")
        self._cal_info_var.set("Click and drag over the scale bar...")
        self._canvas_left.get_tk_widget().config(cursor="crosshair")

        self._cal_cids = [
            self._canvas_left.mpl_connect("button_press_event",   self._cal_on_press),
            self._canvas_left.mpl_connect("motion_notify_event",  self._cal_on_motion),
            self._canvas_left.mpl_connect("button_release_event", self._cal_on_release),
        ]

    def _cal_disconnect(self) -> None:
        for cid in self._cal_cids:
            self._canvas_left.mpl_disconnect(cid)
        self._cal_cids = []
        self._canvas_left.get_tk_widget().config(cursor="")
        self._cal_btn.config(state="normal")

    def _cal_on_press(self, event) -> None:
        if event.inaxes != self._ax_left or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._cal_start = (event.xdata, event.ydata)
        if self._cal_line_artist is not None:
            try:
                self._cal_line_artist.remove()
            except ValueError:
                pass
        self._cal_line_artist, = self._ax_left.plot(
            [event.xdata, event.xdata], [event.ydata, event.ydata],
            color="yellow", linewidth=2, solid_capstyle="round"
        )
        self._canvas_left.draw_idle()

    def _cal_on_motion(self, event) -> None:
        if self._cal_start is None or event.inaxes != self._ax_left:
            return
        if event.xdata is None or event.ydata is None:
            return
        x0, y0 = self._cal_start
        # Keep Y fixed at the start row so the line is always horizontal
        self._cal_line_artist.set_data([x0, event.xdata], [y0, y0])
        self._canvas_left.draw_idle()

    def _cal_on_release(self, event) -> None:
        if self._cal_start is None or event.button != 1:
            self._cal_disconnect()
            self._cal_info_var.set("Calibration cancelled.")
            return
        if event.xdata is None or event.ydata is None:
            self._cal_disconnect()
            self._cal_info_var.set("Released outside image: please try again.")
            return

        x0, y0 = self._cal_start
        # Line is horizontal — pixel distance is the horizontal span only
        px_dist = abs(event.xdata - x0)
        # Snap the final line to the same Y so it stays perfectly horizontal
        if self._cal_line_artist is not None:
            self._cal_line_artist.set_data([x0, event.xdata], [y0, y0])
            self._canvas_left.draw_idle()
        self._cal_disconnect()

        if px_dist < 2:
            self._cal_info_var.set("Line too short: please try again.")
            if self._cal_line_artist is not None:
                try:
                    self._cal_line_artist.remove()
                except ValueError:
                    pass
                self._cal_line_artist = None
                self._canvas_left.draw_idle()
            self._cal_start = None
            return

        px_adjusted, real_len = self._ask_calibration_values(x0, event.xdata, y0)

        if px_adjusted is not None and real_len is not None:
            um_per_px        = real_len / px_adjusted
            self._pixel_area = um_per_px ** 2
            self._cal_info_var.set(
                f"Calibrated:  {um_per_px:.4f} µm/pixel  "
                f"({self._pixel_area:.6f} µm²/pixel²)"
            )
        else:
            if self._cal_line_artist is not None:
                try:
                    self._cal_line_artist.remove()
                except ValueError:
                    pass
                self._cal_line_artist = None
                self._canvas_left.draw_idle()
            self._cal_info_var.set("Calibration cancelled: using previous scale.")

        self._cal_start = None

    def _ask_calibration_values(self, x0: float, x1: float, y0: float):
        """
        Modal dialog showing the line's start/end X coordinates (editable).
        An "Update Line" button visually moves the line without closing the dialog.
        OK/Cancel confirm or discard.
        Returns (px_distance, real_um) or (None, None) if cancelled.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Scale Calibration")
        dialog.resizable(False, False)
        dialog.grab_set()

        pad = {"padx": (14, 6), "pady": 5, "sticky": "e"}

        # Start X
        ttk.Label(dialog, text="Start X (pixels):", anchor="e").grid(row=0, column=0, **pad)
        start_var = tk.StringVar(value=str(round(x0)))
        ttk.Entry(dialog, textvariable=start_var, width=10).grid(
            row=0, column=1, padx=(0, 14), pady=5, sticky="w"
        )

        # End X
        ttk.Label(dialog, text="End X (pixels):", anchor="e").grid(row=1, column=0, **pad)
        end_var = tk.StringVar(value=str(round(x1)))
        ttk.Entry(dialog, textvariable=end_var, width=10).grid(
            row=1, column=1, padx=(0, 14), pady=5, sticky="w"
        )

        # Update Line button — visually moves the line, does NOT close the dialog
        def on_update_line(*_):
            try:
                new_x0 = float(start_var.get())
                new_x1 = float(end_var.get())
            except ValueError:
                messagebox.showwarning("Invalid Input", "Start and End must be numbers.", parent=dialog)
                return
            if self._cal_line_artist is not None:
                self._cal_line_artist.set_data([new_x0, new_x1], [y0, y0])
                self._canvas_left.draw_idle()

        ttk.Button(dialog, text="Update Line on Image", command=on_update_line).grid(
            row=2, column=0, columnspan=2, pady=(6, 10)
        )

        ttk.Separator(dialog, orient="horizontal").grid(
            row=3, column=0, columnspan=2, sticky="ew", padx=10
        )

        # Real length in µm
        ttk.Label(dialog, text="Real length (µm):", anchor="e").grid(
            row=4, column=0, padx=(14, 6), pady=(10, 5), sticky="e"
        )
        um_var = tk.StringVar()
        um_entry = ttk.Entry(dialog, textvariable=um_var, width=10)
        um_entry.grid(row=4, column=1, padx=(0, 14), pady=(10, 5), sticky="w")
        um_entry.focus_set()

        result = {"px": None, "um": None}

        def on_ok(*_):
            try:
                new_x0 = float(start_var.get())
                new_x1 = float(end_var.get())
                um     = float(um_var.get())
                px     = abs(new_x1 - new_x0)
                if px <= 0 or um <= 0:
                    raise ValueError
                result["px"] = px
                result["um"] = um
                dialog.destroy()
            except ValueError:
                messagebox.showwarning(
                    "Invalid Input", "Please enter valid positive numbers.", parent=dialog
                )

        def on_cancel(*_):
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.grid(row=5, column=0, columnspan=2, pady=(4, 12))
        ttk.Button(btn_frame, text="OK",     command=on_ok,     width=8).pack(side="left", padx=4)
        ttk.Button(btn_frame, text="Cancel", command=on_cancel, width=8).pack(side="left", padx=4)

        dialog.bind("<Return>", on_ok)
        dialog.bind("<Escape>", on_cancel)

        # Position at the right edge of the screen, vertically centred
        dialog.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        x  = sw - dialog.winfo_width() - 20
        y  = (sh - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")

        self.root.wait_window(dialog)
        return result["px"], result["um"]

    # -----------------------------------------------------------------------
    # Analysis
    # -----------------------------------------------------------------------
    def _run_analysis(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("No Image", "Open an image first.")
            return

        # Apply crop
        coords = self._parse_crop()
        if coords is not None:
            x0, y0, x1, y1 = coords
            image_to_analyze = self.original_image[y0:y1, x0:x1]
        else:
            image_to_analyze = self.original_image

        if image_to_analyze.size == 0:
            messagebox.showerror("Crop Error", "The crop region is empty. Please check the values.")
            return

        # Write the (possibly cropped) image to a temp file so process_one_image can read it
        tmp_path = os.path.join(tempfile.gettempdir(), "_sem_analysis_input.jpg")
        cv2.imwrite(tmp_path, image_to_analyze)

        self._stats_var.set("Running analysis, please wait...")
        self._run_with_cursor(lambda: self._do_pipeline(tmp_path))

        # Clean up temp file
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    def _do_pipeline(self, image_path: str) -> None:
        mode = self._mode_var.get()

        self.pipeline_results = process_one_image(
            image_path     = image_path,
            adjust_fe      = _DEFAULT_ADJUST_FE,
            adjust_w       = _DEFAULT_ADJUST_W,
            offset_value   = _DEFAULT_OFFSET_VALUE,
            fixed_boundary = _DEFAULT_FIXED_BOUNDARY,
            n_materials    = mode,
        )

        self.adjusted_image      = self.pipeline_results["adjusted_image"]
        self.solid_tool_mask     = self.pipeline_results["tool_mask"]
        self.hist_adjusted       = self.pipeline_results["histogram"]
        self.current_boundaries  = list(self.pipeline_results["boundaries"])

        # Sync sliders
        for i, val in enumerate(self.current_boundaries):
            self._bvar[i].set(val)

        # Update mask
        self.pixel_counts, self.color_image = mask_generator(
            self.adjusted_image, self.current_boundaries
        )

        # Switch to results phase
        self._show_results_phase()
        self._refresh_left_panel_adjusted()
        self._refresh_right_panel()
        self._refresh_stats()

    # -----------------------------------------------------------------------
    # Results: boundary adjustment
    # -----------------------------------------------------------------------
    def _apply_boundaries(self) -> None:
        if self.adjusted_image is None:
            return

        mode = self._mode_var.get()
        if mode == 3:
            b = [v.get() for v in self._bvar]
            if not (b[0] < b[1] < b[2]):
                messagebox.showwarning(
                    "Invalid Boundaries",
                    "Thresholds must increase:\n"
                    "Background  <  Coating/Adhesion  <  Adhesion/Substrate\n"
                    f"Current values:  {b[0]},  {b[1]},  {b[2]}"
                )
                return
        else:
            b = [self._bvar[0].get(), self._bvar[1].get()]
            if not (b[0] < b[1]):
                messagebox.showwarning(
                    "Invalid Boundaries",
                    "Background threshold must be lower than Coating/Adhesion threshold.\n"
                    f"Current values:  {b[0]},  {b[1]}"
                )
                return

        self.current_boundaries = b
        self._run_with_cursor(self._update_mask)

    def _update_mask(self) -> None:
        if self.adjusted_image is None:
            return
        self.pixel_counts, self.color_image = mask_generator(
            self.adjusted_image, self.current_boundaries
        )
        self._refresh_right_panel()
        self._refresh_stats()

    # -----------------------------------------------------------------------
    # Display helpers
    # -----------------------------------------------------------------------
    def _refresh_left_panel_adjusted(self) -> None:
        self._ax_left.clear()
        self._ax_left.set_title("Adjusted SEM Image", fontsize=8, pad=4)
        self._ax_left.axis("off")
        if self.adjusted_image is not None:
            self._ax_left.imshow(self.adjusted_image, cmap="gray", vmin=0, vmax=255)
        self._canvas_left.draw()

    def _refresh_right_panel(self) -> None:
        mode = self._mode_var.get()
        title = (
            "Wear Map  (Blue=Coating  |  Red=Adhesion  |  Green=Substrate)"
            if mode == 3 else
            "Wear Map  (Blue=Coating  |  Red=Adhesion)"
        )
        self._ax_right.clear()
        self._ax_right.set_title(title, fontsize=8, pad=4)
        self._ax_right.axis("off")
        if self.color_image is not None:
            self._ax_right.imshow(cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB))
        self._canvas_right.draw()

    def _refresh_stats(self) -> None:
        if not self.pixel_counts or self.solid_tool_mask is None:
            return
        n_tool = int(cv2.countNonZero(self.solid_tool_mask)) or 1
        wo = self.pixel_counts.get("wo_pixel_count", 0)
        fe = self.pixel_counts.get("fe_pixel_count", 0)
        ti = self.pixel_counts.get("ti_pixel_count", 0)

        if self._mode_var.get() == 3:
            undetected = max(0.0, 100.0 - (wo + fe + ti) * 100.0 / n_tool)
            self._stats_var.set(
                f"Coating: {ti*100/n_tool:.1f}%    "
                f"Adhesion: {fe*100/n_tool:.1f}%    "
                f"Substrate: {wo*100/n_tool:.1f}%    "
                f"Undetected: {undetected:.1f}%"
            )
        else:
            undetected = max(0.0, 100.0 - (fe + ti) * 100.0 / n_tool)
            self._stats_var.set(
                f"Coating: {ti*100/n_tool:.1f}%    "
                f"Adhesion: {fe*100/n_tool:.1f}%    "
                f"Undetected: {undetected:.1f}%"
            )

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    def save_results(self) -> None:
        if self.adjusted_image is None:
            messagebox.showwarning("No Data", "Run the analysis before saving.")
            return

        save_dir = filedialog.askdirectory(title="Select Output Folder")
        if not save_dir:
            return

        def _do_save() -> None:
            pixel_counts, _ = mask_generator(
                self.adjusted_image, self.current_boundaries, save_dir=save_dir
            )
            cv2.imwrite(os.path.join(save_dir, "Adjusted_Image.jpg"), self.adjusted_image)

            if "threshold_image" in self.pipeline_results:
                cv2.imwrite(
                    os.path.join(save_dir, "Erkannte_Flaeche.jpg"),
                    self.pipeline_results["threshold_image"],
                )

            results_for_report = dict(self.pipeline_results)
            results_for_report["boundaries"] = self.current_boundaries
            _, thresh = cv2.threshold(
                self.adjusted_image, self.current_boundaries[0], 255, cv2.THRESH_BINARY
            )
            results_for_report["white_pixels"] = int(np.sum(thresh == 255))

            auswertung_file_creator(
                os.path.join(save_dir, "Auswertung.txt"),
                results_for_report,
                pixel_counts,
                pixel_area=self._pixel_area,
            )

        self._run_with_cursor(_do_save)
        messagebox.showinfo("Saved", f"All results saved to:\n{save_dir}")

    # -----------------------------------------------------------------------
    # Utility
    # -----------------------------------------------------------------------
    def _run_with_cursor(self, fn) -> None:
        self.root.config(cursor="watch")
        self.root.update_idletasks()
        try:
            fn()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
        finally:
            self.root.config(cursor="")


# ---------------------------------------------------------------------------
def main() -> None:
    root = tk.Tk()
    SEMAnalyzerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
