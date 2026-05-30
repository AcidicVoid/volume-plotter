"""
audio_volume_graph.py
---------------------
Reads all audio files from a folder and plots their volume over time.

For each file:
  - Volume is computed in short frames and converted to dBFS.
  - Time axis is normalised to 0–100% of song duration.
  - Individual tracks are drawn in semi-transparent blue.
  - A smoothed median curve is drawn on top in yellow.
  - A raw (unsmoothed) mean curve is drawn on top in purple.
  - A red horizontal line marks 0 dB.
  - Y-axis is fixed to -40 dB … +5 dB.

Interactive song list (separate Tk panel, optional):
  - Scrollable listbox with all track names.
  - Click to highlight / click again to un-highlight.
  - Multiple tracks can be highlighted simultaneously.
  - Highlighting uses blitting for near-instant redraws.
  - Disabled automatically when saving to PNG (--output flag).
  - Can also be disabled with --no-song-list.

Audio files are loaded in parallel using a thread pool for faster I/O.

Dependencies:
    pip install numpy matplotlib soundfile librosa
    (librosa handles MP3/FLAC/OGG via audioread; soundfile handles WAV/FLAC directly)
    Tkinter is included with standard Python on Windows/macOS/most Linux distros.
"""

import os
import sys
import tkinter as tk
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.lines import Line2D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Use the Tk backend so we can embed matplotlib inside a Tk window
matplotlib.use("TkAgg")

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------

DB_MIN, DB_MAX = -40.0, 5.0

# Normal track appearance
TRACK_COLOR  = "#4fc3f7"
TRACK_ALPHA  = 0.25
TRACK_WIDTH  = 0.6
TRACK_ZORDER = 2

# Highlighted track appearance
HI_COLOR     = "#ffffff"
HI_ALPHA     = 0.90
HI_WIDTH     = 1.0
HI_ZORDER    = 6      # above normal tracks; median (8) and mean (9) stay on top

YELLOW_WIDTH = 2.5
PURPLE_WIDTH = 2.5

LIST_FONT    = ("Segoe UI", 11)   # Tk font for the song list
TITLE_FONT   = ("Segoe UI", 16, "bold")

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a"}


def find_audio_files(folder: str) -> list[str]:
    """Return a sorted list of audio file paths found in *folder*."""
    paths = []
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder, name))
    return paths


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file as a mono float32 array.
    Tries soundfile first (WAV/FLAC/OGG), falls back to librosa (MP3/M4A/…).
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        return data.mean(axis=1).astype(np.float32), sr
    except Exception:
        import librosa
        mono, sr = librosa.load(path, sr=None, mono=True)
        return mono, sr


def process_file(path: str, frame_ms: int, n_points: int) -> tuple[str, np.ndarray] | None:
    """
    Full pipeline for one file: load → RMS dB → resample to n_points.
    Returns (filename, curve) or None on error. Safe to call from a thread.
    """
    name = os.path.basename(path)
    try:
        samples, sr = load_audio_mono(path)
        db          = compute_rms_db(samples, sr, frame_ms)
        resampled   = normalise_to_percent(db, n_points)
        print(f"  OK  {name}  ({sr} Hz, {len(samples)/sr:.1f} s)")
        return name, resampled
    except Exception as exc:
        print(f"  SKIP  {name}  — {exc}")
        return None


def load_tracks_parallel(paths: list[str], frame_ms: int,
                          n_points: int) -> tuple[list[np.ndarray], list[str]]:
    """
    Load all files concurrently via a thread pool.

    Threads (not processes) are used because the bottleneck is I/O / decoder
    wait time, not CPU. The GIL is released during those waits, so all threads
    make real progress in parallel. A ProcessPoolExecutor would add pickle
    overhead for the large NumPy arrays without any benefit here.

    Results are re-sorted by filename so order matches find_audio_files.
    """
    results: dict[str, np.ndarray] = {}
    max_workers = min(len(paths), os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(process_file, p, frame_ms, n_points): p for p in paths}
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results[result[0]] = result[1]

    sorted_names = sorted(results)
    return [results[n] for n in sorted_names], sorted_names


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------

def compute_rms_db(samples: np.ndarray, sr: int, frame_ms: int = 50) -> np.ndarray:
    """
    Split samples into non-overlapping frames and return per-frame RMS in dBFS.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames  = len(samples) // frame_len
    frames    = samples[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms       = np.sqrt(np.mean(frames ** 2, axis=1))
    rms       = np.where(rms < 1e-10, 1e-10, rms)   # guard log(0)
    return (20.0 * np.log10(rms)).astype(np.float32)


def normalise_to_percent(db: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """Resample db curve to n_points so all tracks share a 0–100 % time axis."""
    src = np.linspace(0.0, 100.0, len(db))
    dst = np.linspace(0.0, 100.0, n_points)
    return np.interp(dst, src, db).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistical curves
# ---------------------------------------------------------------------------

def compute_median_curve(tracks: list[np.ndarray]) -> np.ndarray:
    """Per-point median across all tracks."""
    return np.median(np.stack(tracks), axis=0)


def compute_mean_curve(tracks: list[np.ndarray]) -> np.ndarray:
    """Per-point arithmetic mean across all tracks (unsmoothed)."""
    return np.mean(np.stack(tracks), axis=0)


def smooth_curve(curve: np.ndarray, window: int = 21) -> np.ndarray:
    """Uniform moving-average; window is auto-clamped to odd & ≤ len(curve)."""
    window = min(window, len(curve))
    if window % 2 == 0:
        window -= 1
    return np.convolve(curve, np.ones(window) / window, mode="same")


# ---------------------------------------------------------------------------
# Blitted highlight engine
# ---------------------------------------------------------------------------

class HighlightController:
    """
    Manages track highlighting using matplotlib blitting for fast redraws.

    How blitting works here
    -----------------------
    After the figure is first drawn, we capture a pixel snapshot of the axes
    (the "background"). On every highlight toggle we:
      1. Restore that snapshot (instantly, no Python drawing).
      2. Re-draw only the currently highlighted lines on top of it.
      3. Blit (copy) the result to the screen.

    This avoids re-rendering the 59 static blue lines, grid, axes, legend,
    etc. on every click — which is what made it slow before.

    The background is re-captured whenever the figure is resized.
    """

    def __init__(self, fig: plt.Figure, ax: plt.Axes,
                 track_lines: list[Line2D]) -> None:
        self.fig         = fig
        self.ax          = ax
        self.lines       = track_lines
        self.highlighted = set()    # indices of currently highlighted tracks
        self._bg         = None     # cached background bitmap

        # Capture background after the first full draw
        fig.canvas.mpl_connect("draw_event",   self._on_draw)
        fig.canvas.mpl_connect("resize_event", self._on_resize)

    # --- background management -------------------------------------------

    def _capture_bg(self) -> None:
        """Snapshot the current canvas (all static elements already drawn)."""
        self._bg = self.fig.canvas.copy_from_bbox(self.ax.bbox)

    def _on_draw(self, _event) -> None:
        """Re-cache background whenever matplotlib does a full redraw."""
        self._capture_bg()

    def _on_resize(self, _event) -> None:
        """Invalidate cache on resize; it will be rebuilt on the next draw."""
        self._bg = None

    # --- highlight logic -------------------------------------------------

    def toggle(self, idx: int) -> None:
        """Toggle the highlight state of track *idx* and blit the update."""
        if idx in self.highlighted:
            self.highlighted.discard(idx)
            line = self.lines[idx]
            line.set_color(TRACK_COLOR)
            line.set_alpha(TRACK_ALPHA)
            line.set_linewidth(TRACK_WIDTH)
            line.set_zorder(TRACK_ZORDER)
        else:
            self.highlighted.add(idx)
            line = self.lines[idx]
            line.set_color(HI_COLOR)
            line.set_alpha(HI_ALPHA)
            line.set_linewidth(HI_WIDTH)
            line.set_zorder(HI_ZORDER)

        self._blit_highlights()

    def _blit_highlights(self) -> None:
        """Restore background, draw highlighted lines, push to screen."""
        canvas = self.fig.canvas

        if self._bg is None:
            # Fallback: no cached background yet — do a normal redraw once
            canvas.draw_idle()
            return

        # 1. Restore the static background
        canvas.restore_region(self._bg)

        # 2. Draw only the highlighted lines into the axes
        for idx in self.highlighted:
            self.ax.draw_artist(self.lines[idx])

        # 3. Push the updated axes rectangle to the screen
        canvas.blit(self.ax.bbox)


# ---------------------------------------------------------------------------
# Scrollable song list (native Tk panel)
# ---------------------------------------------------------------------------

class SongListPanel:
    """
    A native Tkinter panel showing a scrollable list of track names.

    Clicking a name calls HighlightController.toggle() and updates the
    visual selection state (background colour) in the listbox itself.

    Why Tk instead of matplotlib CheckButtons?
    ------------------------------------------
    CheckButtons has no built-in scroll support. Fitting 59 items into a
    fixed axes area either makes the text unreadably small or clips entries.
    A Tk Listbox + Scrollbar is a single-widget solution that handles any
    number of tracks, uses the OS scrollbar, and responds to the mouse wheel.
    """

    def __init__(self, parent: tk.Widget,
                 labels: list[str],
                 controller: HighlightController) -> None:
        self.controller  = controller
        self.highlighted = set()   # mirrors controller.highlighted for colour sync

        # --- outer frame -----------------------------------------------------
        frame = tk.Frame(parent, bg="#0f3460")
        frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(6, 8), pady=8)

        # --- title -----------------------------------------------------------
        tk.Label(frame, text="Songs", font=TITLE_FONT,
                 bg="#0f3460", fg="white").pack(pady=(4, 6))

        # --- listbox + scrollbar ---------------------------------------------
        box_frame = tk.Frame(frame, bg="#0f3460")
        box_frame.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(box_frame, orient=tk.VERTICAL)
        self.listbox = tk.Listbox(
            box_frame,
            yscrollcommand = scrollbar.set,
            font           = LIST_FONT,
            bg             = "#1a1a2e",
            fg             = "white",
            selectbackground = "#1a1a2e",   # suppress default blue selection
            selectforeground = "white",
            activestyle    = "none",
            width          = 36,
            highlightthickness = 0,
            borderwidth    = 0,
        )
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Populate
        for name in labels:
            display = name[:40] + "…" if len(name) > 41 else name
            self.listbox.insert(tk.END, display)

        # Use Button-1 press (not release) so we read nearest() before Tk
        # moves its internal selection cursor — that shift was causing the
        # one-click-behind bug.  We also suppress Tk's built-in selection
        # highlight entirely so our itemconfig colours are never overwritten.
        self.listbox.bind("<Button-1>", self._on_click)
        self.listbox.bind("<<ListboxSelect>>", lambda e: self.listbox.selection_clear(0, tk.END))
        # Mouse-wheel scroll (cross-platform)
        self.listbox.bind("<MouseWheel>",      self._on_scroll)   # Windows/macOS
        self.listbox.bind("<Button-4>",        self._on_scroll)   # Linux scroll up
        self.listbox.bind("<Button-5>",        self._on_scroll)   # Linux scroll down

    def _on_click(self, event: tk.Event) -> None:
        """Toggle highlight for whichever item was clicked."""
        idx = self.listbox.nearest(event.y)
        if idx < 0:
            return

        # Update our own state first, then reflect it in the widget.
        # (Do NOT read back from listbox.curselection — Tk updates that
        # after the event handler returns, making it one click behind.)
        if idx in self.highlighted:
            self.highlighted.discard(idx)
            self.listbox.itemconfig(idx, bg="#1a1a2e", fg="white")
        else:
            self.highlighted.add(idx)
            self.listbox.itemconfig(idx, bg="#ffffff", fg="#0a0a1a")

        self.controller.toggle(idx)

    def _on_scroll(self, event: tk.Event) -> None:
        """Forward mouse-wheel events to the listbox for smooth scrolling."""
        if event.num == 4:          # Linux scroll up
            self.listbox.yview_scroll(-1, "units")
        elif event.num == 5:        # Linux scroll down
            self.listbox.yview_scroll(1, "units")
        else:                       # Windows / macOS: event.delta is ±120
            self.listbox.yview_scroll(-1 * (event.delta // 120), "units")


# ---------------------------------------------------------------------------
# Main plot
# ---------------------------------------------------------------------------

def build_axes(fig: plt.Figure, with_song_list: bool) -> plt.Axes:
    """Create and style the main graph axes, leaving room for the song list."""
    # When the song list is shown the canvas itself is narrower (the Tk
    # listbox sits outside matplotlib), so the axes can fill the full figure.
    ax = fig.add_axes([0.06, 0.10, 0.91, 0.84])
    ax.set_facecolor("#16213e")
    return ax


def draw_graph(ax: plt.Axes, tracks: list[np.ndarray],
               median: np.ndarray, mean: np.ndarray
               ) -> list[Line2D]:
    """
    Draw all static elements onto *ax*.
    Returns the list of individual track Line2D objects for later manipulation.
    """
    x = np.linspace(0.0, 100.0, len(tracks[0]))

    # Individual tracks (blue, semi-transparent)
    track_lines: list[Line2D] = []
    for db in tracks:
        (line,) = ax.plot(
            x, np.clip(db, DB_MIN, DB_MAX),
            color=TRACK_COLOR, alpha=TRACK_ALPHA,
            linewidth=TRACK_WIDTH, zorder=TRACK_ZORDER,
        )
        track_lines.append(line)

    # 0 dB reference (red dashed)
    ax.axhline(0.0, color="#ef5350", linewidth=1.4, linestyle="--", zorder=3)

    # Median (yellow) — zorder 8: always above highlighted tracks (6)
    ax.plot(x, np.clip(median, DB_MIN, DB_MAX),
            color="#ffeb3b", linewidth=YELLOW_WIDTH, zorder=8)

    # Mean (purple, unsmoothed) — zorder 9: on top of everything
    ax.plot(x, np.clip(mean, DB_MIN, DB_MAX),
            color="#ce93d8", linewidth=PURPLE_WIDTH, zorder=9)

    # Axes formatting
    ax.set_xlim(0, 100)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.set_xlabel("Song time (%)", color="white", fontsize=11)
    ax.set_ylabel("Volume (dBFS)", color="white", fontsize=11)
    ax.set_title(f"Volume overlay — {len(tracks)} track(s)",
                 color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which="major", color="#ffffff22", linewidth=0.6)
    ax.grid(which="minor", color="#ffffff0d", linewidth=0.3)
    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff33")

    # Legend
    ax.legend(
        handles=[
            Line2D([0], [0], color=TRACK_COLOR, alpha=0.7, linewidth=1.5,
                   label=f"Tracks ({len(tracks)})"),
            Line2D([0], [0], color="#ffeb3b", linewidth=2.5, label="Median"),
            Line2D([0], [0], color="#ce93d8", linewidth=2.5, label="Mean"),
            Line2D([0], [0], color="#ef5350", linewidth=1.4, linestyle="--",
                   label="0 dB"),
        ],
        facecolor="#0f3460", edgecolor="#ffffff33",
        labelcolor="white", fontsize=9, loc="upper right",
    )

    return track_lines


def plot_volume_overlay(tracks: list[np.ndarray],
                        labels: list[str],
                        median: np.ndarray,
                        mean: np.ndarray,
                        output_path: str | None = None,
                        show_song_list: bool = True) -> None:
    """
    Render the volume overlay.

    When interactive (no output_path) and show_song_list is True, opens a
    Tk window with the matplotlib canvas on the left and a scrollable song
    list on the right. Highlighting uses blitting for fast updates.

    When output_path is given, saves a plain PNG (no interactive elements).
    """

    # --- PNG export (no interactivity needed) --------------------------------
    if output_path is not None:
        fig = plt.figure(figsize=(14, 6), facecolor="#1a1a2e")
        ax  = build_axes(fig, with_song_list=False)
        draw_graph(ax, tracks, median, mean)
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {output_path}")
        plt.close(fig)
        return

    # --- interactive Tk window -----------------------------------------------
    root = tk.Tk()
    root.title("Volume Overlay")
    root.configure(bg="#1a1a2e")

    # Matplotlib figure embedded in Tk
    fig = plt.Figure(figsize=(14, 7), facecolor="#1a1a2e")
    ax  = build_axes(fig, with_song_list=show_song_list)
    track_lines = draw_graph(ax, tracks, median, mean)

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    canvas.draw()

    # Song list panel (optional)
    if show_song_list:
        controller = HighlightController(fig, ax, track_lines)
        _panel_ref = SongListPanel(root, labels, controller)  # held by root

    root.mainloop()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(folder: str,
         output_png: str | None = None,
         frame_ms: int = 50,
         n_points: int = 1000,
         smooth_window: int = 31,
         show_song_list: bool = True) -> None:
    """
    Entry point.

    Parameters
    ----------
    folder         : directory containing audio files
    output_png     : save PNG here; if None, opens an interactive window
    frame_ms       : RMS frame length in milliseconds
    n_points       : shared x-axis resolution (time points per track)
    smooth_window  : moving-average kernel width for the median curve
    show_song_list : show the interactive scrollable song list (interactive mode only)
    """
    paths = find_audio_files(folder)
    if not paths:
        print(f"No supported audio files found in '{folder}'.")
        sys.exit(1)

    print(f"Found {len(paths)} audio file(s) in '{folder}'. Loading in parallel…\n")

    tracks, labels = load_tracks_parallel(paths, frame_ms, n_points)
    if not tracks:
        print("No tracks could be loaded.")
        sys.exit(1)

    print(f"\nComputing statistics across {len(tracks)} track(s) …")
    smooth_med = smooth_curve(compute_median_curve(tracks), window=smooth_window)
    raw_mean   = compute_mean_curve(tracks)   # plain per-point average, no smoothing

    print("Rendering graph …")
    plot_volume_overlay(tracks, labels, smooth_med, raw_mean,
                        output_path=output_png,
                        show_song_list=show_song_list)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overlay volume graphs for all audio files in a folder."
    )
    parser.add_argument("folder",
                        help="Path to the folder containing audio files.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save graph to PNG instead of opening interactively.")
    parser.add_argument("--frame-ms", type=int, default=50,
                        help="RMS frame length in milliseconds (default: 50).")
    parser.add_argument("--points", type=int, default=1000,
                        help="Number of time points on the shared x-axis (default: 1000).")
    parser.add_argument("--smooth", type=int, default=31,
                        help="Moving-average window for the median curve (default: 31).")
    parser.add_argument("--no-song-list", dest="song_list",
                        action="store_false", default=True,
                        help="Hide the interactive song list panel.")
    args = parser.parse_args()

    main(
        folder         = args.folder,
        output_png     = args.output,
        frame_ms       = args.frame_ms,
        n_points       = args.points,
        smooth_window  = args.smooth,
        show_song_list = args.song_list,
    )
