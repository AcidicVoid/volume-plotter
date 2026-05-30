"""
audio_volume_graph.py
---------------------
Reads all audio files from a folder and plots their volume over time.

For each file:
  - Volume is computed in short frames and converted to dBFS.
  - Time axis is normalised to 0–100% of song duration.
  - Individual tracks are drawn in semi-transparent blue.
  - A smoothed median curve is drawn on top in yellow.
  - A smoothed mean curve is drawn on top in purple.
  - A red horizontal line marks 0 dB.
  - Y-axis is fixed to -40 dB … +5 dB.

Audio files are loaded in parallel using a thread pool for faster I/O.

Dependencies:
    pip install numpy matplotlib soundfile librosa
    (librosa handles MP3/FLAC/OGG via audioread; soundfile handles WAV/FLAC directly)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from concurrent.futures import ThreadPoolExecutor, as_completed
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a"}


def find_audio_files(folder: str) -> list[str]:
    """Return a sorted list of audio file paths found in *folder*."""
    paths = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder, name))
    return paths


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (samples_mono, sample_rate).
    Tries soundfile first (fast, handles WAV/FLAC/OGG natively), then falls
    back to librosa which uses ffmpeg/audioread for MP3, M4A, etc.
    """
    try:
        import soundfile as sf
        data, sr = sf.read(path, always_2d=True)
        mono = data.mean(axis=1)          # mix down to mono
        return mono.astype(np.float32), sr
    except Exception:
        import librosa
        mono, sr = librosa.load(path, sr=None, mono=True)
        return mono, sr


def process_file(path: str, frame_ms: int, n_points: int) -> tuple[str, np.ndarray] | None:
    """
    Load one audio file, compute its dB curve, and resample to *n_points*.
    Returns (filename, resampled_db) on success, or None on failure.

    This function is designed to be called from a thread pool — each file is
    fully independent, so parallelism is safe and effective here.
    """
    name = os.path.basename(path)
    try:
        samples, sr = load_audio_mono(path)
        db = compute_rms_db(samples, sr, frame_ms=frame_ms)
        resampled = normalise_to_percent(db, n_points=n_points)
        duration = len(samples) / sr
        print(f"  OK  {name}  ({sr} Hz, {duration:.1f} s)")
        return name, resampled
    except Exception as exc:
        print(f"  SKIP  {name}  — {exc}")
        return None


def load_tracks_parallel(paths: list[str], frame_ms: int,
                          n_points: int) -> tuple[list[np.ndarray], list[str]]:
    """
    Load and process all audio files concurrently using a thread pool.

    Threads are ideal here: the bottleneck is I/O (disk reads + decoder), not
    CPU, so multiple threads can wait on different files simultaneously with no
    GIL contention on the heavy NumPy maths. ProcessPoolExecutor would add
    serialisation overhead that outweighs any benefit for typical folder sizes.
    """
    resampled_tracks: list[np.ndarray] = []
    labels: list[str] = []

    # Cap workers at the number of files; no point spawning more than needed.
    max_workers = min(len(paths), os.cpu_count() or 4)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(process_file, p, frame_ms, n_points): p
            for p in paths
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                name, resampled = result
                labels.append(name)
                resampled_tracks.append(resampled)

    return resampled_tracks, labels


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------

def compute_rms_db(samples: np.ndarray, sr: int, frame_ms: int = 50) -> np.ndarray:
    """
    Split *samples* into non-overlapping frames of *frame_ms* milliseconds
    and compute the RMS amplitude of each frame in dBFS.

    Returns a 1-D float32 array of dB values.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    # Trim to a multiple of frame_len so reshape works cleanly
    n_frames = len(samples) // frame_len
    trimmed = samples[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Guard against log(0) before converting to dB
    rms = np.where(rms < 1e-10, 1e-10, rms)
    db = 20.0 * np.log10(rms)
    return db.astype(np.float32)


def normalise_to_percent(db_array: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """
    Resample *db_array* to exactly *n_points* via linear interpolation so
    all tracks share the same 0–100 % time axis regardless of duration.
    """
    x_src = np.linspace(0.0, 100.0, len(db_array))
    x_dst = np.linspace(0.0, 100.0, n_points)
    return np.interp(x_dst, x_src, db_array).astype(np.float32)


# ---------------------------------------------------------------------------
# Statistical curves
# ---------------------------------------------------------------------------

def compute_median_curve(tracks: list[np.ndarray]) -> np.ndarray:
    """Per-point median across all resampled tracks (shape: n_points)."""
    matrix = np.stack(tracks, axis=0)   # (n_tracks, n_points)
    return np.median(matrix, axis=0)


def compute_mean_curve(tracks: list[np.ndarray]) -> np.ndarray:
    """Per-point arithmetic mean across all resampled tracks (shape: n_points)."""
    matrix = np.stack(tracks, axis=0)   # (n_tracks, n_points)
    return np.mean(matrix, axis=0)


def smooth_curve(curve: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Uniform (box) moving-average smoother.
    *window* is auto-clamped to be odd and no wider than the data.
    """
    window = min(window, len(curve))
    window = window if window % 2 == 1 else window - 1
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DB_MIN, DB_MAX = -40.0, 5.0   # fixed display range

BLUE_ALPHA   = 0.25             # individual track transparency
BLUE_WIDTH   = 0.6
YELLOW_WIDTH = 2.5
PURPLE_WIDTH = 2.5


def plot_volume_overlay(tracks: list[np.ndarray],
                        labels: list[str],
                        median: np.ndarray,
                        mean: np.ndarray,
                        output_path: str | None = None) -> None:
    """
    Draw all tracks in blue, median in yellow, mean in purple, 0 dB in red.

    Z-order (front to back):
        5 — mean (purple)
        4 — median (yellow)
        3 — 0 dB line (red)
        2 — individual tracks (blue)

    Parameters
    ----------
    tracks      : resampled dB arrays (all same length)
    labels      : file-name labels (one per track, for potential future use)
    median      : smoothed median curve
    mean        : raw (unsmoothed) per-point mean curve
    output_path : save to PNG here; if None, open interactive window
    """
    x = np.linspace(0.0, 100.0, len(tracks[0]))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # --- individual tracks (blue, semi-transparent) -----------------------
    for db in tracks:
        ax.plot(x, np.clip(db, DB_MIN, DB_MAX),
                color="#4fc3f7", alpha=BLUE_ALPHA,
                linewidth=BLUE_WIDTH, zorder=2)

    # --- 0 dB reference line (red, dashed) --------------------------------
    ax.axhline(0.0, color="#ef5350", linewidth=1.4, linestyle="--", zorder=3)

    # --- median curve (yellow) --------------------------------------------
    ax.plot(x, np.clip(median, DB_MIN, DB_MAX),
            color="#ffeb3b", linewidth=YELLOW_WIDTH, zorder=4)

    # --- mean curve (purple) — drawn on top of median ---------------------
    ax.plot(x, np.clip(mean, DB_MIN, DB_MAX),
            color="#ce93d8", linewidth=PURPLE_WIDTH, zorder=5)

    # --- axes formatting --------------------------------------------------
    ax.set_xlim(0, 100)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.set_xlabel("Song time (%)", color="white", fontsize=11)
    ax.set_ylabel("Volume (dBFS)", color="white", fontsize=11)
    ax.set_title(
        f"Volume overlay — {len(tracks)} track(s)",
        color="white", fontsize=13, pad=12
    )

    ax.tick_params(colors="white")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which="major", color="#ffffff22", linewidth=0.6)
    ax.grid(which="minor", color="#ffffff0d", linewidth=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff33")

    # --- legend -----------------------------------------------------------
    legend_handles = [
        Line2D([0], [0], color="#4fc3f7", alpha=0.7, linewidth=1.5,
               label=f"Tracks ({len(tracks)})"),
        Line2D([0], [0], color="#ffeb3b", linewidth=2.5, label="Median"),
        Line2D([0], [0], color="#ce93d8", linewidth=2.5, label="Mean"),
        Line2D([0], [0], color="#ef5350", linewidth=1.4, linestyle="--",
               label="0 dB"),
    ]
    ax.legend(handles=legend_handles, facecolor="#0f3460",
              edgecolor="#ffffff33", labelcolor="white",
              fontsize=9, loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved → {output_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(folder: str, output_png: str | None = None,
         frame_ms: int = 50, n_points: int = 1000,
         smooth_window: int = 31) -> None:
    """
    Entry point.

    Parameters
    ----------
    folder        : directory containing audio files
    output_png    : save PNG here; if None, opens an interactive window
    frame_ms      : RMS frame length in milliseconds
    n_points      : shared x-axis resolution (time points per track)
    smooth_window : moving-average kernel width for median and mean curves
    """
    paths = find_audio_files(folder)
    if not paths:
        print(f"No supported audio files found in '{folder}'.")
        sys.exit(1)

    print(f"Found {len(paths)} audio file(s) in '{folder}'. Loading in parallel…\n")

    resampled_tracks, labels = load_tracks_parallel(paths, frame_ms, n_points)

    if not resampled_tracks:
        print("No tracks could be loaded.")
        sys.exit(1)

    print(f"\nComputing statistics across {len(resampled_tracks)} track(s) …")
    smooth_med  = smooth_curve(compute_median_curve(resampled_tracks), window=smooth_window)
    raw_mean    = compute_mean_curve(resampled_tracks)   # no smoothing — plain per-point average

    print("Rendering graph …")
    plot_volume_overlay(resampled_tracks, labels, smooth_med, raw_mean,
                        output_path=output_png)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overlay volume graphs for all audio files in a folder."
    )
    parser.add_argument("folder",
                        help="Path to the folder containing audio files.")
    parser.add_argument("--output", "-o", default=None,
                        help="Save graph to this PNG path instead of showing interactively.")
    parser.add_argument("--frame-ms", type=int, default=50,
                        help="RMS frame length in milliseconds (default: 50).")
    parser.add_argument("--points", type=int, default=1000,
                        help="Number of time points on the shared x-axis (default: 1000).")
    parser.add_argument("--smooth", type=int, default=31,
                        help="Moving-average window for median/mean curves (default: 31).")
    args = parser.parse_args()

    main(
        folder=args.folder,
        output_png=args.output,
        frame_ms=args.frame_ms,
        n_points=args.points,
        smooth_window=args.smooth,
    )
