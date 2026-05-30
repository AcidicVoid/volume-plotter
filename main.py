"""
audio_volume_graph.py
---------------------
Reads all audio files from a folder and plots their volume over time.

For each file:
  - Volume is computed in short frames and converted to dBFS
  - Time axis is normalised to 0–100% of song duration
  - Individual tracks are drawn in semi-transparent blue
  - A median curve (computed across all tracks) is drawn on top in yellow
  - A red horizontal line marks 0 dB
  - Y-axis is fixed

Dependencies:
    pip install numpy scipy matplotlib soundfile librosa
    (librosa handles MP3/FLAC/OGG via audioread; soundfile handles WAV/FLAC directly)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Audio loading
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif", ".m4a"}


def find_audio_files(folder: str) -> list[str]:
    """Return sorted list of audio file paths found in *folder*."""
    paths = []
    for name in sorted(os.listdir(folder)):
        ext = os.path.splitext(name)[1].lower()
        if ext in SUPPORTED_EXTENSIONS:
            paths.append(os.path.join(folder, name))
    return paths


def load_audio_mono(path: str) -> tuple[np.ndarray, int]:
    """
    Load an audio file and return (samples_mono, sample_rate).
    Tries soundfile first (fast, lossless formats), falls back to librosa
    (handles MP3, M4A, etc. via ffmpeg/audioread).
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


# ---------------------------------------------------------------------------
# Volume computation
# ---------------------------------------------------------------------------

DB_FLOOR = -80.0   # silence floor before clipping to display range


def compute_rms_db(samples: np.ndarray, sr: int, frame_ms: int = 50) -> np.ndarray:
    """
    Split *samples* into non-overlapping frames of *frame_ms* milliseconds
    and compute the RMS amplitude of each frame in dBFS.

    Returns a 1-D array of dB values (float32).
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    # Trim to a multiple of frame_len so reshape works cleanly
    n_frames = len(samples) // frame_len
    trimmed = samples[: n_frames * frame_len]
    frames = trimmed.reshape(n_frames, frame_len)

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    # Convert to dBFS; guard against log(0)
    rms = np.where(rms < 1e-10, 1e-10, rms)
    db = 20.0 * np.log10(rms)
    return db.astype(np.float32)


def normalise_to_percent(db_array: np.ndarray, n_points: int = 1000) -> np.ndarray:
    """
    Resample *db_array* to exactly *n_points* via linear interpolation so
    all tracks share the same 0–100 % time axis.
    """
    x_src = np.linspace(0.0, 100.0, len(db_array))
    x_dst = np.linspace(0.0, 100.0, n_points)
    return np.interp(x_dst, x_src, db_array).astype(np.float32)


# ---------------------------------------------------------------------------
# Median curve
# ---------------------------------------------------------------------------

def compute_median_curve(tracks: list[np.ndarray]) -> np.ndarray:
    """
    Stack all resampled track arrays and return the per-point median.
    *tracks* must all have the same length (enforced by normalise_to_percent).
    """
    matrix = np.stack(tracks, axis=0)   # shape: (n_tracks, n_points)
    return np.median(matrix, axis=0)


def smooth_curve(curve: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Apply a simple uniform (box) moving-average to *curve* for a smoother line.
    *window* is the kernel width in samples; it is auto-clamped to be odd and
    no larger than the data.
    """
    window = min(window, len(curve))
    window = window if window % 2 == 1 else window - 1
    kernel = np.ones(window) / window
    return np.convolve(curve, kernel, mode="same")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

DB_MIN, DB_MAX = -35.0, 5.0   # fixed display range

BLUE_ALPHA   = 0.5  # individual track transparency
BLUE_WIDTH   = 0.8
YELLOW_WIDTH = 2.5


def plot_volume_overlay(tracks: list[np.ndarray],
                        labels: list[str],
                        median: np.ndarray,
                        output_path: str | None = None) -> None:
    """
    Draw all tracks in blue, the median in yellow, and a 0 dB red line.

    Parameters
    ----------
    tracks      : list of resampled dB arrays (all same length)
    labels      : file-name labels for the legend (one per track)
    median      : smoothed median curve
    output_path : if given, save the figure there; else show interactively
    """
    x = np.linspace(0.0, 100.0, len(tracks[0]))

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#16213e")

    # --- individual tracks (blue, semi-transparent) -----------------------
    for db in tracks:
        clipped = np.clip(db, DB_MIN, DB_MAX)
        ax.plot(x, clipped, color="#4fc3f7", alpha=BLUE_ALPHA,
                linewidth=BLUE_WIDTH, zorder=2)

    # --- 0 dB reference line (red) ----------------------------------------
    ax.axhline(0.0, color="#ef5350", linewidth=1.4, linestyle="--",
               zorder=3, label="0 dB")

    # --- median curve (yellow) drawn ON TOP of blue tracks ----------------
    ax.plot(x, np.clip(median, DB_MIN, DB_MAX),
            color="#ffeb3b", linewidth=YELLOW_WIDTH,
            zorder=4, label="Median")

    # --- axes & labels ----------------------------------------------------
    ax.set_xlim(0, 100)
    ax.set_ylim(DB_MIN, DB_MAX)
    ax.set_xlabel("Song time (%)", color="white", fontsize=11)
    ax.set_ylabel("Volume (dBFS)", color="white", fontsize=11)
    ax.set_title(
        f"Volume Overlay for {len(tracks)} track(s)",
        color="white", fontsize=13, pad=12
    )

    ax.tick_params(colors="white")
    ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(1))
    ax.grid(which="major", color="#ffffff22", linewidth=0.6)
    ax.grid(which="minor", color="#ffffff0d", linewidth=0.3)

    for spine in ax.spines.values():
        spine.set_edgecolor("#ffffff33")

    # Dummy blue patch for the legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#4fc3f7", alpha=0.7, linewidth=1.5,
               label=f"Tracks ({len(tracks)})"),
        Line2D([0], [0], color="#ffeb3b", linewidth=2.5, label="Median"),
        Line2D([0], [0], color="#ef5350", linewidth=1.4, linestyle="--",
               label="0 dB"),
    ]
    ax.legend(handles=legend_handles, facecolor="#0f3460", edgecolor="#ffffff33",
              labelcolor="white", fontsize=9, loc="upper right")

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
    folder       : directory containing audio files
    output_png   : optional path to save PNG; if None, opens interactive window
    frame_ms     : RMS frame length in milliseconds
    n_points     : number of resampled time points (shared x-axis resolution)
    smooth_window: moving-average window size for the median curve (samples)
    """
    paths = find_audio_files(folder)
    if not paths:
        print(f"No supported audio files found in '{folder}'.")
        sys.exit(1)

    print(f"Found {len(paths)} audio file(s) in '{folder}'.\n")

    resampled_tracks: list[np.ndarray] = []
    labels: list[str] = []

    for path in paths:
        name = os.path.basename(path)
        print(f"  Loading  {name} …", end=" ", flush=True)
        try:
            samples, sr = load_audio_mono(path)
            db = compute_rms_db(samples, sr, frame_ms=frame_ms)
            resampled = normalise_to_percent(db, n_points=n_points)
            resampled_tracks.append(resampled)
            labels.append(name)
            print(f"OK  ({sr} Hz, {len(samples)/sr:.1f} s)")
        except Exception as exc:
            print(f"SKIP — {exc}")

    if not resampled_tracks:
        print("No tracks could be loaded.")
        sys.exit(1)

    print(f"\nComputing median across {len(resampled_tracks)} track(s) …")
    raw_median = compute_median_curve(resampled_tracks)
    smooth_med = smooth_curve(raw_median, window=smooth_window)

    print("Rendering graph …")
    plot_volume_overlay(resampled_tracks, labels, smooth_med,
                        output_path=output_png)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Overlay volume graphs for all audio files in a folder."
    )
    parser.add_argument(
        "folder",
        help="Path to the folder containing audio files."
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save graph to this PNG path instead of showing interactively."
    )
    parser.add_argument(
        "--frame-ms", type=int, default=50,
        help="RMS frame length in milliseconds (default: 50)."
    )
    parser.add_argument(
        "--points", type=int, default=1000,
        help="Number of time points on the shared x-axis (default: 1000)."
    )
    parser.add_argument(
        "--smooth", type=int, default=31,
        help="Moving-average window for the median curve (default: 31)."
    )
    args = parser.parse_args()

    main(
        folder=args.folder,
        output_png=args.output,
        frame_ms=args.frame_ms,
        n_points=args.points,
        smooth_window=args.smooth,
    )
