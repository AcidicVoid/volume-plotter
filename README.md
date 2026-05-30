# Audio Volume Plotter

Visualises the volume of every audio file in a folder as overlapping graphs,
making it easy to spot unusual peaks, inconsistent loudness, or outlier tracks
at a glance.

---

## What it shows

| Element | Colour | Description |
|---|---|---|
| Individual tracks | Blue (faint) | Per-frame RMS volume, all songs overlaid |
| Median | Yellow | Smoothed middle value across all tracks |
| Mean | Purple | Plain per-point average across all tracks |
| 0 dB reference | Red dashed | Clipping threshold |

The X axis is **percentage of song duration** (0–100 %), so tracks of different
lengths all align on the same scale.  
The Y axis is fixed to **−40 dB … +5 dB**.

---

## Requirements

Python 3.10 or newer, plus:

```
pip install numpy matplotlib soundfile librosa
```

| Package | Purpose |
|---|---|
| `numpy` | Array maths and RMS computation |
| `matplotlib` | Rendering the graph |
| `soundfile` | Fast loading of WAV, FLAC, OGG |
| `librosa` | Fallback loader for MP3, M4A, AAC (needs `ffmpeg` on PATH) |

**Tkinter** is also required for the interactive window — it ships with the
standard Python installer on Windows and macOS. On Linux install it with:

```
sudo apt install python3-tk   # Debian/Ubuntu
sudo dnf install python3-tkinter  # Fedora
```

---

## Usage

### Basic — open interactive window

```
python audio_volume_graph.py "C:\Music\MyAlbum"
```

### Save to PNG instead

```
python audio_volume_graph.py "C:\Music\MyAlbum" --output report.png
```

### Hide the song list panel

```
python audio_volume_graph.py "C:\Music\MyAlbum" --no-song-list
```

### All options

```
python audio_volume_graph.py <folder> [options]

Positional:
  folder              Path to the folder containing audio files

Options:
  -o, --output        Save to this PNG file instead of opening interactively
  --frame-ms N        RMS frame size in milliseconds         (default: 50)
  --points N          Time-axis resolution (points per track) (default: 1000)
  --smooth N          Median smoothing window in points       (default: 31)
  --no-song-list      Hide the interactive song list panel
```

---

## Supported formats

`.wav` · `.flac` · `.ogg` · `.mp3` · `.aiff` · `.aif` · `.m4a`

MP3 and M4A require **ffmpeg** to be installed and available on your system
PATH. WAV and FLAC work with no extra tools.

---

## Interactive song list

When running in interactive mode a scrollable panel on the right lists every
loaded track. Click any name to **highlight** it — the corresponding curve
turns bright blue and is brought to the front of the graph. Click again to
un-highlight. Multiple tracks can be highlighted at the same time.

Highlighting is near-instant because it uses **blitting**: the static graph is
cached as a pixel snapshot after the first draw, and only the changed line is
redrawn on top of it on each click.

---

## How volume is calculated

1. The audio file is loaded and mixed down to mono.
2. It is split into short non-overlapping frames (`--frame-ms`, default 50 ms).
3. The RMS amplitude of each frame is computed and converted to dBFS:  
   `dB = 20 × log₁₀(rms)`
4. The resulting curve is resampled to a fixed number of points (`--points`)
   so all tracks share the same 0–100 % time axis.

---

## Performance notes

**Loading** is parallelised with a thread pool. The bottleneck is disk I/O and
decoder wait time, so threads (not processes) give the best speedup — the GIL
is released during those waits, letting all threads make progress at once.
Typical speedup on an SSD with 50+ files is 4–8×.

**Highlighting** uses matplotlib blitting, avoiding a full figure redraw on
every click. With 59 tracks the difference is roughly 400 ms → under 10 ms
per toggle.

---

## Project structure

```
audio_volume_graph.py   Main script (single file, no package required)
README.md               This file
```
