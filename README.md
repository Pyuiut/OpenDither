# OpenDither

OpenDither is a premium desktop lab for pixel-perfect dithering, color grading, and experimental retro FX. Built with PyQt6 and powered by OpenCV + NumPy, it combines a production-grade processing pipeline with an opinionated creative workspace.

## ✨ Highlights
- **50+ curated dithering algorithms** (error diffusion, ordered, halftone, pattern, modulation) with a true `None` bypass for color workflows.
- **Advanced lens flares** with tint mixing, shape presets, ghost trails, and placement controls.
- **Glitch lab** featuring RGB split, block shift, and analog ripple modes with frequency/shift sliders.
- **Gradient maps, palette library, and LUT-style curves** for deep color play.
- **Non-destructive adjustments** (tone, detail, transform) and instant reset back to neutral.
- **Preset system + batch processor** for rapid experimentation across entire folders.

## 🖥️ UI Experience
The PyQt6 interface is split into focused workspaces:
1. **Adjust** – exposure, contrast, vibrance, tones, B&W mix, detail, transforms.
2. **Color** – palette browser (with category filter), gradient mapper, temperature/tint, hue shift.
3. **Effects** – blur/glow, flares, glitches, grain, vignette.
4. **Curves** – histogram, RGB & channel curves, full levels toolset.
5. **Compare** – split-view before/after overlay.
6. **Export** – format pickers, DPI/dimension settings, metadata toggles.

Every control is debounced for live preview responsiveness. An “Update Comparison” button refreshes the split view, and `_reset_all` returns the full UI to pristine defaults (including algorithm = `"None"`).

## 🚀 Getting Started
```bash
# 1. Create / activate a virtualenv (recommended)
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python run.py
```

### Requirements
- Python 3.9+
- PyQt6 6.4+
- OpenCV (headless build) 4.5+
- NumPy, SciPy, Numba, Pillow, ImageIO (+ ffmpeg plugin)

> **Tip:** On macOS/Linux replace the activation command with `source .venv/bin/activate`.

## 🧠 Architecture
```
OpenDither/
├─ opendither/
│  ├─ core/              # Palettes, algorithms, presets
│  ├─ processing/        # Pipeline, dithering, effects, workers
│  ├─ ui/                # PyQt6 windows, widgets, styles
│  └─ main.py            # App bootstrap
├─ run.py                # Entry point
└─ requirements.txt      # Dependencies
```
Key pieces:
- `processing/pipeline.py` orchestrates adjustments → grading → curves → dithering → effects.
- `ui/main_window.py` wires the parameter model to sliders/combos and dispatches threaded processing via `processing.worker`.
- `ui/widgets/*` hosts reusable components (CurvesEditor, HistogramWidget, SplitView).

## 🧪 Feature Matrix
| Area | Capabilities |
| --- | --- |
| Dithering | 15+ error diffusion, Bayer sizes, halftone, patterns, glitch/pixelate/VHS, with palette mapping |
| Color | Palette library with category filtering, gradient map presets, hue/temperature/tint sliders |
| Effects | Blur, glow (multi-kernel), lens flares (ghosts, distribution, tint HSV), glitch lab, grain, vignette |
| Tone | Highlights/Shadows/Whites/Blacks, vibrance/saturation, B&W mix per channel |
| Detail | Sharpness, clarity, dehaze |
| Transform | Scale %, rotation |
| Export | PNG/JPEG/TIFF/BMP/WebP/PDF/SVG (customizable quality, DPI, dimensions) |

## 🛣️ Roadmap
- Video processing surface (hooks already stubbed in `_process_video`).
- Custom palette editor + drag-to-reorder swatches.
- Keyframe-based animation export.
- Cloud preset sync.

## 🤝 Contributing
1. Fork & clone the repo.
2. Create a branch (`git checkout -b feature/my-improvement`).
3. Install deps + run `python run.py` to QA changes.
4. Submit a PR – keep commits tidy and describe UX decisions.

### Coding Notes
- Prefer `_default_parameters` for resets.
- Use `_schedule_update` instead of direct `_do_update` when responding to UI changes.
- Add new effects inside `_apply_effects` to piggyback caching/progress reporting.

## 📄 License
MIT (add `LICENSE` file as needed for distribution).

## 📸 Screenshots
_(Add UI shots or GIFs here to showcase workflows.)_
