# Real-Time Slow Motion using Optical Flow

A real-time webcam demo that creates a **“slow motion zone”**: inside a circular region you see delayed frames played back slower, while the rest of the frame stays live. The slow zone also visualizes motion using **dense optical flow (Farneback)**.

## Features

- Slow-motion playback inside a circular zone using a rolling frame buffer
- Dense optical flow arrows inside the zone (direction-colored)
- Optional Hough circle tracking to move the zone with a physical circular object
- On-screen HUD and keyboard controls

## Requirements

- Python 3
- `numpy`
- `opencv-python`

## Setup

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install numpy opencv-python
```

## Run

```bash
./.venv/bin/python slow.py
```

## Controls

- `Q` quit
- `R` reset buffer
- `+` / `-` resize zone
- `M` toggle Hough tracking mode
- `1` `2` `3` `4` change slow factor (speed)
