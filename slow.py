"""
╔══════════════════════════════════════════════════════════════╗
║           SLOW MOTION ZONE DETECTOR                         ║
║           Classical Computer Vision Assignment              ║
╠══════════════════════════════════════════════════════════════╣
║  Techniques Used:                                           ║
║    1. Dense Optical Flow (Farneback) — motion vectors       ║
║       inside the slow zone, colored by direction            ║
║    2. Hough Circle Detection — detects a held-up circle     ║
║       to let you MOVE the zone around the scene             ║
║    3. Rolling Frame Buffer — the engine behind slow motion  ║
╠══════════════════════════════════════════════════════════════╣
║  Controls:                                                  ║
║    Q       → Quit                                           ║
║    R       → Reset buffer                                   ║
║    + / -   → Resize the slow zone                           ║
║    M       → Toggle Hough tracking mode (move zone with     ║
║              a circular object like a bowl or ring)         ║
║    1       → 1/2x speed (subtle)                            ║
║    2       → 1/4x speed (default)                           ║
║    3       → 1/6x speed (dramatic)                          ║
║    4       → 1/10x speed (extreme)                          ║
╚══════════════════════════════════════════════════════════════╝

HOW IT WORKS — THE CORE IDEA:
  Every frame is written into a circular buffer.
  The slow zone reads from that buffer at a reduced rate:
  for every SLOW_FACTOR real frames captured, only 1 buffer
  step is taken. The outside always shows the current live frame.
  Both regions are composited at every tick — making it appear
  that one region of the screen is stuck in slow time.
"""

import cv2
import numpy as np
from collections import deque

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
BUFFER_SIZE  = 150      # Rolling frame buffer depth (~5s at 30fps)
SLOW_FACTOR  = 4        # Default: show 1 frame per SLOW_FACTOR captured
ZONE_RADIUS  = 160      # Default slow zone radius (pixels)
GLOW_CYAN    = (200, 255, 0)   # BGR — neon cyan glow


# ──────────────────────────────────────────────────────────────
# VISUAL HELPERS
# ──────────────────────────────────────────────────────────────

def draw_glow_circle(img, center, radius, color_bgr, tick):
    """
    Draw a pulsing neon halo around the slow zone boundary.
    Renders several semi-transparent rings of decreasing opacity
    to simulate a glowing edge effect.
    """
    pulse = 0.55 + 0.45 * np.sin(tick * 0.09)

    # Outer soft rings (glow halo)
    for i in range(7, 0, -1):
        alpha = pulse * (1.0 - i / 8.0)
        c = tuple(int(ch * alpha) for ch in color_bgr)
        cv2.circle(img, center, radius + i * 3, c, 1, cv2.LINE_AA)

    # Solid inner ring
    cv2.circle(img, center, radius, color_bgr, 2, cv2.LINE_AA)

    # Small ticks around the ring (like a portal)
    for angle_deg in range(0, 360, 30):
        angle_rad = np.deg2rad(angle_deg)
        outer_x = int(center[0] + (radius + 8) * np.cos(angle_rad))
        outer_y = int(center[1] + (radius + 8) * np.sin(angle_rad))
        inner_x = int(center[0] + (radius + 2) * np.cos(angle_rad))
        inner_y = int(center[1] + (radius + 2) * np.sin(angle_rad))
        cv2.line(img, (inner_x, inner_y), (outer_x, outer_y), color_bgr, 1, cv2.LINE_AA)


def draw_optical_flow_arrows(canvas, flow, roi_offset, mask_roi, step=20):
    """
    Draw motion arrows from Dense Optical Flow inside the slow zone.
    Arrow color encodes direction (HSV hue = angle).
    Arrow length encodes speed (magnitude).
    Only draws arrows where motion exceeds a minimum threshold.
    """
    ox, oy = roi_offset
    roi_h, roi_w = flow.shape[:2]

    for y in range(step, roi_h - step, step):
        for x in range(step, roi_w - step, step):
            # Skip pixels outside the circular mask
            if mask_roi[y, x] == 0:
                continue

            fx, fy = flow[y, x]
            magnitude = np.hypot(fx, fy)

            # Only draw meaningful motion
            if magnitude < 1.2:
                continue

            # Source and target in full-frame coordinates
            sx, sy = x + ox, y + oy
            ex = int(np.clip(sx + fx * 4, 0, canvas.shape[1] - 1))
            ey = int(np.clip(sy + fy * 4, 0, canvas.shape[0] - 1))

            # Encode direction as hue
            hue = int((np.arctan2(fy, fx) + np.pi) / (2 * np.pi) * 179)
            hsv_pixel = np.uint8([[[hue, 230, 255]]])
            bgr = cv2.cvtColor(hsv_pixel, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

            cv2.arrowedLine(canvas, (sx, sy), (ex, ey),
                            color, 1, cv2.LINE_AA, tipLength=0.35)


def draw_hud(img, zone_cx, zone_cy, zone_r, slow_factor,
             buf_filled, buf_total, hough_mode, W, H, tick):
    """
    Draw all on-screen overlay elements:
      - Speed label inside the zone
      - 'LIVE' label outside
      - Buffer progress bar (bottom-left)
      - Mode & controls text
    """
    # ── Speed label inside zone ──────────────────────────────
    speed_str = f"1/{slow_factor}x  SLOW"
    label_x = zone_cx - 55
    label_y = zone_cy - zone_r + 30
    cv2.putText(img, speed_str, (label_x, label_y),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (0, 255, 200), 2, cv2.LINE_AA)

    # ── LIVE label — pulsing dot outside zone ─────────────────
    live_pulse = int(200 + 55 * abs(np.sin(tick * 0.07)))
    cv2.circle(img, (W - 90, 28), 7, (0, 0, live_pulse), -1, cv2.LINE_AA)
    cv2.putText(img, "LIVE", (W - 78, 34),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (0, 0, live_pulse), 1, cv2.LINE_AA)

    # ── Buffer progress bar (bottom-left) ────────────────────
    bx, by, bw, bh = 20, H - 36, 210, 13
    buf_frac = buf_filled / max(buf_total, 1)
    cv2.rectangle(img, (bx - 1, by - 1), (bx + bw + 1, by + bh + 1), (60, 60, 60), 1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (20, 20, 20), -1)
    cv2.rectangle(img, (bx, by), (bx + int(bw * buf_frac), by + bh), (0, 200, 120), -1)
    cv2.putText(img, f"Buffer  {int(buf_frac * 100)}%", (bx, by - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

    # ── Mode indicator (top-left) ─────────────────────────────
    mode_str = "HOUGH TRACK MODE" if hough_mode else "FIXED ZONE MODE"
    mode_col = (0, 220, 255) if hough_mode else (180, 180, 180)
    cv2.putText(img, mode_str, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, mode_col, 1, cv2.LINE_AA)

    # ── Controls reminder (bottom-right) ──────────────────────
    controls = "Q:Quit  R:Reset  +/-:Size  M:Hough  1-4:Speed"
    cv2.putText(img, controls, (W - 470, H - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.37, (110, 110, 110), 1)


# ──────────────────────────────────────────────────────────────
# HOUGH CIRCLE DETECTION
# ──────────────────────────────────────────────────────────────

def detect_hough_zone(gray_frame, expected_r, tolerance=50):
    """
    Use HoughCircles to detect a circle of roughly expected_r
    in the current grayscale frame.

    This is used in Hough Tracking Mode: hold a hula hoop,
    a bowl, or a large printed circle up to the camera —
    the slow zone will snap to it and follow it.

    Returns (cx, cy, r) if found, else None.
    """
    # Blur to suppress noise before Hough
    blurred = cv2.GaussianBlur(gray_frame, (9, 9), 2)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=80,
        param1=80,   # Canny high threshold inside HoughCircles
        param2=38,   # Accumulator threshold — lower = more detections
        minRadius=max(40, expected_r - tolerance),
        maxRadius=expected_r + tolerance
    )

    if circles is not None:
        # Take the strongest (first) detected circle
        c = np.round(circles[0, 0]).astype(int)
        return int(c[0]), int(c[1]), int(c[2])

    return None


# ──────────────────────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────────────────────

def main():
    # Open default webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Verify camera opened
    ret, first = cap.read()
    if not ret:
        print("[ERROR] Could not open webcam. Check connection.")
        return

    first = cv2.flip(first, 1)
    H, W = first.shape[:2]
    print(f"[INFO] Resolution: {W}x{H}")

    # ── State variables ────────────────────────────────────────
    zone_cx, zone_cy = W // 2, H // 2   # Zone center
    zone_r   = ZONE_RADIUS               # Zone radius
    slow_factor = SLOW_FACTOR            # Frames per buffer step
    hough_mode  = False                  # Hough zone tracking on/off

    # Rolling buffer — stores the last BUFFER_SIZE frames
    buffer   = [None] * BUFFER_SIZE
    write_pos = 0    # Always points to where next frame will go
    slow_pos  = 0    # Lags behind write_pos; advances at 1/slow_factor rate
    frame_count = 0  # Total frames captured since last reset

    # Optical flow — keep previous grayscale frame of slow zone
    prev_slow_gray = None

    # Tick counter for glow animation
    tick = 0

    print("\n[READY] Slow Motion Zone Detector is running.")
    print("        Press Q to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Frame capture failed.")
            break

        # Flip horizontally — mirror effect (more natural for webcam)
        frame = cv2.flip(frame, 1)
        frame_count += 1
        tick += 1

        # ── STEP 1: Write current frame into rolling buffer ────
        buffer[write_pos] = frame.copy()
        write_pos = (write_pos + 1) % BUFFER_SIZE

        # ── STEP 2: Advance slow-read pointer at reduced rate ──
        # slow_pos moves 1 step forward for every slow_factor frames.
        # This means the slow zone always shows delayed footage
        # at 1/slow_factor playback speed.
        if frame_count % slow_factor == 0:
            slow_pos = (slow_pos + 1) % BUFFER_SIZE

        # Retrieve the slow-zone frame (fallback to current if empty)
        slow_frame = buffer[slow_pos] if buffer[slow_pos] is not None else frame.copy()

        # ── STEP 3: Hough Circle Detection (optional) ─────────
        # When Hough mode is ON, detect a physical circle held by
        # the user and use it to reposition/resize the slow zone.
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if hough_mode:
            detected = detect_hough_zone(gray_frame, zone_r)
            if detected is not None:
                zone_cx, zone_cy, zone_r = detected

        # ── STEP 4: Build circular mask for the slow zone ─────
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (zone_cx, zone_cy), zone_r, 255, -1)

        # ── STEP 5: Dense Optical Flow inside the slow zone ───
        # Compute Farneback flow between the previous and current
        # slow-zone frames. We restrict computation to the zone's
        # bounding box to keep performance acceptable.
        slow_gray = cv2.cvtColor(slow_frame, cv2.COLOR_BGR2GRAY)
        flow_canvas = slow_frame.copy()  # We'll draw arrows on this

        if prev_slow_gray is not None:
            # Bounding box of the slow zone circle
            x1 = max(0, zone_cx - zone_r)
            y1 = max(0, zone_cy - zone_r)
            x2 = min(W, zone_cx + zone_r)
            y2 = min(H, zone_cy + zone_r)

            roi_curr = slow_gray[y1:y2, x1:x2]
            roi_prev = prev_slow_gray[y1:y2, x1:x2]

            # Compute dense optical flow (Farneback algorithm)
            flow = cv2.calcOpticalFlowFarneback(
                roi_prev, roi_curr,
                flow=None,
                pyr_scale=0.5,   # Image pyramid scale
                levels=3,        # Pyramid levels
                winsize=15,      # Averaging window size
                iterations=3,    # Iterations per pyramid level
                poly_n=5,        # Pixel neighborhood size
                poly_sigma=1.2,  # Gaussian std for polynomial expansion
                flags=0
            )

            # Draw direction arrows on the slow frame canvas
            roi_mask = mask[y1:y2, x1:x2]
            draw_optical_flow_arrows(flow_canvas, flow, (x1, y1), roi_mask, step=18)

        prev_slow_gray = slow_gray.copy()

        # ── STEP 6: Composite — slow zone + live feed ─────────
        # Where mask == 255: use slow frame (with flow arrows)
        # Where mask == 0:   use current live frame
        mask_3ch = cv2.merge([mask, mask, mask])
        result = np.where(mask_3ch > 0, flow_canvas, frame)

        # ── STEP 7: Draw glowing zone boundary ────────────────
        draw_glow_circle(result, (zone_cx, zone_cy), zone_r, GLOW_CYAN, tick)

        # ── STEP 8: Draw HUD overlays ─────────────────────────
        buf_filled = sum(1 for b in buffer if b is not None)
        draw_hud(result, zone_cx, zone_cy, zone_r, slow_factor,
                 buf_filled, BUFFER_SIZE, hough_mode, W, H, tick)

        # ── STEP 9: Display ────────────────────────────────────
        cv2.imshow("Slow Motion Zone Detector", result)

        # ── STEP 10: Keyboard controls ─────────────────────────
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("[EXIT] Quitting.")
            break

        elif key == ord('r'):
            # Reset the buffer and all counters
            buffer      = [None] * BUFFER_SIZE
            write_pos   = 0
            slow_pos    = 0
            frame_count = 0
            prev_slow_gray = None
            print("[RESET] Buffer cleared.")

        elif key in (ord('+'), ord('=')):
            zone_r = min(min(W, H) // 2 - 30, zone_r + 15)
            print(f"[ZONE] Radius increased → {zone_r}px")

        elif key == ord('-'):
            zone_r = max(60, zone_r - 15)
            print(f"[ZONE] Radius decreased → {zone_r}px")

        elif key == ord('m'):
            hough_mode = not hough_mode
            state = "ON  (hold a circular object to move zone)" if hough_mode else "OFF (fixed center)"
            print(f"[HOUGH] Tracking mode: {state}")

        elif key == ord('1'):
            slow_factor = 2
            print("[SPEED] 1/2x — subtle slowdown")
        elif key == ord('2'):
            slow_factor = 4
            print("[SPEED] 1/4x — default")
        elif key == ord('3'):
            slow_factor = 6
            print("[SPEED] 1/6x — dramatic")
        elif key == ord('4'):
            slow_factor = 10
            print("[SPEED] 1/10x — extreme")

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()