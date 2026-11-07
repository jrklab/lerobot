"""Container markers: single ArUco markers to stick on a container's side, so the front camera
can measure exactly where the container is and which way it faces. Same dictionary as the
calibration boards (DICT_5X5_100), ids after theirs (0-68): 90 and 91 for the shoe box, one
for each long side. 60 mm: fits the 10 cm tall box's side with its white margin, and is
~40 px across in the front camera at 60 cm."""

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
MM = DPI / 25.4
OUT = str(__import__("pathlib").Path(__file__).resolve().parent)  # next to this script
PAGES = {"letter": (215.9, 279.4), "a4": (210.0, 297.0)}  # portrait, mm
MARKERS = [(90, "shoe box, side 1"), (91, "shoe box, side 2")]  # id, what it goes on
MARKER_MM = 60.0
MARGIN_MM = 10.0  # white around the marker: the detector needs a quiet zone

try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(3.2 * MM))
except OSError:
    font = ImageFont.load_default()
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
side = int(round(MARKER_MM * MM))
for page, (pw, ph) in PAGES.items():
    W, H = int(round(pw * MM)), int(round(ph * MM))
    canvas = Image.new("L", (W, H), 255)
    dr = ImageDraw.Draw(canvas)
    x0 = int(round(25 * MM))
    y = int(round(20 * MM))
    for mid, what in MARKERS:
        canvas.paste(Image.fromarray(cv2.aruco.generateImageMarker(d, mid, side, borderBits=1)), (x0, y))
        # Thin cut line MARGIN_MM outside the marker: cut there to keep the white margin.
        m = int(round(MARGIN_MM * MM))
        dr.rectangle([x0 - m, y - m, x0 + side + m, y + side + m], outline=0, width=2)
        dr.text((x0 + side + m + int(5 * MM), y + int(5 * MM)), f"ArUco DICT_5X5_100 id {mid}", fill=0, font=font)
        dr.text((x0 + side + m + int(5 * MM), y + int(10 * MM)), f"{MARKER_MM:.0f} mm, for the {what}", fill=0, font=font)
        y += side + 2 * m + int(15 * MM)
    # 100 mm scale bar, to check the print is 100%
    dr.rectangle([x0, y, x0 + int(round(100 * MM)), y + int(1.0 * MM)], fill=0)
    for t in (0, 50, 100):
        xt = x0 + int(round(t * MM))
        dr.rectangle([xt, y - int(2 * MM), xt + int(0.3 * MM), y + int(1.0 * MM)], fill=0)
    lines = [
        "Print at 100% / actual size (no scaling). The bar is 100 mm: measure it.",
        "Cut on the thin lines (keep the white margin). Stick flat, upright, on the box's long sides, centred.",
        "Measure the black square's side after printing and tell Claude if it isn't 60 mm.",
    ]
    for k, line in enumerate(lines):
        dr.text((x0, y + int(5 * MM) + k * int(5 * MM)), line, fill=0, font=font)
    canvas.point(lambda v: 255 if v > 127 else 0).convert("1").save(f"{OUT}/box_markers_{page}.pdf", resolution=DPI)
    if page == "letter":
        canvas.convert("RGB").resize((W // 6, H // 6)).save(f"{OUT}/box_markers_preview.png")

# The printed markers must read back as their ids.
for mid, _ in MARKERS:
    img = np.pad(cv2.aruco.generateImageMarker(d, mid, 200, borderBits=1), 40, constant_values=255)
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(img)
    assert ids is not None and ids.ravel().tolist() == [mid], (mid, ids)
print("box markers:", [m for m, _ in MARKERS], f"{MARKER_MM:.0f} mm")
