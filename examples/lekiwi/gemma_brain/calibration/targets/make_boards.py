import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DPI = 600
MM = DPI / 25.4
OUT = str(__import__("pathlib").Path(__file__).resolve().parent)  # next to this script
PAGES = {"letter": (279.4, 215.9), "a4": (297.0, 210.0)}  # landscape, mm
# Each board has its own marker IDs (first_id onward), so several can be in one view.
BOARDS = [  # name, squares x, y, square mm, marker mm, dictionary, first marker id
    ("wrist_charuco_10x7_25mm", 10, 7, 25.0, 18.0, cv2.aruco.DICT_5X5_100, 0),  # ids 0-34
    ("front_charuco_7x5_35mm", 7, 5, 35.0, 26.0, cv2.aruco.DICT_5X5_100, 35),  # ids 35-51
    ("side_charuco_7x5_35mm", 7, 5, 35.0, 26.0, cv2.aruco.DICT_5X5_100, 52),  # ids 52-68
]
try:
    font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", int(3.2 * MM))
except OSError:
    font = ImageFont.load_default()
for name, sx, sy, sq, mk, dic, first in BOARDS:
    d = cv2.aruco.getPredefinedDictionary(dic)
    n = (sx * sy) // 2  # markers on the white squares
    board = cv2.aruco.CharucoBoard((sx, sy), sq / 1000, mk / 1000, d, np.arange(first, first + n))
    bw, bh = int(round(sx * sq * MM)), int(round(sy * sq * MM))
    img = board.generateImage((bw, bh), marginSize=0, borderBits=1)
    for page, (pw, ph) in PAGES.items():
        W, H = int(round(pw * MM)), int(round(ph * MM))
        canvas = Image.new("L", (W, H), 255)
        x0, y0 = (W - bw) // 2, int(round(8 * MM))
        canvas.paste(Image.fromarray(img), (x0, y0))
        dr = ImageDraw.Draw(canvas)
        # 100 mm scale bar under the board, to check the print is 100%
        yb = y0 + bh + int(5 * MM)
        dr.rectangle([x0, yb, x0 + int(round(100 * MM)), yb + int(1.0 * MM)], fill=0)
        for t in (0, 50, 100):
            xt = x0 + int(round(t * MM))
            dr.rectangle([xt, yb - int(2 * MM), xt + int(0.3 * MM), yb + int(1.0 * MM)], fill=0)
        lines = [f"LeKiwi {name.split('_')[0]} board: ChArUco {sx}x{sy} squares, square {sq:.0f} mm, marker {mk:.0f} mm, DICT_5X5_100, ids {first}-{first + n - 1}.",
                 "Print at 100% / actual size (no scaling). The bar is 100 mm: measure it. Glue flat on stiff board."]
        for k, line in enumerate(lines):
            dr.text((x0 + int(105 * MM), yb - int(2.5 * MM) + k * int(4.5 * MM)), line, fill=0, font=font)
        canvas.point(lambda v: 255 if v > 127 else 0).convert("1").save(f"{OUT}/{name}_{page}.pdf", resolution=DPI)
        if page == "letter":
            canvas.convert("RGB").resize((W // 6, H // 6)).save(f"{OUT}/{name}_preview.png")
    print(name, f"board {sx*sq:.0f} x {sy*sq:.0f} mm")
