"""
Extract ground-control points (PDF pixel position <-> real lat/lon) directly
from PORC_COMP.pdf's own printed coordinate text next to each waypoint/navaid.

The chart prints degree-symbol glyphs using a font PyMuPDF can't decode (shows
as U+FFFD replacement char), but the surrounding digits extract fine, so we
treat any single non-digit character in that position as the degree symbol.

Usage: python extract_control_points.py > control_points.json
"""
import json
import re
import sys
import fitz

PDF_PATH = r"C:\Users\conno\Desktop\PORC_COMP.pdf"

# Loose regexes: one non-digit char stands in for the garbled degree symbol.
LAT_RE = re.compile(r"^([NS])(\d{2}).(\d{2}\.\d)'$")
LON_RE = re.compile(r"^([EW])(\d{2,3}).(\d{2}\.\d)'$")
LABEL_RE = re.compile(r"^[A-Z]{2,6}$")

MAX_PAIR_DIST = 25    # pts, max distance between a lat line and its lon line
MAX_LABEL_DIST = 25   # pts, max distance from coord pair to its identifier
MIN_DIR_DOT = 0.999   # required cosine similarity between text rotations to
                       # count as "same rotated label block" -- distance
                       # alone isn't enough in dense areas: many different
                       # waypoints' labels sit within a few points of each
                       # other, but each is rotated to follow its own route
                       # bearing, so their `dir` vectors reliably differ.


def dms_to_decimal(sign_char, deg, minutes, positive_char):
    val = int(deg) + float(minutes) / 60.0
    return val if sign_char == positive_char else -val


def bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def dist(a, b):
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def extract_lines(page, clip):
    text_dict = page.get_text("dict", clip=clip)
    lines = []
    for block in text_dict["blocks"]:
        for line in block.get("lines", []):
            full = "".join(s["text"] for s in line["spans"])
            lines.append((full.strip(), line["bbox"], line.get("dir", (1.0, 0.0))))
    return lines


def dir_dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def find_control_points(page, clip):
    lines = extract_lines(page, clip)

    lat_points, lon_points, labels = [], [], []
    for text, bbox, d in lines:
        m = LAT_RE.match(text)
        if m:
            lat = dms_to_decimal(m.group(1), m.group(2), m.group(3), "N")
            lat_points.append((lat, bbox_center(bbox), d))
            continue
        m = LON_RE.match(text)
        if m:
            lon = dms_to_decimal(m.group(1), m.group(2), m.group(3), "E")
            lon_points.append((lon, bbox_center(bbox), d))
            continue
        if LABEL_RE.match(text):
            labels.append((text, bbox_center(bbox), d))

    used_lon = set()
    pairs = []
    for lat, lat_c, lat_dir in lat_points:
        best_j, best_d = None, MAX_PAIR_DIST
        for j, (lon, lon_c, lon_dir) in enumerate(lon_points):
            if j in used_lon:
                continue
            if abs(dir_dot(lat_dir, lon_dir)) < MIN_DIR_DOT:
                continue
            d = dist(lat_c, lon_c)
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            lon, lon_c, lon_dir = lon_points[best_j]
            used_lon.add(best_j)
            px = (lat_c[0] + lon_c[0]) / 2
            py = (lat_c[1] + lon_c[1]) / 2
            pairs.append({"lat": lat, "lon": lon, "px": px, "py": py, "dir": lat_dir})

    # The identifier label sits much closer to the true point/vertex symbol
    # than the lat/lon coordinate text does -- the whole "NAME / lat / lon"
    # block is rotated to follow the route line and extends well past the
    # actual point, so the lat/lon text's own position is a poor anchor.
    # Require the label's rotation to match the coordinate pair's rotation
    # (not just proximity) -- dense areas have many different waypoints'
    # labels within a few points of each other, each following its own
    # route bearing, so `dir` reliably tells them apart where distance
    # alone does not.
    good_pairs = []
    for pt in pairs:
        pc = (pt["px"], pt["py"])
        best_label, best_lc, best_d = None, None, MAX_LABEL_DIST
        for text, lc, l_dir in labels:
            if abs(dir_dot(pt["dir"], l_dir)) < MIN_DIR_DOT:
                continue
            d = dist(pc, lc)
            if d < best_d:
                best_label, best_lc, best_d = text, lc, d
        if best_label is not None:
            pt["label"] = best_label
            pt["px"], pt["py"] = best_lc
            pt["label_dist"] = best_d
            del pt["dir"]
            good_pairs.append(pt)
        # drop points with no rotation-matched label at all -- without a
        # label anchor we're back to the less-accurate coordinate-text
        # midpoint, which isn't worth keeping now that we know better.

    return good_pairs


def main():
    doc = fitz.open(PDF_PATH)
    page = doc[0]
    w, h = page.rect.width, page.rect.height

    # Region argument: "x0frac,y0frac,x1frac,y1frac" of page width/height,
    # or "full" for the whole page. Defaults to the Guam validation crop.
    if len(sys.argv) > 1 and sys.argv[1] != "full":
        x0f, y0f, x1f, y1f = map(float, sys.argv[1].split(","))
        clip = fitz.Rect(x0f * w, y0f * h, x1f * w, y1f * h)
    elif len(sys.argv) > 1 and sys.argv[1] == "full":
        clip = page.rect
    else:
        cx, cy = 0.38 * w, 0.29 * h
        clip = fitz.Rect(cx - 0.12 * w, cy - 0.10 * h, cx + 0.12 * w, cy + 0.10 * h)

    points = find_control_points(page, clip)
    print(json.dumps({"page_width": w, "page_height": h, "points": points}, indent=2))


if __name__ == "__main__":
    main()
