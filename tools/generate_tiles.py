"""
Generate a Leaflet-compatible {z}/{x}/{y}.png XYZ tile pyramid directly from
the rasterized chart, using thin-plate-spline interpolation fit on ground
control points extracted from the chart's own printed coordinates.

Why not gdalwarp -tps + gdal2tiles.py: no GDAL CLI / osgeo bindings are
installed in this environment (only rasterio, which doesn't expose GDAL's
TPS transformer through its simplified reproject() API -- its default GCP
transformer is a low-order global polynomial, which produced a badly
distorted, mostly-empty result over an area this large/curved). Fitting our
own TPS with scipy and sampling it per-tile gives an accurate inverse
mapping (destination Mercator pixel -> source raster pixel) without needing
a giant intermediate reprojected canvas.
"""
import json
import os

import mercantile
import numpy as np
import pyproj
from PIL import Image
from scipy.interpolate import RBFInterpolator

Image.MAX_IMAGE_PIXELS = None

RASTER_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\chart_raster_200dpi.png"
CONTROL_POINTS_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\control_points_v4.json"
TILES_OUT_DIR = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\api\static\map_tiles"

CROP_ORIGIN_X_PT = 950
CROP_ORIGIN_Y_PT = 150
DPI = 200
ZOOM = DPI / 72

TILE_SIZE = 256

# Transformer: WGS84 lon/lat -> Web Mercator meters
to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def load_control_points():
    data = json.load(open(CONTROL_POINTS_PATH))
    pts = data["points"]
    src_px, merc_xy = [], []
    for p in pts:
        rx = (p["px"] - CROP_ORIGIN_X_PT) * ZOOM
        ry = (p["py"] - CROP_ORIGIN_Y_PT) * ZOOM
        mx, my = to_merc.transform(p["lon"], p["lat"])
        src_px.append((rx, ry))
        merc_xy.append((mx, my))
    return np.array(src_px), np.array(merc_xy)


# Exact TPS interpolation (smoothing=0) overfits the inherent noise in
# text-extracted control points, producing wild local swirling especially
# in dense clusters (verified visually near Guam). This smoothing value was
# chosen by visual inspection: lower values still showed distortion, this
# level gives smooth, correctly-shaped coastlines/graticule with ~24px
# (~30km) median residual on the training points -- a reasonable trade for
# a chart-overlay/SIGMET-situational-awareness use case, not survey-grade
# navigation.
TPS_SMOOTHING = 1e11


def fit_inverse_interpolators(merc_xy, src_px):
    """Given Mercator (x,y), predict source raster (px,py). This is the
    inverse mapping needed for pull-based tile resampling."""
    fx = RBFInterpolator(merc_xy, src_px[:, 0], kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
    fy = RBFInterpolator(merc_xy, src_px[:, 1], kernel="thin_plate_spline", smoothing=TPS_SMOOTHING)
    return fx, fy


def report_fit_quality(fx, fy, merc_xy, src_px):
    pred_x = fx(merc_xy)
    pred_y = fy(merc_xy)
    err = np.sqrt((pred_x - src_px[:, 0]) ** 2 + (pred_y - src_px[:, 1]) ** 2)
    print(f"fit residuals (source pixels): mean={err.mean():.2f} max={err.max():.2f} "
          f"p95={np.percentile(err, 95):.2f}")


def choose_zoom_range(merc_xy, src_px, src_w, src_h):
    """Estimate native max zoom from control-point spacing: pick the zoom
    where Web Mercator meters/pixel roughly matches the source raster's own
    ground resolution, so we're not just blurrily upsampling."""
    # crude scale estimate: median nearest-neighbor pair gives m/src-px
    from scipy.spatial import cKDTree
    tree = cKDTree(merc_xy)
    dists, idx = tree.query(merc_xy, k=2)
    merc_nn = dists[:, 1]
    src_nn = np.sqrt(((src_px[:, None, :] - src_px[idx]) ** 2).sum(-1))[:, 1]
    valid = src_nn > 1e-6
    m_per_srcpx = np.median(merc_nn[valid] / src_nn[valid])
    print(f"estimated Mercator meters per source pixel: {m_per_srcpx:.1f}")

    best_z, best_diff = 0, float("inf")
    for z in range(0, 14):
        res = 156543.03392804097 / (2 ** z)  # meters/px at zoom z, equator
        diff = abs(res - m_per_srcpx)
        if diff < best_diff:
            best_z, best_diff = z, diff
    return best_z


def main():
    src_px, merc_xy = load_control_points()
    print(f"loaded {len(src_px)} control points")

    fx, fy = fit_inverse_interpolators(merc_xy, src_px)
    report_fit_quality(fx, fy, merc_xy, src_px)

    src_img = Image.open(RASTER_PATH).convert("RGB")
    src_w, src_h = src_img.size
    src_arr = np.asarray(src_img)
    print("source raster:", src_w, "x", src_h)

    max_zoom = choose_zoom_range(merc_xy, src_px, src_w, src_h)
    min_zoom = max(0, max_zoom - 6)
    print(f"chosen zoom range: {min_zoom}-{max_zoom}")

    # Geographic bounding box covered by control points, used to enumerate
    # which tiles are worth generating at each zoom. Points span the
    # antimeridian (Guam side ~+120..+180, California side ~-180..-110), so
    # naive min/max on raw signed longitude would wrongly span nearly the
    # whole globe -- shift negatives by +360 into a continuous range first,
    # take min/max there, then shift back.
    data = json.load(open(CONTROL_POINTS_PATH))
    lats = [p["lat"] for p in data["points"]]
    lons_shifted = [p["lon"] + 360 if p["lon"] < 0 else p["lon"] for p in data["points"]]
    west_shifted, east_shifted = min(lons_shifted), max(lons_shifted)
    west = west_shifted - 360 if west_shifted > 180 else west_shifted
    east = east_shifted - 360 if east_shifted > 180 else east_shifted
    south, north = min(lats), max(lats)
    print(f"bbox: lon [{west},{east}] lat [{south},{north}]")

    total_written = 0
    for z in range(min_zoom, max_zoom + 1):
        # tiles crossing the antimeridian: mercantile.tiles handles bbox
        # arguments spanning it if west > east by splitting internally is
        # NOT supported, so issue two passes when needed.
        bboxes = [(west, south, east, north)] if west <= east else [
            (west, south,180.0, north), (-180.0, south, east, north)
        ]
        for w, s, e, n in bboxes:
            for tile in mercantile.tiles(w, s, e, n, [z]):
                out_path = os.path.join(TILES_OUT_DIR, str(z), str(tile.x), f"{tile.y}.png")
                tile_bounds = mercantile.xy_bounds(tile)

                xs = np.linspace(tile_bounds.left, tile_bounds.right, TILE_SIZE, endpoint=False)
                ys = np.linspace(tile_bounds.top, tile_bounds.bottom, TILE_SIZE, endpoint=False)
                gx, gy = np.meshgrid(xs, ys)
                query = np.stack([gx.ravel(), gy.ravel()], axis=1)

                sx = fx(query)
                sy = fy(query)

                in_bounds = (sx >= 0) & (sx < src_w) & (sy >= 0) & (sy < src_h)
                if not in_bounds.any():
                    continue

                sx_i = np.clip(sx, 0, src_w - 1).astype(np.int32)
                sy_i = np.clip(sy, 0, src_h - 1).astype(np.int32)
                tile_rgb = src_arr[sy_i, sx_i].reshape(TILE_SIZE, TILE_SIZE, 3)

                alpha = (in_bounds.reshape(TILE_SIZE, TILE_SIZE) * 255).astype(np.uint8)
                tile_rgba = np.dstack([tile_rgb, alpha])

                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                Image.fromarray(tile_rgba, "RGBA").save(out_path, optimize=True)
                total_written += 1
        print(f"zoom {z}: {total_written} tiles written so far")

    print("done,", total_written, "tiles written to", TILES_OUT_DIR)


if __name__ == "__main__":
    main()
