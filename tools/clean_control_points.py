"""
Iteratively prune control points via leave-one-out consistency: fit TPS on
all-but-one, predict the held-out point's source pixel from its known real
coordinate, and remove points whose prediction disagrees badly with what was
actually extracted from the chart. This catches cross-waypoint mismatches
regardless of *why* they happened (loose distance match, same-route/same-
rotation collision, etc.) -- it only trusts points that are internally
consistent with their neighbors.
"""
import json
import sys

import numpy as np
import pyproj
from scipy.interpolate import RBFInterpolator

CROP_ORIGIN_X_PT = 950
CROP_ORIGIN_Y_PT = 150
DPI = 200
ZOOM = DPI / 72

ERROR_THRESHOLD_KM = 40  # drop points whose LOO-predicted position is off by more than this
MAX_ITERATIONS = 6

to_merc = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def to_arrays(points):
    src_px, merc_xy = [], []
    for p in points:
        rx = (p["px"] - CROP_ORIGIN_X_PT) * ZOOM
        ry = (p["py"] - CROP_ORIGIN_Y_PT) * ZOOM
        mx, my = to_merc.transform(p["lon"], p["lat"])
        src_px.append((rx, ry))
        merc_xy.append((mx, my))
    return np.array(src_px), np.array(merc_xy)


def m_per_srcpx(src_px, merc_xy):
    from scipy.spatial import cKDTree
    tree = cKDTree(merc_xy)
    dists, idx = tree.query(merc_xy, k=2)
    merc_nn = dists[:, 1]
    src_nn = np.sqrt(((src_px - src_px[idx[:, 1]]) ** 2).sum(1))
    valid = src_nn > 1
    return float(np.median(merc_nn[valid] / src_nn[valid]))


def loo_errors_km(points):
    src_px, merc_xy = to_arrays(points)
    scale = m_per_srcpx(src_px, merc_xy)
    n = len(points)
    err = np.zeros(n)
    idx_all = np.arange(n)
    for i in range(n):
        mask = idx_all != i
        fx = RBFInterpolator(merc_xy[mask], src_px[mask, 0], kernel="thin_plate_spline")
        fy = RBFInterpolator(merc_xy[mask], src_px[mask, 1], kernel="thin_plate_spline")
        px_pred = fx(merc_xy[i:i + 1])[0]
        py_pred = fy(merc_xy[i:i + 1])[0]
        err[i] = ((px_pred - src_px[i, 0]) ** 2 + (py_pred - src_px[i, 1]) ** 2) ** 0.5
    return err * scale / 1000, scale


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\control_points_raw.json"
    out_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\control_points_clean.json"

    points = json.load(open(in_path))["points"]
    print(f"starting with {len(points)} points")

    for iteration in range(MAX_ITERATIONS):
        err_km, scale = loo_errors_km(points)
        n_bad = int((err_km > ERROR_THRESHOLD_KM).sum())
        print(f"iter {iteration}: n={len(points)}, m/srcpx={scale:.1f}, "
              f"median_err={np.median(err_km):.1f}km, p90={np.percentile(err_km,90):.1f}km, "
              f"max={err_km.max():.1f}km, over_threshold={n_bad}")
        if n_bad == 0:
            break
        points = [p for p, e in zip(points, err_km) if e <= ERROR_THRESHOLD_KM]

    print(f"final: {len(points)} points")
    json.dump({"points": points}, open(out_path, "w"), indent=2)
    print("wrote", out_path)


if __name__ == "__main__":
    main()
