"""
Warp the rasterized chart into Web Mercator (EPSG:3857) using ground control
points extracted directly from the chart's own printed coordinate text.

Uses a GCP-based transform (rasterio wraps GDAL's warper) rather than a naive
linear/equirectangular mapping, since the source chart's projection is not
plate-carree.
"""
import json

import numpy as np
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from rasterio.warp import calculate_default_transform, reproject, Resampling

RASTER_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\chart_raster_200dpi.png"
CONTROL_POINTS_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\control_points_final.json"
OUT_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\chart_webmercator.tif"

# Crop origin/zoom used when the raster was rendered from the PDF -- must
# match rasterize step exactly to convert PDF-point control points into
# raster pixel coordinates.
CROP_ORIGIN_X_PT = 950
CROP_ORIGIN_Y_PT = 150
DPI = 200
ZOOM = DPI / 72


def pdf_pt_to_raster_px(px_pt, py_pt):
    return (px_pt - CROP_ORIGIN_X_PT) * ZOOM, (py_pt - CROP_ORIGIN_Y_PT) * ZOOM


def main():
    data = json.load(open(CONTROL_POINTS_PATH))
    points = data["points"]

    # The control-point set spans the antimeridian (Guam/Japan side at
    # +120..+180, California side at -180..-110): raw signed longitudes jump
    # from +179 to -179, which breaks GDAL's polynomial GCP fit. Shift
    # negative longitudes by +360 so the whole set is numerically continuous
    # (~120..~250); PROJ normalizes this correctly during the actual
    # geographic->Web Mercator projection step.
    gcps = []
    for p in points:
        rx, ry = pdf_pt_to_raster_px(p["px"], p["py"])
        lon = p["lon"] + 360 if p["lon"] < 0 else p["lon"]
        gcps.append(GroundControlPoint(row=ry, col=rx, x=lon, y=p["lat"]))

    print(f"Using {len(gcps)} ground control points")

    src_crs = CRS.from_epsg(4326)  # GCPs are in plain lon/lat
    dst_crs = CRS.from_epsg(3857)

    # Step 1: read the plain (untransformed) raster and re-save it as a
    # GeoTIFF with the GCPs embedded directly in the file. rasterio/GDAL's
    # transform auto-detection only works reliably when GCPs live on the
    # source dataset itself, not when passed as a loose override.
    gcp_tagged_path = OUT_PATH.replace(".tif", "_gcp_tagged.tif")
    with rasterio.open(RASTER_PATH) as src:
        print("source raster:", src.width, "x", src.height, src.count, "bands")
        data_arr = src.read()
        profile = src.profile.copy()
        profile.update({"driver": "GTiff", "compress": "deflate"})
        with rasterio.open(gcp_tagged_path, "w", **profile) as tmp:
            tmp.write(data_arr)
            tmp.gcps = (gcps, src_crs)

    # Step 2: reopen the GCP-tagged file and reproject using its own GCPs,
    # which GDAL can now find on the dataset itself.
    with rasterio.open(gcp_tagged_path) as src:
        print("GCPs on source:", len(src.gcps[0]), "crs:", src.gcps[1])
        dst_transform, dst_width, dst_height = calculate_default_transform(
            src.gcps[1], dst_crs, src.width, src.height, gcps=src.gcps[0],
        )
        print("output raster:", dst_width, "x", dst_height)

        profile = src.profile.copy()
        profile.update({
            "height": dst_height,
            "width": dst_width,
            "transform": dst_transform,
            "crs": dst_crs,
            "compress": "deflate",
        })

        with rasterio.open(OUT_PATH, "w", **profile) as dst:
            for band_i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_i),
                    destination=rasterio.band(dst, band_i),
                    src_crs=src.gcps[1],
                    gcps=src.gcps[0],
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                    num_threads=4,
                )

    print("wrote", OUT_PATH)


if __name__ == "__main__":
    main()
