"""
Interim, non-georeferenced tile pyramid for the new chart: a plain pixel
image pyramid (like the existing deployed app uses with L.CRS.Simple), no
lat/lon coordinates involved. This ships the visual chart upgrade now while
real georeferencing (via QGIS, done by hand) is in progress separately.

Uses the standard Leaflet "CRS.Simple with a big image" pattern: pick a
maxZoom where native resolution lives, and let the frontend compute bounds
with map.unproject([width, height], maxZoom) rather than hand-deriving a
coordinate convention.
"""
import math
import os

from PIL import Image

Image.MAX_IMAGE_PIXELS = None

SOURCE_PATH = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\scratch\full_chart_150dpi.png"
TILES_OUT_DIR = r"C:\Users\conno\Desktop\CE-PIREP-GEOCHART\api\static\map_tiles_flat"
TILE_SIZE = 256


def main():
    src = Image.open(SOURCE_PATH).convert("RGB")
    w, h = src.size
    print(f"source: {w}x{h}")

    max_zoom = math.ceil(math.log2(max(w, h) / TILE_SIZE))
    print(f"max_zoom: {max_zoom}")

    total = 0
    for z in range(max_zoom, -1, -1):
        scale = 2 ** (z - max_zoom)
        zw, zh = max(1, round(w * scale)), max(1, round(h * scale))
        level_img = src.resize((zw, zh), Image.LANCZOS) if scale != 1 else src

        cols = math.ceil(zw / TILE_SIZE)
        rows = math.ceil(zh / TILE_SIZE)
        for tx in range(cols):
            for ty in range(rows):
                box = (tx * TILE_SIZE, ty * TILE_SIZE,
                       min((tx + 1) * TILE_SIZE, zw), min((ty + 1) * TILE_SIZE, zh))
                tile = level_img.crop(box)
                if tile.size != (TILE_SIZE, TILE_SIZE):
                    padded = Image.new("RGB", (TILE_SIZE, TILE_SIZE), (255, 255, 255))
                    padded.paste(tile, (0, 0))
                    tile = padded
                out_dir = os.path.join(TILES_OUT_DIR, str(z), str(tx))
                os.makedirs(out_dir, exist_ok=True)
                tile.save(os.path.join(out_dir, f"{ty}.png"), optimize=True)
                total += 1
        print(f"zoom {z}: {cols}x{rows} tiles, {total} total so far")

    print(f"done: {total} tiles written to {TILES_OUT_DIR}")
    print(f"frontend needs: imageWidth={w}, imageHeight={h}, maxZoom={max_zoom}")


if __name__ == "__main__":
    main()
