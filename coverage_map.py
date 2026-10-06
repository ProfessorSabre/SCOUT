import argparse
import math
import textwrap
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pyproj import Transformer
from scipy.spatial import ConvexHull

from homography_middleware import SITE_KEY, camera_entries, load_site_registry, site_origin_utm

# SCOUT coverage map: where each camera places people on the ground, over aerial
# imagery, plus where camera coverage overlaps. Produced by default at the end
# of every mcmt_fusion.py run; also runnable on its own (e.g. a single camera
# that skips fusion).
#
# It's the quickest whole-run sanity check there is: a good calibration puts
# people on walkways, not roofs; a camera placing people far outside its
# calibrated area is extrapolating; and the overlap panel shows how much ground
# cross-camera matching actually has to work with.

PERSON_CLASS_NAME = "person"

# Reference categorical palette, fixed order (dataviz reference instance). On a
# map every pair of colors can sit side by side, and only the first three slots
# stay distinguishable for color-blind viewers under that all-pairs condition --
# so per-camera colors are only relied on for identity when there are <= 3
# cameras; beyond that, each camera's own panel title carries its identity.
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MAX_COLOR_CODED_CAMERAS = 3
TEXT, MUTED, RULE, SURFACE = "#0b0b0b", "#52514e", "#d0cfca", "#fcfcfb"

# A 2 m cell counts as covered by a camera once that camera puts at least this
# many person detections in it -- enough to ignore a stray mis-projection,
# small enough that a walkway people only cross briefly still registers.
CELL_M = 2.0
MIN_HITS = 5

# Aerial imagery. Tile sources are fetched in Web Mercator and resampled into
# the site's UTM grid; ArcGIS image/map services are asked for the UTM extent
# directly. "Esri" export is not an option -- World_Imagery is tile-only and
# rejects export requests (HTTP 500).
BASEMAPS = {
    "esri": {"kind": "tiles", "label": "Esri World Imagery", "max_zoom": 19,
             "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"},
    "esri-clarity": {"kind": "tiles", "label": "Esri World Imagery (Clarity)", "max_zoom": 19,
                     "url": "https://clarity.maptiles.arcgis.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"},
    "indiana": {"kind": "arcgis", "label": "Indiana statewide orthophotos (IGIO, CC0)",
                "url": "https://di-ingov.img.arcgis.com/arcgis/rest/services/DynamicWebMercator/Indiana_Current_Imagery/ImageServer"},
}
MAX_TILES = 144  # keeps a very large site from requesting thousands of tiles; zoom drops instead


def basemap_spec(name: str) -> dict | None:
    if name in (None, "", "none"):
        return None
    if name in BASEMAPS:
        return BASEMAPS[name]
    if "{z}" in name:
        return {"kind": "tiles", "label": "custom imagery", "max_zoom": 19, "url": name}
    return {"kind": "arcgis", "label": "custom imagery", "url": name.rstrip("/")}


def _lonlat_to_global_px(lon, lat, z):
    n = 256 * 2 ** z
    lat_r = np.radians(lat)
    return (np.asarray(lon) + 180) / 360 * n, (1 - np.log(np.tan(lat_r) + 1 / np.cos(lat_r)) / math.pi) / 2 * n


def fetch_basemap(lo_utm, hi_utm, epsg: int, width_px: int, spec: dict):
    """RGB image covering [lo_utm, hi_utm] in the given UTM zone, north up, or
    None if the imagery can't be fetched (offline, blocked, service down) --
    the map is still drawn, just on a plain background."""
    width_m, height_m = hi_utm - lo_utm
    height_px = max(1, int(round(width_px * height_m / width_m)))
    try:
        if spec["kind"] == "arcgis":
            op = "exportImage" if spec["url"].lower().endswith("/imageserver") else "export"
            fmt = "jpgpng" if op == "exportImage" else "jpg"
            r = requests.get(f"{spec['url']}/{op}?bbox={lo_utm[0]},{lo_utm[1]},{hi_utm[0]},{hi_utm[1]}&bboxSR={epsg}"
                             f"&imageSR={epsg}&size={width_px},{height_px}&format={fmt}&f=image", timeout=120)
            img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
            return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        to_ll = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        corner_lon, corner_lat = to_ll.transform([lo_utm[0], hi_utm[0], lo_utm[0], hi_utm[0]], [lo_utm[1], lo_utm[1], hi_utm[1], hi_utm[1]])
        mid_lat = float(np.mean(corner_lat))
        target_m_per_px = width_m / width_px
        z = int(min(spec["max_zoom"], max(1, math.ceil(math.log2(156543.03 * math.cos(math.radians(mid_lat)) / target_m_per_px)))))
        while True:
            gx, gy = _lonlat_to_global_px(np.array(corner_lon), np.array(corner_lat), z)
            tx0, tx1 = int(gx.min() // 256), int(gx.max() // 256)
            ty0, ty1 = int(gy.min() // 256), int(gy.max() // 256)
            if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= MAX_TILES or z <= 1:
                break
            z -= 1
        mosaic = np.zeros(((ty1 - ty0 + 1) * 256, (tx1 - tx0 + 1) * 256, 3), np.uint8)
        with requests.Session() as session:
            for ty in range(ty0, ty1 + 1):
                for tx in range(tx0, tx1 + 1):
                    r = session.get(spec["url"].format(z=z, x=tx, y=ty), timeout=60)
                    tile = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
                    if tile is not None:
                        mosaic[(ty - ty0) * 256:(ty - ty0 + 1) * 256, (tx - tx0) * 256:(tx - tx0 + 1) * 256] = cv2.resize(tile, (256, 256))

        # Resample the Web Mercator mosaic onto the UTM output grid, pixel by pixel.
        res_x, res_y = width_m / width_px, height_m / height_px
        u, v = np.meshgrid(np.arange(width_px), np.arange(height_px))
        utm_x = lo_utm[0] + (u + 0.5) * res_x
        utm_y = hi_utm[1] - (v + 0.5) * res_y
        lon, lat = to_ll.transform(utm_x.ravel(), utm_y.ravel())
        px, py = _lonlat_to_global_px(lon, lat, z)
        map_x = (px - tx0 * 256).reshape(height_px, width_px).astype(np.float32)
        map_y = (py - ty0 * 256).reshape(height_px, width_px).astype(np.float32)
        return cv2.cvtColor(cv2.remap(mosaic, map_x, map_y, cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
    except (requests.RequestException, cv2.error, ValueError) as err:
        print(f"Coverage map: aerial imagery unavailable ({err.__class__.__name__}); drawing on a plain background")
        return None


def render_coverage_map(per_camera: dict, site_id: str, output_path, basemap: str = "esri") -> dict:
    """per_camera: {camera_id: DataFrame with Class, Local_X_Meters, Local_Y_Meters
    (and optionally Beyond_Horizon)}. Writes a PNG and returns the coverage
    areas (m²), so callers can log them."""
    registry = load_site_registry(site_id)
    cameras = sorted(per_camera)
    entries = camera_entries(registry)
    georeferenced = all(entries.get(c, {}).get("mode") == "global" for c in cameras)

    pts, beyond, calib = {}, {}, {}
    for c in cameras:
        df = per_camera[c]
        person = df[df["Class"] == PERSON_CLASS_NAME]
        xy = person[["Local_X_Meters", "Local_Y_Meters"]].to_numpy(dtype=np.float64)
        pts[c] = xy[np.isfinite(xy).all(1)]
        beyond[c] = int(person["Beyond_Horizon"].sum()) if "Beyond_Horizon" in person else 0
        cal = entries.get(c)
        if cal is None:
            calib[c] = np.zeros((0, 2))
        elif georeferenced:
            calib[c] = np.array(cal["utm_coords"]) - np.array(site_origin_utm(registry))
        else:
            calib[c] = np.array(cal["site_coords"], dtype=np.float64)

    # Extent: where most of each camera's people are (1st-99th percentile) plus
    # its calibration points, padded. Stray far projections are counted, not
    # drawn, so they can't shrink the site into a corner of the panel.
    corners = [np.percentile(pts[c], [1, 99], axis=0) for c in cameras if len(pts[c])]
    corners += [np.vstack([calib[c].min(0), calib[c].max(0)]) for c in cameras if len(calib[c])]
    if not corners:
        print("Coverage map: no positioned person detections to draw")
        return {}
    lo = np.min([k[0] for k in corners], axis=0) - 10
    hi = np.max([k[1] for k in corners], axis=0) + 10
    centre, half = (lo + hi) / 2, np.maximum((hi - lo) / 2, 20)
    lo, hi = centre - half, centre + half
    outside = {c: int((~((pts[c] >= lo) & (pts[c] <= hi)).all(1)).sum()) for c in cameras}

    image, imagery_label = None, "no aerial imagery"
    spec = basemap_spec(basemap) if georeferenced else None
    if spec:
        origin = np.array(site_origin_utm(registry))
        epsg = registry.get(SITE_KEY, {}).get("utm_epsg") or entries[cameras[0]]["utm_epsg"]
        image = fetch_basemap(lo + origin, hi + origin, epsg, 1600, spec)
        if image is not None:
            image = (image * 0.55 + 255 * 0.45).astype(np.uint8)  # recede the photo so the data carries the eye
            imagery_label = spec["label"]
    elif not georeferenced:
        imagery_label = "no aerial imagery (local tape-measure calibration has no geographic position)"

    nx, ny = int(math.ceil((hi[0] - lo[0]) / CELL_M)), int(math.ceil((hi[1] - lo[1]) / CELL_M))
    covered = {}
    for c in cameras:
        H, _, _ = np.histogram2d(pts[c][:, 0], pts[c][:, 1], bins=[nx, ny], range=[[lo[0], hi[0]], [lo[1], hi[1]]])
        covered[c] = H >= MIN_HITS
    n_cov = sum(covered[c].astype(int) for c in cameras)
    area = {c: float(covered[c].sum() * CELL_M ** 2) for c in cameras}
    if len(cameras) >= 2:
        area["2+ cameras"] = float((n_cov >= 2).sum() * CELL_M ** 2)
    if len(cameras) >= 3:
        area["all cameras"] = float((n_cov == len(cameras)).sum() * CELL_M ** 2)

    # A camera keeps the same color in every run at a site, whichever subset of
    # cameras a given run includes: colors follow the camera's place in the
    # site registry, not its place in this run's list.
    site_order = list(entries) + [c for c in cameras if c not in entries]
    color = {c: (CATEGORICAL[site_order.index(c)] if site_order.index(c) < len(CATEGORICAL) else MUTED) for c in cameras}
    n_panels = len(cameras) + (1 if len(cameras) >= 2 else 0)
    ncols = 1 if n_panels == 1 else (2 if n_panels <= 4 else 3)
    nrows = math.ceil(n_panels / ncols)
    aspect = (hi[1] - lo[1]) / (hi[0] - lo[0])
    # Each panel's plot area is ~6 in wide after its axis labels; its height
    # follows the site's aspect, plus ~1.3 in for the panel title and notes,
    # plus ~1 in for the figure header -- so rows sit snugly instead of
    # floating in whitespace an equal-aspect map would otherwise leave.
    header_in = 1.0
    fig_h = (6.0 * aspect + 1.3) * nrows + header_in
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, fig_h), dpi=110, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    extent = [lo[0], hi[0], lo[1], hi[1]]

    def base(ax, title, sub):
        if image is not None:
            ax.imshow(image, extent=extent, origin="upper", zorder=0)
        else:
            ax.set_facecolor("#f0efec")
        ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_aspect("equal")
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT, pad=36)
        ax.text(0, 1.01, sub, transform=ax.transAxes, fontsize=9, color=MUTED, va="bottom")
        ax.tick_params(colors=MUTED, labelsize=8)
        for s in ax.spines.values():
            s.set_color(RULE)
        ax.set_xlabel("meters east of site origin" if georeferenced else "meters (local calibration frame)", fontsize=8, color=MUTED)
        ax.set_ylabel("meters north" if georeferenced else "meters", fontsize=8, color=MUTED)

    rng = np.random.default_rng(0)
    flat = list(axes.flat)
    for ax, c in zip(flat, cameras):
        p = pts[c]
        show = p[rng.choice(len(p), min(len(p), 20000), replace=False)] if len(p) else p
        cal = entries.get(c, {})
        err = cal.get("error_m", {})
        quality = f" · median leave-one-out error {err['loo_median']:.1f} m" if "loo_median" in err else ""
        excluded = f" · {beyond[c]:,} above the ground horizon, not shown" if beyond[c] else ""
        base(ax, c, f"{len(p):,} person detections with a ground position ({outside[c]:,} outside this view){excluded}\n"
                    f"{len(calib[c])} calibration points{quality}")
        if len(show):
            ax.scatter(show[:, 0], show[:, 1], s=2.5, c=color[c], alpha=0.35, linewidths=0, zorder=2)
        handles = [Line2D([], [], ls="", marker="o", ms=5, mfc=color[c], mec=color[c], label="person detections")]
        if len(calib[c]):
            if len(calib[c]) >= 3:
                hull = calib[c][ConvexHull(calib[c]).vertices]
                ax.fill(hull[:, 0], hull[:, 1], fill=False, ls="--", lw=1.6, ec=TEXT, zorder=3)
                handles.append(Line2D([], [], ls="--", color=TEXT, label="area the calibration points span"))
            ax.scatter(calib[c][:, 0], calib[c][:, 1], marker="o", s=42, facecolor="white", edgecolor=TEXT, linewidths=1.4, zorder=4)
            handles.insert(1, Line2D([], [], ls="", marker="o", ms=7, mfc="white", mec=TEXT, label="calibration points (map side)"))
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.92, edgecolor=RULE)

    if len(cameras) >= 2:
        ax = flat[len(cameras)]
        all_line = f" · by all {len(cameras)}: {area['all cameras']:,.0f} m²" if "all cameras" in area else ""
        base(ax, "Coverage overlap",
             f"A {CELL_M:g} m cell counts as covered with >= {MIN_HITS} detections\nSeen by 2+ cameras: {area['2+ cameras']:,.0f} m²{all_line}")
        levels = list(range(2, len(cameras) + 1))
        grays = ["#52514e", "#0b0b0b"] if len(levels) <= 2 else [matplotlib.colors.to_hex(g) for g in plt.cm.Greys(np.linspace(0.45, 0.95, len(levels)))]
        gx, gy = np.linspace(lo[0], hi[0], nx + 1), np.linspace(lo[1], hi[1], ny + 1)
        ax.pcolormesh(gx, gy, np.ma.masked_where(n_cov.T < 2, n_cov.T), cmap=ListedColormap(grays[:len(levels)]),
                      vmin=1.5, vmax=len(cameras) + 0.5, alpha=0.75, zorder=2, shading="flat")
        handles = []
        if len(cameras) <= MAX_COLOR_CODED_CAMERAS:
            cx, cy = (gx[:-1] + gx[1:]) / 2, (gy[:-1] + gy[1:]) / 2
            for c in cameras:
                if covered[c].any():
                    ax.contour(cx, cy, covered[c].T.astype(float), levels=[0.5], colors=[color[c]], linewidths=1.8, zorder=3)
                handles.append(Line2D([], [], color=color[c], lw=2, label=f"{c} coverage ({area[c]:,.0f} m²)"))
        handles += [Patch(fc=g, alpha=0.75, label=f"seen by {k} cameras" if k < len(cameras) or len(cameras) == 2 else f"seen by all {k}")
                    for k, g in zip(levels, grays)]
        ax.legend(handles=handles, loc="lower left", fontsize=8, framealpha=0.92, edgecolor=RULE)

    for ax in flat[n_panels:]:
        ax.set_visible(False)

    fig_w = 7 * ncols
    title = f"Coverage: {site_id}" if ncols == 1 else f"Coverage: {site_id} - where each camera places people"
    note = textwrap.fill(f"Background: {imagery_label}. Positions are each detection's foot point, projected through its camera's "
                         "calibration; detections above the ground horizon have no ground position and are not shown.",
                         width=int(fig_w * 15))  # ~15 characters per inch at 9 pt
    fig.suptitle(title, x=0.01, y=1 - 0.15 / fig_h, ha="left", va="top", fontsize=15, fontweight="bold", color=TEXT)
    fig.text(0.01, 1 - 0.5 / fig_h, note, fontsize=9, color=MUTED, va="top")
    fig.tight_layout(rect=[0, 0.01, 1, 1 - header_in / fig_h], h_pad=2.5)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    return area


def main():
    parser = argparse.ArgumentParser(description="SCOUT coverage map: where each camera places people, and where cameras overlap")
    parser.add_argument("--csvs", nargs="+", required=True, help="Per-camera transformed (or fused) CSVs, one per camera")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--output", required=True, help="PNG to write")
    parser.add_argument("--basemap", default="esri",
                        help="esri | esri-clarity | indiana | none | an XYZ tile URL with {z}/{x}/{y} | an ArcGIS MapServer/ImageServer URL")
    args = parser.parse_args()

    frames = pd.concat([pd.read_csv(p) for p in args.csvs], ignore_index=True)
    per_camera = {str(c): g for c, g in frames.groupby("Camera")}
    areas = render_coverage_map(per_camera, args.site_id, args.output, args.basemap)
    print(f"Coverage map written to {args.output}  " + "  ".join(f"{k}: {v:,.0f} sq m" for k, v in areas.items()))


if __name__ == "__main__":
    main()
