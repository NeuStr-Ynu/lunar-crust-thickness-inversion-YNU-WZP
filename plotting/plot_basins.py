import numpy as np
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic

R_MOON = 1737400.0
_geod_moon = Geodesic(radius=R_MOON, flattening=0)

def _wrap180(lon_deg):
    return ((lon_deg + 180) % 360) - 180

def add_basin(
    ax, name, lat0, lon0, r_km,
    *,
    edgecolor="0.15",         # 比纯黑更柔和（Nature常用深灰）
    lw=0.8,
    alpha=0.75,
    n_samples=360,
    label=True,
    label_kwargs=None,
    center_marker=True,
    center_s=8,
    label_offset=(0.0, 0.0),  # (d_lon, d_lat) 给缩写一个小偏移，防止挡住细节
):
    pts = _geod_moon.circle(lon=lon0, lat=lat0, radius=r_km * 1000, n_samples=n_samples)

    lons = _wrap180(pts[:, 0])
    lats = pts[:, 1]

    # 在跨越±180°处断开，避免连线
    dlon = np.diff(lons)
    breaks = np.where(np.abs(dlon) > 180)[0] + 1
    segments = np.split(np.column_stack([lons, lats]), breaks)

    for seg in segments:
        if len(seg) < 2:
            continue
        ax.plot(
            seg[:, 0], seg[:, 1],
            transform=ccrs.PlateCarree(),
            color=edgecolor,
            linewidth=lw,
            alpha=alpha,
            zorder=6
        )

    # 圆心
    if center_marker:
        ax.scatter(
            [_wrap180(lon0)], [lat0],
            transform=ccrs.PlateCarree(),
            s=center_s,
            color=edgecolor,
            alpha=alpha,
            zorder=7
        )

    # 标注（更克制的 bbox）
    if label:
        if label_kwargs is None:
            label_kwargs = {}

        dlon_off, dlat_off = label_offset
        lon_t = _wrap180(lon0 + dlon_off)
        lat_t = lat0 + dlat_off

        ax.text(
            lon_t, lat_t, name,
            transform=ccrs.PlateCarree(),
            fontsize=8,
            ha="center",
            va="center",
            color=edgecolor,
            bbox=dict(
                facecolor="white",
                alpha=0.35,          # 更淡
                edgecolor="none",
                pad=0.8              # 更紧凑
            ),
            zorder=8,
            **label_kwargs
        )