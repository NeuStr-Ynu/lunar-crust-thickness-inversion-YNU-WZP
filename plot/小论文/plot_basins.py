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
    edgecolor="0.15",
    lw=0.8,
    alpha=0.75,
    n_samples=720,            # 加密采样,limb 处更顺滑
    label=True,
    label_kwargs=None,
    label_offset=(0.0, 0.0),
):
    pts = _geod_moon.circle(lon=lon0, lat=lat0, radius=r_km * 1000, n_samples=n_samples)
    lons = _wrap180(pts[:, 0])
    lats = pts[:, 1]

    # 闭合一下,首尾相接
    lons = np.append(lons, lons[0])
    lats = np.append(lats, lats[0])

    # 只在"经度跨日界线 且 不是绕极"时断开
    # 绕极的特征:相邻两点纬度都很高(接近 ±90)
    dlon = np.diff(lons)
    polar_threshold = 75  # 纬度绝对值大于此值认为是绕极区
    near_pole = (np.abs(lats[:-1]) > polar_threshold) | (np.abs(lats[1:]) > polar_threshold)
    breaks = np.where((np.abs(dlon) > 180) & ~near_pole)[0] + 1
    segments = np.split(np.column_stack([lons, lats]), breaks)

    for seg in segments:
        if len(seg) < 2:
            continue
        ax.plot(
            seg[:, 0], seg[:, 1],
            transform=ccrs.Geodetic(),   # 关键改动:用 Geodetic 让 cartopy 沿大圆插值
            color=edgecolor,
            linewidth=lw,
            alpha=alpha,
            zorder=6,
        )

    if label:
        if label_kwargs is None:
            label_kwargs = {}
        dlon_off, dlat_off = label_offset
        lon_t = _wrap180(lon0 + dlon_off)
        lat_t = lat0 + dlat_off
        ax.text(
            lon_t, lat_t, name,
            transform=ccrs.PlateCarree(),
            fontsize=8, ha="center", va="center",
            color=edgecolor,
            bbox=dict(facecolor="white", alpha=0.35, edgecolor="none", pad=0.8),
            zorder=8,
            **label_kwargs,
        )