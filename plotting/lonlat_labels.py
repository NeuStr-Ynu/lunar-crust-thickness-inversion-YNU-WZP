import numpy as np
import cartopy.crs as ccrs


def fmt_lat(lat):
    if np.isclose(lat, 0): return "0°"
    return f"{abs(int(lat))}°{'N' if lat > 0 else 'S'}"

def fmt_lon(lon):
    lon = ((lon + 180) % 360) - 180
    if np.isclose(lon, 0): return "0°"
    if np.isclose(abs(lon), 180): return "180°"
    return f"{abs(int(lon))}°{'E' if lon > 0 else 'W'}"

def _visible_on_front(lon, lat, lon0, lat0):
    """前半球可见性：cosc >= 0"""
    lam = np.deg2rad(lon - lon0)
    phi = np.deg2rad(lat)
    phi0 = np.deg2rad(lat0)
    cosc = np.sin(phi0)*np.sin(phi) + np.cos(phi0)*np.cos(phi)*np.cos(lam)
    return cosc >= 0

# ✅ 补上：把经度 wrap 到离投影中心最近的等价经度，避免背面 transform 错半球
def wrap_lon_near_center(lon, lon0):
    return lon0 + (((lon - lon0) + 180) % 360) - 180

def limb_lon_for_lat(lat, lon0, lat0, side="left"):
    """
    给定纬度 lat，求它与 limb(cosc=0) 的交点经度。
    side: 'left' 或 'right'（相对于投影画面）
    解：sinφ0 sinφ + cosφ0 cosφ cosΔλ = 0
        => cosΔλ = -tanφ0 tanφ
    """
    phi = np.deg2rad(lat)
    phi0 = np.deg2rad(lat0)

    rhs = -np.tan(phi0) * np.tan(phi)

    # rhs 超出 [-1,1] 说明该纬度圈与 limb 不相交（接近极区会发生）
    if rhs < -1 or rhs > 1:
        return None

    dlam = np.rad2deg(np.arccos(rhs))  # Δλ >= 0

    lon_a = lon0 - dlam
    lon_b = lon0 + dlam

    return lon_a, lon_b

def add_lat_labels_on_limb(ax, ylocs, lat_formatter=fmt_lat, side="left",
                           offset_frac=0.03, text_kwargs=None):
    """
    纬度标签严格贴圆周(limb)。把标签放在 limb 交点处并沿半径方向外移 offset。
    side: left/right 选圆周的左侧或右侧交点。
    """
    if text_kwargs is None:
        text_kwargs = dict(fontsize=9)

    proj = ax.projection
    data_crs = ccrs.PlateCarree()

    lon0 = getattr(proj, "central_longitude", 0.0)
    lat0 = getattr(proj, "central_latitude", 0.0)

    # 估计圆盘半径，用一个 limb 点
    xr, yr = proj.transform_point(lon0 + 90, 0, data_crs)
    rmax = (xr**2 + yr**2) ** 0.5
    off = offset_frac * rmax

    for lat in ylocs:
        sol = limb_lon_for_lat(lat, lon0, lat0, side=side)
        if sol is None:
            continue
        lon_a, lon_b = sol

        # ✅ 关键补丁：背面也要 wrap，否则 transform_point 可能把点投到“另一侧”
        lon_a_p = wrap_lon_near_center(lon_a, lon0)
        lon_b_p = wrap_lon_near_center(lon_b, lon0)

        # 投影两解，挑选屏幕“左/右”的那个（看 x）
        xa, ya = proj.transform_point(lon_a_p, lat, data_crs)
        xb, yb = proj.transform_point(lon_b_p, lat, data_crs)

        if side == "left":
            x, y = (xa, ya) if xa < xb else (xb, yb)
            ha = "right"
        else:
            x, y = (xa, ya) if xa > xb else (xb, yb)
            ha = "left"

        # 严格在 limb 上：cosc=0 时 (x,y) 就在圆周；再沿半径方向外移一点
        rr = (x**2 + y**2) ** 0.5
        if rr == 0:
            continue
        x_text = x * (1 + off / rr)
        y_text = y * (1 + off / rr)

        ax.text(
            x_text, y_text, lat_formatter(lat),
            transform=proj, ha=ha, va="center",
            clip_on=False, **text_kwargs
        )

def add_lon_labels_on_equator(ax, xlocs, lon_formatter=fmt_lon, central_lonlat=(0.0,0.0),
                              offset_frac=0.03, text_kwargs=None):
    if text_kwargs is None:
        text_kwargs = dict(fontsize=9)

    proj = ax.projection

    lon0 = getattr(proj, "central_longitude", central_lonlat[0])
    lat0 = getattr(proj, "central_latitude", central_lonlat[1])

    # ✅ 源 CRS 用真实经纬度坐标系（不要带 central_longitude）
    data_crs = ccrs.PlateCarree()

    # 圆盘半径与偏移
    xr, yr = proj.transform_point(lon0 + 90, 0, data_crs)
    rmax = (xr**2 + yr**2) ** 0.5
    off = offset_frac * rmax

    lat_eq = 0.0
    phi0 = np.deg2rad(lat0)

    for lon in xlocs:
        # 把经度 wrap 到离中心最近（等价经度）
        lon_plot = wrap_lon_near_center(lon, lon0)

        # 可见性（lat=0）
        lam = np.deg2rad(lon_plot - lon0)
        cosc = np.cos(phi0) * np.cos(lam)
        if cosc < 0:
            continue

        # ✅ 用真实经纬度 CRS 做 transform
        x, y = proj.transform_point(lon_plot, lat_eq, data_crs)

        # 有些实现对“边界点”可能给 nan，这里保险一下
        if not np.isfinite(x) or not np.isfinite(y):
            continue

        ax.text(
            x, y - off, lon_formatter(lon),
            transform=proj, ha="center", va="top",
            clip_on=False, **text_kwargs
        )
        
def plot_apollo_sites(ax, sites, show_labels=True):
    for name, info in sites.items():
        lat0 = info["lat"]
        lon0 = info["lon"]

        # 点
        ax.scatter(
            [lon0], [lat0],
            transform=ccrs.PlateCarree(),
            s=42,
            marker="o",
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=20
        )

        # 中心再加一个小黑点，更醒目
        ax.scatter(
            [lon0], [lat0],
            transform=ccrs.PlateCarree(),
            s=10,
            marker="o",
            color="black",
            zorder=21
        )

        if show_labels:
            # 根据站点稍微调整文字偏移，避免互相遮挡
            if name == "Apollo 12":
                dx, dy = -10, -2
                ha, va = "right", "center"
            elif name == "Apollo 15":
                dx, dy = 4, 3
                ha, va = "left", "bottom"
            elif name == "Apollo 16":
                dx, dy = 4, -3
                ha, va = "left", "top"
            else:
                dx, dy = 3, 3
                ha, va = "left", "bottom"

            ax.text(
                lon0 + dx, lat0 + dy,
                name,
                transform=ccrs.PlateCarree(),
                fontsize=8.5,
                color="black",
                ha=ha, va=va,
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.55,
                    pad=0.8
                ),
                zorder=22
            )