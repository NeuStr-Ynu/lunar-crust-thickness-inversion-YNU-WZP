import matplotlib.patheffects as pe
import cartopy.crs as ccrs

def add_text_lonlat(ax, lon0, lat0, text, *,
                    fontsize=10, ha="center", va="center",
                    color="k", weight="normal",
                    dx=0.0, dy=0.0,
                    outline=True, outline_lw=2.0, outline_color="white",
                    bbox=False):
    """
    在经纬度位置添加文字（支持白色描边，论文常用，防止压在底图上看不清）
    dx/dy: 经纬度偏移（单位：度），用来微调避免压线
    """
    txt = ax.text(
        lon0 + dx, lat0 + dy, text,
        transform=ccrs.PlateCarree(),
        fontsize=fontsize,
        ha=ha, va=va,
        color=color,
        fontweight=weight,
        zorder=20,
        bbox=(dict(facecolor="white", alpha=0.5, edgecolor="none", pad=1.2) if bbox else None)
    )
    if outline:
        txt.set_path_effects([
            pe.Stroke(linewidth=outline_lw, foreground=outline_color),
            pe.Normal()
        ])
    return txt