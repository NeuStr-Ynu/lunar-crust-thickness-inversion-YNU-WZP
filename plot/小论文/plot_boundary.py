import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs

def _normalize_lon(lon):
    """把经度统一到 [-180, 180) 方便切段。"""
    lon = np.asarray(lon, dtype=float)
    return ((lon + 180.0) % 360.0) - 180.0

def _split_by_dateline(lon, lat, threshold_deg=180.0):
    """
    把跨越日期变更线(±180)造成的“经度跳变”切成多段，避免画出横穿全球的连线。
    threshold_deg: 判定跳变的阈值，默认 180 度。
    返回: list of (lon_seg, lat_seg)
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)

    if len(lon) < 2:
        return [(lon, lat)]

    # 计算相邻点经度差
    dlon = np.abs(np.diff(lon))
    break_idx = np.where(dlon > threshold_deg)[0]  # 在 i 与 i+1 之间断开

    if break_idx.size == 0:
        return [(lon, lat)]

    segments = []
    start = 0
    for i in break_idx:
        end = i + 1
        segments.append((lon[start:end], lat[start:end]))
        start = end
    segments.append((lon[start:], lat[start:]))
    return segments

def plot_digitized_boundary_on_ax(
    ax,
    csv_path,
    *,
    thick=False,
    linewidth_thick=2.8,
    linewidth_thin=1.2,
    transform=ccrs.PlateCarree(),
    normalize_longitude=True,
    dateline_split_threshold=180.0,
    label=None,
):
    """
    把单个 digitized 边界 CSV 画到你传入的 cartopy ax 上。

    参数：
    - ax: 你已有的 cartopy GeoAxes（比如 Orthographic 投影）
    - csv_path: *_digitized_boundary.csv
    - thick: True/False，控制粗线（SPATU、MTU）
    - label: 图例标签（可选）
    """
    df = pd.read_csv(csv_path)

    lon = df["lon_deg"].to_numpy()
    lat = df["lat_deg"].to_numpy()

    if normalize_longitude:
        lon = _normalize_lon(lon)

    # 切段避免跨 ±180° 的长连线
    segments = _split_by_dateline(lon, lat, threshold_deg=dateline_split_threshold)

    lw = linewidth_thick if thick else linewidth_thin

    # 逐段画
    handle = None
    for k, (lon_seg, lat_seg) in enumerate(segments):
        if len(lon_seg) < 2:
            continue
        # 只在第一段传 label，避免图例重复
        seg_label = label if (label is not None and k == 0) else None
        handle = ax.plot(
            lon_seg,
            lat_seg,
            linewidth=lw,
            color="black",
            transform=transform,
            label=seg_label,
        )[0]
    return handle

def plot_boundaries_folder_on_ax(
    ax,
    folder_path,
    *,
    thick_keywords=("SPATU", "MTU"),
    file_suffix="_digitized_boundary.csv",
    linewidth_thick=2.8,
    linewidth_thin=1.2,
    sort_files=True,
    add_legend=False,
):
    """
    扫描文件夹内所有 *_digitized_boundary.csv，画到给定 ax 上。
    自动：文件名包含 SPATU 或 MTU => 粗线。
    """
    files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(file_suffix)
    ]
    if sort_files:
        files.sort()

    handles = []
    for p in files:
        base = os.path.basename(p)
        thick = any(k in base for k in thick_keywords)

        # 可选：用文件名前缀做图例名（不想要 legend 就不传）
        label = None
        if add_legend:
            label = base.replace(file_suffix, "")

        h = plot_digitized_boundary_on_ax(
            ax,
            p,
            thick=thick,
            linewidth_thick=linewidth_thick,
            linewidth_thin=linewidth_thin,
            label=label,
        )
        if h is not None:
            handles.append(h)

    if add_legend and handles:
        ax.legend(loc="lower left", fontsize=8)

    return handles