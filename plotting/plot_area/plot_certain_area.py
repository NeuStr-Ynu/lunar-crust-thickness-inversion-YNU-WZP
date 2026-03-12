import numpy as np
import pyshtools as pysh
import boule as bl
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
import pickle
import pandas as pd
import lonlat_labels as label
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from plot_boundary import plot_boundaries_folder_on_ax
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as pe
from pathlib import Path

# 中文依赖
plt.rcParams['font.sans-serif'] = ['SimHei']      # 黑体
plt.rcParams['axes.unicode_minus'] = False        # 解决负号显示问题

# ==========读取数据==========
# 壳厚数据
def read_crust_thick_data():
    PROJECT_ROOT = Path.cwd().parent.parent
    print("[running]: reading crust thick data...")
    # ===========读取数据 频率域==========
    data_filename = PROJECT_ROOT/"result/frequency_domain_run1.pkl"

    with open(data_filename, "rb") as f:
        result = pickle.load(f)

    thick = result["thick_grid"].data
    nlat, nlon = thick.shape

    lats = np.linspace(90, -90, nlat)
    lons = np.linspace(0, 360, nlon, endpoint=False)

    thick_da = xr.DataArray(thick, coords=[lats, lons], dims=["lat", "lon"])

    # 转换到 [-180, 180] 并排序
    frequency_thick_da = thick_da.copy()
    lon = ((frequency_thick_da.lon + 180) % 360) - 180
    frequency_thick_da = frequency_thick_da.assign_coords(lon=lon).sortby("lon")

    # ===========读取数据 空间域==========
    pkl_file = PROJECT_ROOT/"result/run_approach.pkl"

    with open(pkl_file, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

        # 提取所需变量并转为 numpy array
        density = np.asarray(obj['densities'])
        reference_levels = np.asarray(obj['reference_levels'])
        score_refden = np.asarray(obj['scores_refden'])
        score_regul = np.asarray(obj['scores_regul'])
        regul = np.asarray(obj['regul_params'])
        regul_residual = np.asarray(obj['regul_residuals'])
        refden_residual = np.asarray(obj['refden_residuals'])

        refden_moho_grid = obj['best_solutions_refden_moho_grid']
        refden_predict_grid = obj['best_solutions_refden_predict_grid']
        observe = obj['observe']
        lon = obj['lon']
        lat = obj['lat']
        lon_sub = obj['lon_sub']
        lat_sub = obj['lat_sub']
        
    nlat_moho, nlon_moho = refden_moho_grid.shape
    lon_moho = np.linspace(lon.min(), lon.max(), nlon_moho)
    lat_moho = np.linspace(lat.min(), lat.max(), nlat_moho)

    moho_grid = xr.DataArray(
        refden_moho_grid,  # 如果图东西颠倒或拉伸，改为 refden_moho_grid.T
        dims=("lat", "lon"),
        coords={"lat": lat_moho, "lon": lon_moho}
    )
    # 地形数据
    data_file = PROJECT_ROOT/"data/boueguer_frenquency_11km_withoutmoremare_topo_1deg.csv"

    df = pd.read_csv(data_file)
    lon_raw = np.sort(df['lon'].unique())
    lat_raw = np.sort(df['lat'].unique())
    nlat = len(np.unique(lat_raw))
    nlon = len(np.unique(lon_raw))

    # 计算壳厚
    topo_grid  = df.pivot(index='lat', columns='lon', values='topo').loc[lat_raw, lon_raw].values
    LON, LAT = np.meshgrid(lon_raw, lat_raw) 

    arr = topo_grid + bl.Moon2015.mean_radius

    nlon = arr.shape[1]
    half = nlon // 2

    arr_shifted = np.hstack([
        arr[:, half:],   # 后一半列
        arr[:, :half]    # 前一半列
    ])
    topo_da = xr.DataArray(
        arr_shifted,  # 如果图东西颠倒或拉伸，改为 refden_moho_grid.T
        dims=("lat", "lon"),
        coords={"lat": lat_raw, "lon": lon_raw}
    )

    region = [lon_raw.min(), lon_raw.max(), lat_raw.min(), lat_raw.max()]

    topo_da=topo_da.interp_like(moho_grid)

    thick_da=topo_da-moho_grid
    spatial_thick_da = thick_da.where(thick_da >= 0)
    spatial_thick_da.values=spatial_thick_da.values/1e3
    print("[running]: read crust thick data finished")
    return frequency_thick_da,spatial_thick_da
# 地形数据
def read_topo_data():
    print("[running]: reading topo data...")
    topo = pysh.datasets.Moon.LDEM_shape_pa()
    topo_grid = topo.expand(grid="DH2", lmax=1200, extend=False)
    topo_xr = topo_grid.to_xarray()
    topo_xr.values-=bl.Moon2015.mean_radius
    topo_xr.values=topo_xr.values/1e3
    return topo_xr
# 算梯度
def horizontal_gradient_from_xr(da_xr):
    """
    输入: xarray DataArray (lat, lon)，单位任意（这里是 mGal）
    输出: (grad_north, grad_east, grad_mag)  [单位: (输入单位)/m]
    """
    da_grid=pysh.SHGrid.from_xarray(da_xr)
    da_coff=da_grid.expand()
    gba=da_coff.gradient()
    
    GdaE_grid=gba.phi
    GdaN_grid=gba.theta
    
    gN=GdaN_grid.to_xarray()
    gE=GdaE_grid.to_xarray()
    
    gM=gE
    gM.data=np.sqrt(gE.data**2+gN.data**2)
    
    return gN, gE, gM
# 重力数据
def read_gravity_data():
    
    PROJECT_ROOT = Path.cwd().parent.parent
    # ✌就这么定义了
    height = 11e3
    porosity = 0.12
    
    data_filename=PROJECT_ROOT/"data/density_no_mare_n3000_f3050_719.sh"
    lmax=719
    # 密度
    densityfile = data_filename
    density = pysh.SHCoeffs.from_file(densityfile, lmax=lmax)
    density_grid=density.expand(grid='DH2',lmax=lmax,extend=False)
    density_xr=density_grid.to_xarray()
    # 重力
    pot=pysh.datasets.Moon.GRGM1200B()
    pot.set_omega(bl.Moon2015.angular_velocity)
    pot=pot.change_ref(gm=bl.Moon2015.geocentric_grav_const, r0=bl.Moon2015.radius)
    pot_grid= pot.expand(lmax=lmax, a=bl.Moon2015.radius + height, f=0, normal_gravity=False, extend=False)
    pot_xr=pot_grid.to_xarray()
    # 地形
    topo=pysh.datasets.Moon.LDEM_shape_pa()
    topo_grid=topo.expand(grid='DH2',lmax=lmax,extend=False)
    topo_xr=topo_grid.to_xarray()
    
    LON, LAT = np.meshgrid(pot_xr.lon, pot_xr.lat) 
    Moon2015_el=bl.Moon2015
    gamma=Moon2015_el.normal_gravity(latitude=LAT,height=height)
    
    # 自由空气重力异常
    data=-pot_xr.radial.data*1e5-gamma
    freeair_xr=xr.DataArray(data,dims=('lat','lon'),coords={'lat': pot_xr.lat,'lon': pot_xr.lon})
    
    # 布格重力异常
    bc, r0 = pysh.gravmag.CilmPlusRhoHDH(
        topo_grid.data,
        nmax=8,
        mass=pot.mass,
        rho=density_grid.data * (1 - porosity),
        lmax=lmax
    )
    
    bc_coff=pysh.SHGravCoeffs.from_array(bc,gm=bl.Moon2015.geocentric_grav_const, r0=bl.Moon2015.radius)
    
    bc_grid=bc_coff.expand(lmax=lmax,a=bl.Moon2015.mean_radius+height,f=0,extend=False)
    bc_raw_xr=bc_grid.to_xarray()
    
    bc_xr=xr.DataArray(-bc_raw_xr.radial.data*1e5,dims=('lat','lon'),coords={'lat': pot_xr.lat,'lon': pot_xr.lon})
    
    ba_xr=xr.DataArray(freeair_xr.data-bc_xr.data,dims=('lat','lon'),coords={'lat': pot_xr.lat,'lon': pot_xr.lon})
    
    fa_gN, fa_gE, fa_gM = horizontal_gradient_from_xr(freeair_xr)
    ba_gN, ba_gE, ba_gM = horizontal_gradient_from_xr(ba_xr)
    #, fa_gN, fa_gE, fa_gM, ba_gN, ba_gE, ba_gM
    return freeair_xr, ba_xr, fa_gN, fa_gE, fa_gM, ba_gN, ba_gE, ba_gM

# ==========画图==========
# 正投影画区域
def lon_center(lon_min, lon_max):
    lon_min_360 = lon_min % 360
    lon_max_360 = lon_max % 360

    d = (lon_max_360 - lon_min_360) % 360

    if d > 180:
        center = lon_min_360 - (360 - d) / 2
    else:
        center = lon_min_360 + d / 2

    return ((center + 180) % 360) - 180

def _wrap180(lon):
    """把经度 wrap 到 [-180, 180)"""
    return ((lon + 180) % 360) - 180

def _lon_to_continuous(lonA, lonB):
    """
    把 lonB 调到和 lonA 同一条连续经度轴上（以 360 为周期），
    使得从 lonA 到 lonB 走最短弧（避免穿球乱连）。
    返回 lonA_cont, lonB_cont （都在连续轴上，lonB 可能超出 [-180,180]）
    """
    a = lonA % 360
    b = lonB % 360
    d = (b - a) % 360
    if d > 180:
        d -= 360  # 走反向更短
    return a, a + d

def draw_segment(ax, lon_A, lon_B, lat_A, lat_B, mode, n=200,
                 color="red", linewidth=1.5, zorder=10):
    """
    画一段边：
      mode="h": 固定纬度 (lat_A==lat_B)，lon 从 lon_A 到 lon_B
      mode="v": 固定经度 (lon_A==lon_B)，lat 从 lat_A 到 lat_B
    """
    if mode == "h":
        # 水平边需要处理跨 180
        a_cont, b_cont = _lon_to_continuous(lon_A, lon_B)
        lons_cont = np.linspace(a_cont, b_cont, n)
        lons = _wrap180(lons_cont)
        lats = np.full(n, lat_A)

        # 关键：如果这条水平线确实跨 dateline（wrap 后出现跳变），就拆成两段画
        jumps = np.where(np.abs(np.diff(lons)) > 180)[0]
        if jumps.size > 0:
            k = jumps[0] + 1
            ax.plot(lons[:k], lats[:k], transform=ccrs.PlateCarree(),
                    color=color, linewidth=linewidth, zorder=zorder)
            ax.plot(lons[k:], lats[k:], transform=ccrs.PlateCarree(),
                    color=color, linewidth=linewidth, zorder=zorder)
            return

        ax.plot(lons, lats, transform=ccrs.PlateCarree(),
                color=color, linewidth=linewidth, zorder=zorder)

    elif mode == "v":
        # 竖直边不需要跨 180 处理（经度固定）
        lons = np.full(n, _wrap180(lon_A))
        lats = np.linspace(lat_A, lat_B, n)
        ax.plot(lons, lats, transform=ccrs.PlateCarree(),
                color=color, linewidth=linewidth, zorder=zorder)
    else:
        raise ValueError("mode must be 'h' or 'v'")

def plot_Orthographic_area(lon_max, lon_min, lat_max, lat_min, topo_da,
                           out="topo.png", dpi=300):
    print("[running]: plotting topo data...")

    lon_central = lon_center(lon_min, lon_max)
    lat_central = (lat_max + lat_min) / 2

    lon = topo_da["lon"].values
    lat = topo_da["lat"].values
    Z = topo_da.values

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["SimHei", "Microsoft YaHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })

    fig = plt.figure(figsize=(3.6, 3.6), dpi=dpi)
    fig.patch.set_alpha(0)  # figure 背景透明

    ax = plt.subplot(
        1, 1, 1,
        projection=ccrs.Orthographic(
            central_longitude=lon_central,
            central_latitude=lat_central
        )
    )
    ax.set_global()
    ax.patch.set_alpha(0)   # axes 背景透明

    pm = ax.pcolormesh(
        lon, lat, Z,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap="gray",
    )

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        xlocs=np.arange(-180, 181, 30),
        ylocs=np.arange(-90, 91, 30),
        linewidth=0.5, alpha=0.35, linestyle="--",
        draw_labels=False
    )
    
    # 上边：lon_min -> lon_max, lat=lat_max
    draw_segment(ax, lon_min, lon_max, lat_max, lat_max, mode="h")

    # 右边：lon=lon_max, lat_max -> lat_min
    draw_segment(ax, lon_max, lon_max, lat_max, lat_min, mode="v")

    # 下边：lon_max -> lon_min, lat=lat_min
    draw_segment(ax, lon_max, lon_min, lat_min, lat_min, mode="h")

    # 左边：lon=lon_min, lat_min -> lat_max
    draw_segment(ax, lon_min, lon_min, lat_min, lat_max, mode="v")

    xlocs = np.arange(-180, 180, 30)
    ylocs = np.arange(-90, 91, 30)
    label.add_lat_labels_on_limb(ax, ylocs, side="left")
    label.add_lon_labels_on_equator(ax, xlocs, central_lonlat=(lon_central, lat_central))
    PROJECT_ROOT = Path.cwd().parent.parent
    folder_path=PROJECT_ROOT/"plotting/data"
    plot_boundaries_folder_on_ax(ax, folder_path, thick_keywords=("SPATU", "MTU"),linewidth_thick=1.2,linewidth_thin=0.6)
    
    fig.savefig(
        out,
        dpi=dpi,
        transparent=True,
        bbox_inches="tight",
        pad_inches=0.02
    )
    plt.close(fig)
    
    
    print("[running]: plotting topo data finished")
# -------------------------
# 经度 wrap 到 [-180,180]
# -------------------------
def wrap180(lon):
    return ((lon + 180) % 360) - 180


# -------------------------
# 展开经度（解决跨180）
# -------------------------
def unwrap_lon(lon, lon_ref):
    lon = np.asarray(lon)
    return lon_ref + ((lon - lon_ref + 360) % 360)


# ============================================================
# 主绘图函数
# ============================================================
def plot_lonlat_rect(
    lon_min,
    lon_max,
    lat_min,
    lat_max,
    da,
    dpi=300,
    cmap="jet",
    vmin=None,
    vmax=None,
    cbar_label="",
    lon_tick=10,
    lat_tick=10
):

    # ---------- 经度统一 ----------
    da2 = da.copy()
    da2 = da2.assign_coords(lon=wrap180(da2.lon)).sortby("lon")

    # ---------- 纬度顺序 ----------
    lat_vals = da2.lat.values
    if lat_vals[0] > lat_vals[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)

    # ---------- 跨180处理 ----------
    cross = lon_min > lon_max
    if cross:
        lon_max = lon_max + 360

    lon_vals = unwrap_lon(da2.lon.values, lon_min)

    da2 = da2.assign_coords(lon_unwrap=("lon", lon_vals))
    da2 = da2.swap_dims({"lon": "lon_unwrap"}).sortby("lon_unwrap")

    da_cut = da2.sel(lon_unwrap=slice(lon_min, lon_max), lat=lat_slice)

    lon = da_cut.lon_unwrap.values
    lat = da_cut.lat.values
    Z = da_cut.values

    # ---------- 画图 ----------
    fig, ax = plt.subplots(figsize=(5,4), dpi=dpi)

    pm = ax.pcolormesh(
        lon,
        lat,
        Z,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    # ---------- 经纬度比例 ----------
    ax.set_aspect("equal", adjustable="box")

    # ---------- 坐标格式 ----------
    def format_lon(x, pos):
        x = wrap180(x)
        if x > 0:
            return f"{abs(x):.0f}°E"
        elif x < 0:
            return f"{abs(x):.0f}°W"
        else:
            return "0°"

    def format_lat(y, pos):
        if y > 0:
            return f"{abs(y):.0f}°N"
        elif y < 0:
            return f"{abs(y):.0f}°S"
        else:
            return "0°"

    ax.xaxis.set_major_locator(mticker.MultipleLocator(lon_tick))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(lat_tick))

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(format_lon))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(format_lat))

    # ---------- colorbar ----------
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="4%", pad=0.35)

    cbar = plt.colorbar(pm, cax=cax, orientation="horizontal")

    if cbar_label != "":
        cbar.set_label(cbar_label)

    return fig, ax, cbar


# ============================================================
# 球面圆函数
# ============================================================
def plot_spherical_circle(
    ax,
    lon0,
    lat0,
    radius_km,
    lon_ref,
    color="black",
    linewidth=1.5,
    linestyle="-",
    n=360,
    moon_radius_km=1737.4
):

    lon1 = np.radians(lon0)
    lat1 = np.radians(lat0)

    delta = radius_km / moon_radius_km

    bearings = np.linspace(0, 2*np.pi, n)

    lat_circle = np.arcsin(
        np.sin(lat1)*np.cos(delta)
        + np.cos(lat1)*np.sin(delta)*np.cos(bearings)
    )

    lon_circle = lon1 + np.arctan2(
        np.sin(bearings)*np.sin(delta)*np.cos(lat1),
        np.cos(delta) - np.sin(lat1)*np.sin(lat_circle)
    )

    lat_circle = np.degrees(lat_circle)
    lon_circle = np.degrees(lon_circle)

    lon_circle = wrap180(lon_circle)

    # 展开到同一坐标系
    lon_circle = unwrap_lon(lon_circle, lon_ref)

    ax.plot(
        lon_circle,
        lat_circle,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle
    )

# 下面是剖面
def _lonlat_to_unitvec(lon, lat):
    """lon/lat (deg) -> unit vector (x,y,z)"""
    lonr = np.deg2rad(lon)
    latr = np.deg2rad(lat)
    x = np.cos(latr) * np.cos(lonr)
    y = np.cos(latr) * np.sin(lonr)
    z = np.sin(latr)
    return np.array([x, y, z], dtype=float)

def _unitvec_to_lonlat(v):
    """unit vector -> lon/lat (deg)"""
    x, y, z = v
    lon = np.rad2deg(np.arctan2(y, x))
    lat = np.rad2deg(np.arctan2(z, np.sqrt(x*x + y*y)))
    return lon, lat

def sample_greatcircle_AB(lonA, latA, lonB, latB, n=200):
    """
    生成 A->B 的大圆路径点 (lon, lat)，lon 输出会 wrap180 到 [-180,180)
    """
    # 先 wrap 到 [-180,180) 保持一致
    lonA = wrap180(lonA)
    lonB = wrap180(lonB)

    u = _lonlat_to_unitvec(lonA, latA)
    v = _lonlat_to_unitvec(lonB, latB)

    # 夹角
    dot = np.clip(np.dot(u, v), -1.0, 1.0)
    omega = np.arccos(dot)

    if omega < 1e-12:
        # A,B 很近：直接线性返回
        lons = np.linspace(lonA, lonB, n)
        lats = np.linspace(latA, latB, n)
        return wrap180(lons), lats

    t = np.linspace(0.0, 1.0, n)
    sin_omega = np.sin(omega)

    # slerp
    pts = (np.sin((1 - t) * omega)[:, None] * u[None, :] +
           np.sin(t * omega)[:, None] * v[None, :]) / sin_omega

    lons = np.empty(n)
    lats = np.empty(n)
    for i in range(n):
        lon_i, lat_i = _unitvec_to_lonlat(pts[i])
        lons[i] = lon_i
        lats[i] = lat_i

    lons = wrap180(lons)
    return lons, lats

def plot_AB_on_lonlat_rect(
    ax,
    lonA, latA,
    lonB, latB,
    lon_ref,
    n=300,
    line_color="red",
    line_width=2.0,
    marker_size=40,
    marker_color="yellow",
    edge_color="black",
    labelA="A",
    labelB="B",
    text_offset=(0.6, 0.6),   # (dx_deg, dy_deg) 在图坐标里的偏移
    zorder_line=20,
    zorder_pts=30,
):
    """
    在 plot_lonlat_rect 的坐标系里画 A->B 大圆线，并标注 A/B 点。
    关键：lon_ref 要和 plot_lonlat_rect 用的一致（通常传 lon_min）。
    """
    # 1) 采样大圆路径
    lons, lats = sample_greatcircle_AB(lonA, latA, lonB, latB, n=n)

    # 2) unwrap 到与底图一致的连续经度轴
    lons_u = unwrap_lon(lons, lon_ref)

    # 3) 画线
    ax.plot(
        lons_u, lats,
        color=line_color,
        linewidth=line_width,
        zorder=zorder_line
    )

    # 4) 画点（A/B）
    lonA_u = unwrap_lon(np.array([wrap180(lonA)]), lon_ref)[0]
    lonB_u = unwrap_lon(np.array([wrap180(lonB)]), lon_ref)[0]

    ax.scatter(
        [lonA_u, lonB_u], [latA, latB],
        s=marker_size,
        c=marker_color,
        edgecolors=edge_color,
        linewidths=1.0,
        zorder=zorder_pts
    )

    # 5) 标注文字（加描边，任何底图都清晰）
    dx, dy = text_offset
    txtA = ax.text(lonA_u + dx, latA + dy, labelA, fontsize=10, weight="bold",
                   color="black", zorder=zorder_pts+1)
    txtB = ax.text(lonB_u + dx, latB + dy, labelB, fontsize=10, weight="bold",
                   color="black", zorder=zorder_pts+1)

    for t in (txtA, txtB):
        t.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])

    return ax

def _haversine_km(lon1, lat1, lon2, lat2, R_km=1737.4):
    """球面两点距离（km），默认月球半径 1737.4 km"""
    lon1, lat1, lon2, lat2 = map(np.deg2rad, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    # ✅ wrap 到 [-pi, pi]，避免跨 180° 时走“长路”
    dlon = (dlon + np.pi) % (2*np.pi) - np.pi
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    c = 2*np.arcsin(np.sqrt(a))
    return R_km * c

def _profile_distance_km(lons, lats, R_km=1737.4):
    """沿路径累计距离（km）"""
    dist = np.zeros(len(lons), dtype=float)
    for i in range(1, len(lons)):
        dist[i] = dist[i-1] + _haversine_km(lons[i-1], lats[i-1], lons[i], lats[i], R_km=R_km)
    return dist

def _interp_track(da, lons, lats, lon_name="lon", lat_name="lat"):
    """沿轨迹插值采样，支持跨180度经线；返回 numpy"""
    pts_lon = xr.DataArray(lons, dims="p")
    pts_lat = xr.DataArray(lats, dims="p")

    # 如果经度维不存在，就直接按原逻辑
    if lon_name not in da.dims and lon_name not in da.coords:
        return da.interp({lat_name: pts_lat}).values

    lon_coord = da[lon_name]

    # 只在经度是“度”的场景扩展；并假设经度是规则递增坐标（常见栅格是这样的）
    # 扩展：拼接 lon-360, lon, lon+360 三份，覆盖跨180/0的插值
    da_m360 = da.assign_coords({lon_name: lon_coord - 360})
    da_p360 = da.assign_coords({lon_name: lon_coord + 360})
    da_ext = xr.concat([da_m360, da, da_p360], dim=lon_name).sortby(lon_name)

    return da_ext.interp({lon_name: pts_lon, lat_name: pts_lat}).values

def plot_profile_AB_two_panels_nature_cn(
    A, B,
    topo_xr,
    frequency_thick_da,
    spatial_thick_da,
    freeair_xr,
    ba_xr,
    n=600,
    moon_radius_km=1737.4,
    topo_scale=1e-3,
    thick_scale=1e-3,
    gravity_scale=1.0,
    figsize=(10, 5.8),
    dpi=300,
    fill_topo=False,
    fill_crust=True,
    title=None,
    lon_name="lon",
    lat_name="lat",
    annotate_AB=True,
    panel_labels=True,          # (a)(b)
    x_tick_km=200,              # 距离主刻度
):
    """
    Nature-ish：上下两幅 sharex，黑白线型，无框图例，更克制的网格与轴线。
    """

    lonA, latA = A
    lonB, latB = B

    # ---------- 1) 沿大圆采样 ----------
    lons, lats = sample_greatcircle_AB(lonA, latA, lonB, latB, n=n)
    lons = np.rad2deg(np.unwrap(np.deg2rad(lons)))
    dist_km = _profile_distance_km(lons, lats, R_km=moon_radius_km)

    # ---------- 2) 插值取样 ----------
    topo = _interp_track(topo_xr, lons, lats, lon_name=lon_name, lat_name=lat_name) * topo_scale

    thick_f = _interp_track(frequency_thick_da, lons, lats, lon_name=lon_name, lat_name=lat_name) * thick_scale
    thick_s = _interp_track(spatial_thick_da,   lons, lats, lon_name=lon_name, lat_name=lat_name) * thick_scale
    thick_f = np.nan_to_num(thick_f, nan=0.0)
    thick_s = np.nan_to_num(thick_s, nan=0.0)

    freeair = _interp_track(freeair_xr, lons, lats, lon_name=lon_name, lat_name=lat_name) * gravity_scale
    ba      = _interp_track(ba_xr,      lons, lats, lon_name=lon_name, lat_name=lat_name) * gravity_scale

    # ---------- 3) 界面高度 ----------
    moho_f = topo - thick_f
    moho_s = topo - thick_s

    # ---------- 4) 画图 ----------
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=figsize, dpi=dpi, sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.4], "hspace": 0.03}
    )

    # 全局风格：更像 Nature
    for ax in (ax1, ax2):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.0)
        ax.spines["bottom"].set_linewidth(1.0)
        ax.tick_params(direction="out", length=3.2, width=1.0, labelsize=9)
        # 克制网格：只留 y 方向淡网格，x 方向不画（更干净）
        ax.grid(True, axis="y", which="major", alpha=0.18, linewidth=0.7)
        ax.grid(False, axis="x")

    # 让两幅图像一个整体：去掉上图的 bottom spine、下图的 top spine
    ax1.spines["bottom"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    # 上图不显示 x 刻度（只底部显示）
    ax1.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # x 轴主刻度（km）
    ax2.xaxis.set_major_locator(mticker.MultipleLocator(x_tick_km))

    # ---------- 上图：高度/界面 ----------
    if fill_topo:
        base = np.nanmin(topo) - 2.0
        ax1.fill_between(dist_km, base, topo, alpha=0.12, linewidth=0)

    if fill_crust:
        ax1.fill_between(dist_km, moho_s, topo, alpha=0.08, linewidth=0)

    # 黑白线：靠线型 + 线宽区分
    ax1.plot(dist_km, topo,   color="k", lw=2.1, ls="-",  label="地形")
    ax1.plot(dist_km, moho_f, color="k", lw=1.15, ls="--", label="Moho（频率域）")
    ax1.plot(dist_km, moho_s, color="k", lw=1.15, ls="-.", label="Moho（空间域）")

    ax1.set_ylabel("高度 / 界面高度（km）", fontsize=10)
    
    xmin = dist_km.min()
    xmax = dist_km.max()

    ax1.set_xlim(xmin, xmax)

    # 无框图例（Nature 常见）
    #ax1.legend(loc="upper right", frameon=False, fontsize=9, handlelength=2.8)

    # 标注 A/B（放在上图）
    if annotate_AB:
        ax1.text(dist_km[0],  topo[0],  "A", va="bottom", ha="left",  fontsize=10, weight="bold")
        ax1.text(dist_km[-1], topo[-1], "B", va="bottom", ha="right", fontsize=10, weight="bold")

    # subpanel label (a)
    if panel_labels:
        ax1.text(0.01, 0.96, "(a)", transform=ax1.transAxes,
                 ha="left", va="top", fontsize=10, weight="bold")

    # ---------- 下图：重力 ----------
    ax2.plot(dist_km, freeair, color="k", lw=1.1, ls="--", label="自由空气重力异常")
    ax2.plot(dist_km, ba,      color="k", lw=1.3, ls="-",  label="布格重力异常")

    ax2.set_ylabel("重力异常（mGal）", fontsize=10)
    ax2.set_xlabel("距离（km）", fontsize=10)
    #ax2.legend(loc="upper right", frameon=False, fontsize=9, handlelength=2.8)

    if panel_labels:
        ax2.text(0.01, 0.96, "(b)", transform=ax2.transAxes,
                 ha="left", va="top", fontsize=10, weight="bold")

    # ---------- 标题 ----------
    if title is None:
        title = f"AB 剖面：A({wrap180(lonA):.2f}°, {latA:.2f}°) → B({wrap180(lonB):.2f}°, {latB:.2f}°)"
    fig.suptitle(title, fontsize=11, y=0.98)

    # 更紧凑一点（Nature 常见）
    fig.subplots_adjust(top=0.90, left=0.09, right=0.91, bottom=0.10)

    out = {
        "dist_km": dist_km,
        "lon": lons,
        "lat": lats,
        "topo": topo,
        "moho_freq": moho_f,
        "moho_spatial": moho_s,
        "freeair": freeair,
        "bouguer": ba,
    }
    return fig, (ax1, ax2), out