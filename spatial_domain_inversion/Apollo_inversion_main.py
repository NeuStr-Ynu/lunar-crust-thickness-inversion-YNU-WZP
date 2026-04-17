import pickle
import numpy as np
import boule as bl
import pandas as pd
import harmonica as hm
import matplotlib
matplotlib.use('Agg')   # 非交互后端，禁止弹窗，不影响 savefig
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
from mohoinv_development_7 import MohoLayer, MohoInversion
import xarray as xr
from pathlib import Path

# ── 中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei',
                                           'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── Nature 风格 rcParams（全局可复用，各画图块用 rc_context 引入）
_rc_nature = {
    "font.family"       : "sans-serif",
    "font.sans-serif"   : ["Arial", "Helvetica", "SimHei",
                           "Microsoft YaHei", "DejaVu Sans"],
    "font.size"         : 7,
    "axes.titlesize"    : 7,
    "axes.labelsize"    : 7,
    "xtick.labelsize"   : 6,
    "ytick.labelsize"   : 6,
    "legend.fontsize"   : 6,
    "legend.framealpha" : 0.85,
    "legend.edgecolor"  : "0.75",
    "legend.borderpad"  : 0.4,
    "axes.linewidth"    : 0.5,
    "xtick.major.width" : 0.5,
    "ytick.major.width" : 0.5,
    "xtick.major.size"  : 2,
    "ytick.major.size"  : 2,
    "lines.linewidth"   : 0.8,
    "patch.linewidth"   : 0.5,
    "figure.facecolor"  : "white",
    "axes.facecolor"    : "white",
    "savefig.dpi"       : 300,
}

def _panel_tag(ax, tag, x=0.022, y=0.968):
    """Panel label (a–f) — white text on dark background, top-left corner."""
    ax.text(x, y, tag, transform=ax.transAxes,
            fontsize=8, fontweight="bold",
            va="top", ha="left", color="white",
            bbox=dict(boxstyle="round,pad=0.15",
                      facecolor="0.15", edgecolor="none", alpha=0.75))

def _sym_lim(data):
    """Symmetric vmin/vmax centred on mean, bounded by 1st/99th percentile."""
    center = float(np.nanmean(data))
    half   = max(abs(np.nanpercentile(data,  1) - center),
                 abs(np.nanpercentile(data, 99) - center))
    return center - half, center + half

# ── 模式开关
TEST_MODE = True   # True: 降采样快速验证；False: 全分辨率运行

# ── 固定参数
height = 11e3
R_moon = bl.Moon2015.mean_radius

# ── Apollo 地震约束
APOLLO15_LON   =   3.634   # °E
APOLLO15_LAT   =  26.132   # °N
APOLLO15_THICK =  29.4e3   # m  (Kim et al., 2025)

APOLLO16_LON   =  15.498   # °E
APOLLO16_LAT   =  -8.973   # °N
APOLLO16_THICK =  41.0e3   # m

APOLLO12_LON   = -23.422   # °E
APOLLO12_LAT   =  -3.012   # °N
APOLLO12_THICK =  32.0e3   # m

# 列表形式，方便批量传入 evaluate_seismic
SEISMIC_LONS    = [APOLLO15_LON,   APOLLO16_LON,   APOLLO12_LON  ]
SEISMIC_LATS    = [APOLLO15_LAT,   APOLLO16_LAT,   APOLLO12_LAT  ]
SEISMIC_THICKS  = [APOLLO15_THICK, APOLLO16_THICK, APOLLO12_THICK]
SEISMIC_LABELS  = ["Apollo 15",    "Apollo 16",    "Apollo 12"   ]

# ══════════════════════════════════════════════════════════════════════════════
# 读取数据
# ══════════════════════════════════════════════════════════════════════════════
print('reading data ...')

PROJECT_ROOT = Path(__file__).parent.parent
data_filename=PROJECT_ROOT/"data/density_no_mare_n3000_f3050_719.sh"
density_output=PROJECT_ROOT/"data/density_no_mare_n3000_f3050_719_DH2.txt"
result_filename=PROJECT_ROOT/"result/frequency_domain_run1.pkl"

df         = pd.read_csv(PROJECT_ROOT/'data/boueguer_frenquency_11km.csv')
df_density = pd.read_csv(PROJECT_ROOT/'data/density_absolute.csv')

# 经度统一转换为 -180~180（防止 CSV 使用 0-360 格式导致负经度丢失）
df['lon']         = np.where(df['lon']         > 180, df['lon']         - 360, df['lon'])
df_density['lon'] = np.where(df_density['lon'] > 180, df_density['lon'] - 360, df_density['lon'])

# 坐标取整到固定小数位，消除两个 CSV 之间的浮点精度差异
_DP = 8
for _df in (df, df_density):
    _df['lon'] = _df['lon'].round(_DP)
    _df['lat'] = _df['lat'].round(_DP)

lon = np.sort(df['lon'].unique())
lat = np.sort(df['lat'].unique())

if TEST_MODE:
    dlat_orig = float(lat[1] - lat[0])
    dlon_orig = float(lon[1] - lon[0])
    step_lat  = max(1, round(1 / dlat_orig))
    step_lon  = max(1, round(1 / dlon_orig))
    lat = lat[::step_lat]
    lon = lon[::step_lon]
    print(f"[TEST MODE] 降采样: {dlat_orig*step_lat:.2f}°×{dlon_orig*step_lon:.2f}°  "
          f"lat n={len(lat)}, lon n={len(lon)}")
else:
    dlat_orig = float(lat[1] - lat[0])
    dlon_orig = float(lon[1] - lon[0])
    step_lat  = max(1, round(0.25 / dlat_orig))
    step_lon  = max(1, round(0.25 / dlon_orig))
    lat = lat[::step_lat]
    lon = lon[::step_lon]
    print(f"[TEST MODE] 降采样: {dlat_orig*step_lat:.2f}°×{dlon_orig*step_lon:.2f}°  "
          f"lat n={len(lat)}, lon n={len(lon)}")

print(f"lon: {lon[0]:.2f} ~ {lon[-1]:.2f},  n={len(lon)}")
print(f"lat: {lat[0]:.2f} ~ {lat[-1]:.2f},  n={len(lat)}")

go          = df.pivot(index='lat', columns='lon', values='deltaN') \
                .loc[lat, lon].values.astype(float)
topo        = df.pivot(index='lat', columns='lon', values='topo') \
                .loc[lat, lon].values.astype(float)
density_raw = df_density.pivot(index='lat', columns='lon', values='density') \
                         .loc[lat, lon].values.astype(float)

ba_xr      = xr.DataArray(go,          dims=('lat','lon'), coords={'lat': lat,'lon': lon})
topo_xr    = xr.DataArray(topo,        dims=('lat','lon'), coords={'lat': lat,'lon': lon})
density_xr = xr.DataArray(density_raw, dims=('lat','lon'), coords={'lat': lat,'lon': lon})

# ══════════════════════════════════════════════════════════════════════════════
# 裁剪区域（Apollo 15 周围）
# ══════════════════════════════════════════════════════════════════════════════
print("裁剪中....")
dlat = float(lat[1] - lat[0])
dlon = float(lon[1] - lon[0])
lat_lo, lat_hi = -15, 35   # 覆盖 Apollo 12/15/16 全部站点
lon_lo, lon_hi = -30, 20  # Apollo 12 约 -23°，Apollo 16 约 +15°

_test_clip = ba_xr.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
if _test_clip.shape[0] % 2 == 0:
    lat_hi += dlat
if _test_clip.shape[1] % 2 == 0:
    lon_hi += dlon

ba_mare      = ba_xr.sel(     lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
topo_mare    = topo_xr.sel(   lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))
density_mare = density_xr.sel(lat=slice(lat_lo, lat_hi), lon=slice(lon_lo, lon_hi))

data_shape = ba_mare.shape
print(f"裁切完成，共 {data_shape} 个网格点")
print(f"经度范围: {float(ba_mare.lon.min()):.2f} ~ {float(ba_mare.lon.max()):.2f}")
print(f"纬度范围: {float(ba_mare.lat.min()):.2f} ~ {float(ba_mare.lat.max()):.2f}")

# 重新生成精确等间距坐标，消除浮点舍入导致 harmonica 报错
lon_clipped = np.linspace(float(ba_mare.lon[0]),  float(ba_mare.lon[-1]),  len(ba_mare.lon))
lat_clipped = np.linspace(float(ba_mare.lat[0]),  float(ba_mare.lat[-1]),  len(ba_mare.lat))
ba_mare      = ba_mare.assign_coords(lon=lon_clipped, lat=lat_clipped)
topo_mare    = topo_mare.assign_coords(lon=lon_clipped, lat=lat_clipped)
density_mare = density_mare.assign_coords(lon=lon_clipped, lat=lat_clipped)

LON, LAT    = np.meshgrid(ba_mare.lon, ba_mare.lat)
coords_full = (LON, LAT, (R_moon + height) * np.ones_like(LON))

# 三个 Apollo 站点地形高度（插值，m 相对平均球半径）
apollo15_topo = float(topo_mare.interp(
    lat=APOLLO15_LAT, lon=APOLLO15_LON, method="linear").values)
apollo16_topo = float(topo_mare.interp(
    lat=APOLLO16_LAT, lon=APOLLO16_LON, method="linear").values)
apollo12_topo = float(topo_mare.interp(
    lat=APOLLO12_LAT, lon=APOLLO12_LON, method="linear").values)
SEISMIC_TOPOS = [apollo15_topo, apollo16_topo, apollo12_topo]
print(f"Apollo 15 地形高度: {apollo15_topo:.0f} m")
print(f"Apollo 16 地形高度: {apollo16_topo:.0f} m")
print(f"Apollo 12 地形高度: {apollo12_topo:.0f} m")

# 横向密度异常（无量纲基础量，循环内按孔隙度缩放）
weights             = np.cos(np.deg2rad(density_mare.lat))
density_anomaly_raw = density_mare - density_mare.weighted(weights).mean(("lat", "lon"))

# ══════════════════════════════════════════════════════════════════════════════
# 第一步：μ 寻优（训练/测试集交叉验证，与 main.py 策略一致）
#   训练集：[1::2, 1::2]  反演
#   测试集：[::2,  ::2]   forward_testset 评分（测试集 RMS 最小为最优）
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("第一步：μ 寻优 (1e-10 ~ 1e-2，对数间隔 10 个，训练/测试集交叉验证)")
print("=" * 60)

MU_TRIAL_RHO = 3220.0
MU_TRIAL_CT  = 35e3
MU_TRIAL_POR = 0.12

# 试验参数下的密度校正（全网格）
da_trial = (1.0 - MU_TRIAL_POR) * density_anomaly_raw
layer_t  = hm.tesseroid_layer(
    (ba_mare.lon.values, ba_mare.lat.values),
    surface  = (R_moon - MU_TRIAL_CT) * np.ones(data_shape),
    reference= R_moon                  * np.ones(data_shape),
    properties={"density": da_trial.values},
)
g_t    = layer_t.tesseroid_layer.gravity(coords_full, field="g_z", progressbar=False)
g_t_da = xr.DataArray(g_t, dims=("lat","lon"),
                       coords={"lat": ba_mare.lat.values, "lon": ba_mare.lon.values})
ba_trial = ba_mare - g_t_da

# ── 训练/测试集划分
ba_train_mu = ba_trial[1::2, 1::2]   # 训练集（奇行奇列）
ba_test_mu  = ba_trial[::2,  ::2]    # 测试集（偶行偶列）

test_LON_mu = LON[::2, ::2]
test_LAT_mu = LAT[::2, ::2]

# 训练集密度对比度与参考面
crust_dens_t_train = density_mare.values[1::2, 1::2] * (1.0 - MU_TRIAL_POR)
dc_t_train         = MU_TRIAL_RHO - crust_dens_t_train
ref_t_train        = -MU_TRIAL_CT * np.ones_like(dc_t_train)

mu_vals  = np.logspace(-10, -2, 10)
mu_rms   = {}   # key: mu_val  value: 测试集 RMS (mGal)

for mu_val in mu_vals:
    ml = MohoLayer(
        lon=ba_train_mu.lon.values, lat=ba_train_mu.lat.values,
        height=height, reference=ref_t_train,
        surface=np.zeros_like(ref_t_train),
        density_contrast=dc_t_train,
    )
    iv = MohoInversion(ml, ba_train_mu.values, max_iter=20, mu=mu_val, quite=True)
    iv.inversion_gn(gtol=0.05)

    # 测试集正演并计算 RMS
    g_pred = iv.layer.forward_testset(
        test_LON_mu.ravel(), test_LAT_mu.ravel(), quite=True)
    rms_test = float(np.sqrt(np.mean(
        (g_pred - ba_test_mu.values.ravel()) ** 2)))
    mu_rms[mu_val] = rms_test
    print(f"  μ={mu_val:.2e}  测试集 RMS={rms_test:.4f} mGal")

best_mu = min(mu_rms, key=mu_rms.get)
print(f"\n最优 μ = {best_mu:.2e}  (测试集 RMS={mu_rms[best_mu]:.4f} mGal)")

# μ 寻优图
with matplotlib.rc_context(_rc_nature):
    fig_mu, ax_mu = plt.subplots(figsize=(6, 3.5))
    ax_mu.semilogx(mu_vals, [mu_rms[v] for v in mu_vals],
                   marker='o', color='steelblue', linewidth=1.2, markersize=5)
    ax_mu.axvline(best_mu, color='#CC0000', linestyle='--', linewidth=1,
                  label=f'Optimal $\\mu$={best_mu:.2e}\nRMS={mu_rms[best_mu]:.4f} mGal')
    ax_mu.set_xlabel('Regularization parameter $\\mu$')
    ax_mu.set_ylabel('Test-set RMS (mGal)')
    ax_mu.set_title('$\\mu$ optimization (train/test cross-validation)')
    ax_mu.legend(fontsize=6)
    ax_mu.grid(True, which='both', alpha=0.35, linewidth=0.4)
    fig_mu.tight_layout()
    fig_mu.savefig("mu_optimization.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close('all')

# ══════════════════════════════════════════════════════════════════════════════
# 第二步：三参数联合扫描（孔隙度 × 月幔密度 × 参考厚度）→ Apollo 15 地震约束评分
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("第二步：三参数联合扫描（孔隙度 × 月幔密度 × 参考厚度）")
print("=" * 60)

POROSITY_VALS    = np.array([0.07])
MANTLE_RHO_VALS  = np.array(range(3150,3401,50))  # kg/m³，可按需调整
crust_thick_vals = np.arange(35e3, 46e3, 1e3)                   # 30~40 km，共 11 个

total_runs = len(POROSITY_VALS) * len(MANTLE_RHO_VALS) * len(crust_thick_vals)
done = 0

# results[(por, rho_m, ct_m)] = mae (m)
results = {}

for por in POROSITY_VALS:
    # 孔隙度固定时，密度异常只需缩放一次
    density_anomaly_crust = (1.0 - por) * density_anomaly_raw

    for rho_m in MANTLE_RHO_VALS:
        # 密度对比度（不随 ct 变化）
        density_contrast = rho_m - density_mare.values * (1.0 - por)

        for ct in crust_thick_vals:
            done += 1
            print(f"\n[{done}/{total_runs}]  φ={por}  ρ_m={rho_m:.0f} kg/m³  "
                  f"H₀={ct/1e3:.0f} km")

            # 横向密度重力校正
            layer_dens = hm.tesseroid_layer(
                (ba_mare.lon.values, ba_mare.lat.values),
                surface  = (R_moon - ct) * np.ones(data_shape),
                reference= R_moon        * np.ones(data_shape),
                properties={"density": density_anomaly_crust.values},
            )
            density_g    = layer_dens.tesseroid_layer.gravity(
                coords_full, field="g_z", progressbar=False)
            density_g_da = xr.DataArray(
                density_g, dims=("lat", "lon"),
                coords={"lat": ba_mare.lat.values, "lon": ba_mare.lon.values})
            ba_corrected = ba_mare - density_g_da

            # 反演
            reference = -ct * np.ones_like(density_contrast)
            moho_layer = MohoLayer(
                lon=ba_corrected.lon.values, lat=ba_corrected.lat.values,
                height=height, reference=reference,
                surface=np.zeros_like(reference),
                density_contrast=density_contrast,
            )
            inv = MohoInversion(
                moho_layer, ba_corrected.values, max_iter=20, mu=best_mu, quite=True)
            inv.inversion_gn(gtol=0.05)

            # 三站点地震约束评分（Apollo 12 / 15 / 16 均值 MAE）
            _, _, scores = inv.evaluate_seismic(
                SEISMIC_LONS, SEISMIC_LATS, SEISMIC_THICKS, SEISMIC_TOPOS)
            results[(por, rho_m, round(ct, 1))] = scores['mae']
            print(f"  → 三站点均值 MAE = {scores['mae']/1e3:.3f} km")

# —— 找最优参数组合
best_key = min(results, key=results.get)
best_por, best_rho_m, best_ct_m = best_key
print(f"\n最优参数: φ={best_por}  ρ_m={best_rho_m:.0f} kg/m³  "
      f"H₀={best_ct_m/1e3:.0f} km  "
      f"(Apollo 15 MAE={results[best_key]/1e3:.3f} km)")

# —— 保存交叉验证 pkl
pkl_data = {
    "porosity_vals"   : POROSITY_VALS,
    "mantle_rho_vals" : MANTLE_RHO_VALS,
    "crust_thick_vals": crust_thick_vals,
    "results_dict"    : results,
    "best_mu"         : best_mu,
    "best_por"        : best_por,
    "best_rho_m"      : best_rho_m,
    "best_ct_m"       : best_ct_m,
    "seismic_lons"    : SEISMIC_LONS,
    "seismic_lats"    : SEISMIC_LATS,
    "seismic_thicks"  : SEISMIC_THICKS,
    "seismic_labels"  : SEISMIC_LABELS,
    "apollo15_thick"  : APOLLO15_THICK,
    "mu_rms"          : mu_rms,          # μ 寻优评分字典 {mu: test_rms}
    "mu_vals"         : mu_vals,         # μ 候选值数组（有序）
}
with open(PROJECT_ROOT/"result/crosstest_result.pkl", "wb") as f:
    pickle.dump(pkl_data, f)
print("交叉验证结果已保存至 crosstest_result.pkl")

# —— 三参数扫描结果图（每个孔隙度一个子图，热图：ρ_m × H₀）
with matplotlib.rc_context(_rc_nature):
    n_por = len(POROSITY_VALS)
    fig_cv, axes_cv = plt.subplots(
        1, n_por, figsize=(4.5 * n_por, 4.2),
        gridspec_kw={"wspace": 0.28})
    if n_por == 1:
        axes_cv = [axes_cv]

    for col, por in enumerate(POROSITY_VALS):
        # 组装 MAE 矩阵：行=ρ_m，列=H₀
        mae_mat = np.array([
            [results[(por, rho_m, round(ct, 1))] / 1e3
             for ct in crust_thick_vals]
            for rho_m in MANTLE_RHO_VALS
        ])   # shape: (n_rho, n_ct)

        ax = axes_cv[col]
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.5)
        ax.spines["bottom"].set_linewidth(0.5)
        ax.tick_params(width=0.5, length=2)

        im = ax.pcolormesh(
            crust_thick_vals / 1e3, MANTLE_RHO_VALS, mae_mat,
            cmap="viridis_r", shading="auto")
        cb = fig_cv.colorbar(im, ax=ax, shrink=0.85, aspect=18, pad=0.03)
        cb.set_label("Mean MAE (A12/15/16, km)", fontsize=6)
        cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5)
        cb.outline.set_linewidth(0.4)

        # 标记最优点
        best_row = np.where(MANTLE_RHO_VALS == best_rho_m)[0]
        best_col = np.where(np.isclose(crust_thick_vals, best_ct_m))[0]
        if por == best_por and len(best_row) and len(best_col):
            ax.plot(best_ct_m / 1e3, best_rho_m,
                    marker="*", color="#FF4444", markersize=10, zorder=5,
                    label=f"Optimal\n$H_0$={best_ct_m/1e3:.0f} km\n"
                          f"$\\rho_m$={best_rho_m:.0f}\n"
                          f"MAE={results[best_key]/1e3:.3f} km")
            ax.legend(fontsize=5.5, loc="upper right",
                      framealpha=0.85, handlelength=0.6)

        ax.set_xlabel("Reference crustal thickness $H_0$ (km)")
        ax.set_ylabel("Mantle density $\\rho_m$ (kg m$^{-3}$)")
        ax.set_title(f"Porosity $\\phi$ = {por}", pad=3)
        ax.set_yticks(MANTLE_RHO_VALS)
        _panel_tag(ax, chr(ord("a") + col))

    fig_cv.suptitle(
        f"Parameter search — Apollo 12={APOLLO12_THICK/1e3:.0f} km  "
        f"Apollo 15={APOLLO15_THICK/1e3:.1f} km  "
        f"Apollo 16={APOLLO16_THICK/1e3:.0f} km  (mean MAE)",
        fontsize=8, fontweight="semibold", y=1.02)
    fig_cv.savefig("crosstest_result.png", dpi=300,
                   bbox_inches="tight", facecolor="white")
    plt.close("all")
print("三参数扫描图已保存至 crosstest_result.png")

# ══════════════════════════════════════════════════════════════════════════════
# 第三步：最优参数最终反演
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"第三步：最终反演  φ={best_por}  ρ_m={best_rho_m:.0f} kg/m³  "
      f"H₀={best_ct_m/1e3:.0f} km  μ={best_mu:.2e}")
print("=" * 60)

density_anomaly_best = (1.0 - best_por) * density_anomaly_raw

layer_dens_best = hm.tesseroid_layer(
    (ba_mare.lon.values, ba_mare.lat.values),
    surface  = (R_moon - best_ct_m) * np.ones(data_shape),
    reference= R_moon                * np.ones(data_shape),
    properties={"density": density_anomaly_best.values},
)
g_rho_final = layer_dens_best.tesseroid_layer.gravity(
    coords_full, field="g_z", progressbar=False)

g_rho_da = xr.DataArray(
    g_rho_final, dims=("lat","lon"),
    coords={"lat": ba_mare.lat.values, "lon": ba_mare.lon.values})
ba_final = ba_mare - g_rho_da

crust_density_full    = density_mare.values * (1.0 - best_por)
density_contrast_full = best_rho_m - crust_density_full
reference_full        = -best_ct_m * np.ones_like(density_contrast_full)

moho_layer_final = MohoLayer(
    lon=ba_mare.lon.values, lat=ba_mare.lat.values,
    height=height, reference=reference_full,
    surface=np.zeros_like(reference_full),
    density_contrast=density_contrast_full,
)
inv_final = MohoInversion(
    moho_layer_final, ba_final.values, max_iter=20, mu=best_mu, quite=False)
surface_inv_final = inv_final.inversion_gn(gtol=0.05)

print("\n最终反演地震评分（Apollo 12 / 15 / 16）：")
thickness_inv_all, diff_all, scores_all = inv_final.evaluate_seismic(
    SEISMIC_LONS, SEISMIC_LATS, SEISMIC_THICKS, SEISMIC_TOPOS)
# 单独保留 Apollo 15 变量以兼容后续绘图
thickness_inv_a15 = thickness_inv_all[0:1]
diff_a15          = diff_all[0:1]
scores_a15        = scores_all

# 月壳厚度（km）= 地形高度 - 参考面 - 反演起伏
# topo_mare.values > 0：地表高于平均球；reference_full < 0：Moho 在平均球面以下
crust_thickness = (topo_mare.values - reference_full - surface_inv_final) / 1e3
residuals_final  = ba_final.values - inv_final.layer.forward(quite=True)  # obs − pred（正值=数据大于模型）

# ══════════════════════════════════════════════════════════════════════════════
# 保存最终 pkl
# ══════════════════════════════════════════════════════════════════════════════
final_result = {
    "LON": LON, "LAT": LAT,
    "observe"        : ba_mare.values,
    "g_rho"          : g_rho_final,
    "input_g"        : ba_final.values,
    "crust_density"  : crust_density_full,
    "crust_thickness": crust_thickness,
    "surface_inv"    : surface_inv_final,
    "residuals"      : residuals_final,
    "topo"           : topo_mare.values,          # 地形（m，相对平均球半径）
    "density_anomaly": density_anomaly_best.values, # 横向密度异常（kg/m³）
    "rms"            : np.array(inv_final.rms_history),
    "porosity"       : best_por,
    "mantle_rho"     : best_rho_m,
    "best_ct_m"      : best_ct_m,
    "best_mu"        : best_mu,
    "seismic_lons"      : SEISMIC_LONS,
    "seismic_lats"      : SEISMIC_LATS,
    "seismic_thicks"    : SEISMIC_THICKS,
    "seismic_labels"    : SEISMIC_LABELS,
    "thickness_inv_all" : thickness_inv_all,
    "diff_all"          : diff_all,
    "apollo15_thick_ref": APOLLO15_THICK,
    "apollo15_thick_inv": float(thickness_inv_a15[0]),
    "apollo15_scores"   : scores_all,
}
with open(PROJECT_ROOT/"result/final_result.pkl", "wb") as f:
    pickle.dump(final_result, f)
print("最终结果已保存至 final_result.pkl")

# ══════════════════════════════════════════════════════════════════════════════
# 画图：Nature 风格汇总图（2行×3列）
# ══════════════════════════════════════════════════════════════════════════════

# —— 边缘裁剪（四边各向内 1°）
lon_arr = ba_mare.lon.values
lat_arr = ba_mare.lat.values
lon_inner = (lon_arr >= lon_arr[0] + 1.0) & (lon_arr <= lon_arr[-1] - 1.0)
lat_inner = (lat_arr >= lat_arr[0] + 1.0) & (lat_arr <= lat_arr[-1] - 1.0)
ii_lat = np.where(lat_inner)[0]
ii_lon = np.where(lon_inner)[0]

LON_p = LON[np.ix_(ii_lat, ii_lon)]
LAT_p = LAT[np.ix_(ii_lat, ii_lon)]

def _crop(arr2d):
    return arr2d[np.ix_(ii_lat, ii_lon)]

ct_p       = _crop(crust_thickness)
ba_final_p = _crop(ba_final.values)
topo_p     = _crop(topo_mare.values)
dens_p     = _crop(crust_density_full)
g_rho_p    = _crop(g_rho_final)
resid_p    = _crop(residuals_final)   # obs − pred，正值=数据大于模型

EXTENT_P = [float(LON_p.min()), float(LON_p.max()),
            float(LAT_p.min()), float(LAT_p.max())]

PROJ = ccrs.PlateCarree()

# —— 辅助函数

def _gl(ax, extent, dlon=5, dlat=5, left=True, bottom=True):
    """添加经纬网格线及标注"""
    gl = ax.gridlines(
        crs=PROJ, draw_labels=True, linewidth=0.3,
        color="#777777", alpha=0.55, linestyle="--",
        x_inline=False, y_inline=False,
    )
    gl.top_labels    = False
    gl.right_labels  = False
    gl.left_labels   = left
    gl.bottom_labels = bottom
    gl.xlocator = mticker.FixedLocator(
        np.arange(np.ceil(extent[0] / dlon) * dlon, extent[1] + dlon, dlon))
    gl.ylocator = mticker.FixedLocator(
        np.arange(np.ceil(extent[2] / dlat) * dlat, extent[3] + dlat, dlat))
    gl.xlabel_style = {"size": 5.5, "color": "#333333"}
    gl.ylabel_style = {"size": 5.5, "color": "#333333"}
    return gl

def _topo_cont(ax, topo_2d, lon2d, lat2d, n=6):
    """叠加地形等值线（白色半透明）"""
    vlo = np.nanpercentile(topo_2d, 5)
    vhi = np.nanpercentile(topo_2d, 95)
    ax.contour(lon2d, lat2d, topo_2d,
               levels=np.linspace(vlo, vhi, n),
               colors="white", linewidths=0.25, alpha=0.45, transform=PROJ)

def _mark_apollo(ax):
    """Mark Apollo 12 / 15 / 16 stations"""
    _colors = ["#CC0000", "#0055CC", "#009900"]
    for i, (lo, la, label, seis_t) in enumerate(
            zip(SEISMIC_LONS, SEISMIC_LATS, SEISMIC_LABELS, SEISMIC_THICKS)):
        inv_t = float(thickness_inv_all[i]) / 1e3
        ax.plot(lo, la,
                marker="^", color=_colors[i], markersize=5,
                markeredgewidth=0.5, markeredgecolor="white",
                transform=PROJ, zorder=8,
                label=f"{label}  Inv:{inv_t:.1f} / Seis:{seis_t/1e3:.0f} km")
    ax.legend(loc="upper right", fontsize=5, handlelength=0.7, handletextpad=0.4)

def _cbar(fig, ax, im, label):
    """统一风格的色条"""
    cb = fig.colorbar(im, ax=ax, shrink=0.80, aspect=22,
                      pad=0.03, orientation="vertical")
    cb.set_label(label, fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5)
    cb.outline.set_linewidth(0.4)
    return cb

# ══ 绘图主体 ══

with matplotlib.rc_context(_rc_nature):

    fig, axes = plt.subplots(
        2, 3, figsize=(12.5, 7.8),
        subplot_kw={"projection": PROJ},
        gridspec_kw={"hspace": 0.10, "wspace": 0.06},
    )

    # ── (a) 月壳厚度
    ax = axes[0, 0]
    ax.set_extent(EXTENT_P, crs=PROJ)
    im = ax.pcolormesh(LON_p, LAT_p, ct_p,
                       cmap="viridis_r", shading="auto",
                       transform=PROJ, rasterized=True)
    _topo_cont(ax, topo_p, LON_p, LAT_p)
    _gl(ax, EXTENT_P)
    _mark_apollo(ax)
    _cbar(fig, ax, im, "Crustal thickness (km)")
    ax.set_title(
        f"Crustal thickness\n"
        f"$\\rho_m$={best_rho_m:.0f} kg m$^{{-3}}$, "
        f"$H_0$={best_ct_m/1e3:.0f} km, $\\phi$={best_por}",
        pad=3)
    _panel_tag(ax, "a")

    # ── (b) Density-corrected Bouguer gravity
    ax = axes[0, 1]
    ax.set_extent(EXTENT_P, crs=PROJ)
    vmin_b, vmax_b = _sym_lim(ba_final_p)
    im = ax.pcolormesh(LON_p, LAT_p, ba_final_p,
                       cmap="RdBu_r", vmin=vmin_b, vmax=vmax_b,
                       shading="auto", transform=PROJ, rasterized=True)
    _topo_cont(ax, topo_p, LON_p, LAT_p)
    _gl(ax, EXTENT_P, left=False)
    _cbar(fig, ax, im, "Bouguer gravity (corrected, mGal)")
    ax.set_title("Corrected Bouguer gravity\n(Bouguer $-$ $g_\\rho$)", pad=3)
    _panel_tag(ax, "b")

    # ── (c) Topography
    ax = axes[0, 2]
    ax.set_extent(EXTENT_P, crs=PROJ)
    im = ax.pcolormesh(LON_p, LAT_p, topo_p / 1e3,
                       cmap="gist_earth", shading="auto",
                       transform=PROJ, rasterized=True)
    _gl(ax, EXTENT_P, left=False)
    _cbar(fig, ax, im, "Topography (km)")
    ax.set_title("Topography (rel. mean sphere)", pad=3)
    _panel_tag(ax, "c")

    # ── (d) Crustal density
    ax = axes[1, 0]
    ax.set_extent(EXTENT_P, crs=PROJ)
    im = ax.pcolormesh(LON_p, LAT_p, dens_p,
                       cmap="plasma", shading="auto",
                       transform=PROJ, rasterized=True)
    _topo_cont(ax, topo_p, LON_p, LAT_p)
    _gl(ax, EXTENT_P)
    _cbar(fig, ax, im, "Crustal density (kg m$^{-3}$)")
    ax.set_title(f"Effective crustal density\n"
                 f"$(1-\\phi)\\rho_c$, $\\phi$={best_por:.2f}", pad=3)
    _panel_tag(ax, "d")

    # ── (e) Lateral density gravity correction
    ax = axes[1, 1]
    ax.set_extent(EXTENT_P, crs=PROJ)
    vmin_e, vmax_e = _sym_lim(g_rho_p)
    im = ax.pcolormesh(LON_p, LAT_p, g_rho_p,
                       cmap="RdBu_r", vmin=vmin_e, vmax=vmax_e,
                       shading="auto", transform=PROJ, rasterized=True)
    _gl(ax, EXTENT_P, left=False)
    _cbar(fig, ax, im, "$g_\\rho$ (mGal)")
    ax.set_title("Lateral density gravity correction $g_\\rho$", pad=3)
    _panel_tag(ax, "e")

    # ── (f) Inversion residual (obs − pred)
    ax = axes[1, 2]
    ax.set_extent(EXTENT_P, crs=PROJ)
    vr = float(np.nanpercentile(np.abs(resid_p), 98))
    im = ax.pcolormesh(LON_p, LAT_p, resid_p,
                       cmap="RdBu_r", vmin=-vr, vmax=vr,
                       shading="auto", transform=PROJ, rasterized=True)
    _gl(ax, EXTENT_P, left=False)
    rms_final = float(np.sqrt(np.nanmean(resid_p ** 2)))
    _cbar(fig, ax, im, "Residual (obs $-$ pred, mGal)")
    ax.set_title(f"Inversion residual (obs $-$ pred)\nRMS = {rms_final:.3f} mGal", pad=3)
    _panel_tag(ax, "f")

    # —— Overall title
    _parts = [
        f"{lbl}: inv={float(thickness_inv_all[i])/1e3:.1f} / seis={st/1e3:.0f} km"
        for i, (lbl, st) in enumerate(zip(SEISMIC_LABELS, SEISMIC_THICKS))
    ]
    fig.suptitle(
        "Lunar crustal structure  |  " + "   ".join(_parts),
        fontsize=7.5, fontweight="semibold", y=1.010,
    )

    fig.savefig("final_result.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close("all")

print("图像已保存至 final_result.png")

# ══════════════════════════════════════════════════════════════════════════════
# 密度对比度图（地幔密度 − 月壳有效密度）
# ══════════════════════════════════════════════════════════════════════════════
dc_p = _crop(density_contrast_full)   # Δρ = ρ_m − ρ_crust·(1−φ)

with matplotlib.rc_context(_rc_nature):
    fig_dc, axes_dc = plt.subplots(
        1, 2, figsize=(10, 4.2),
        subplot_kw={"projection": PROJ},
        gridspec_kw={"wspace": 0.06},
    )

    # ── (a) 密度对比度空间分布
    ax = axes_dc[0]
    ax.set_extent(EXTENT_P, crs=PROJ)
    vmin_dc, vmax_dc = _sym_lim(dc_p)
    im = ax.pcolormesh(LON_p, LAT_p, dc_p,
                       cmap="RdBu_r", vmin=vmin_dc, vmax=vmax_dc,
                       shading="auto", transform=PROJ, rasterized=True)
    _topo_cont(ax, topo_p, LON_p, LAT_p)
    _gl(ax, EXTENT_P)
    cb = fig_dc.colorbar(im, ax=ax, shrink=0.82, aspect=22,
                         pad=0.03, orientation="vertical")
    cb.set_label("Density contrast $\\Delta\\rho$ (kg m$^{-3}$)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, width=0.4, length=1.5)
    cb.outline.set_linewidth(0.4)
    ax.set_title(
        f"Density contrast $\\Delta\\rho = \\rho_m - (1-\\phi)\\rho_c$\n"
        f"$\\rho_m$={best_rho_m:.0f} kg m$^{{-3}}$, $\\phi$={best_por}",
        pad=3)
    _panel_tag(ax, "a")

    # ── (b) Density contrast histogram
    ax2 = axes_dc[1]
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["left"].set_linewidth(0.5)
    ax2.spines["bottom"].set_linewidth(0.5)
    ax2.tick_params(width=0.5, length=2)

    dc_flat = dc_p[np.isfinite(dc_p)].ravel()
    counts, edges = np.histogram(dc_flat, bins=40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    ax2.bar(centers, counts, width=edges[1]-edges[0],
            color="steelblue", alpha=0.75, edgecolor="white", linewidth=0.3)
    ax2.axvline(float(np.mean(dc_flat)), color="#CC0000", linewidth=1,
                linestyle="--", label=f"Mean {np.mean(dc_flat):.1f} kg m$^{{-3}}$")
    ax2.axvline(float(np.median(dc_flat)), color="darkorange", linewidth=1,
                linestyle=":", label=f"Median {np.median(dc_flat):.1f} kg m$^{{-3}}$")
    ax2.set_xlabel("Density contrast $\\Delta\\rho$ (kg m$^{-3}$)")
    ax2.set_ylabel("Count")
    ax2.set_title("Density contrast distribution", pad=3)
    ax2.legend(fontsize=6, framealpha=0.8)
    ax2.grid(True, axis="y", alpha=0.3, linewidth=0.4)
    _panel_tag(ax2, "b")

    stats_str = (
        f"min  = {dc_flat.min():.1f}\n"
        f"max  = {dc_flat.max():.1f}\n"
        f"std  = {dc_flat.std():.1f}\n"
        f"n    = {len(dc_flat)}"
    )
    ax2.text(0.97, 0.97, stats_str,
             transform=ax2.transAxes, fontsize=6,
             va="top", ha="right", family="monospace",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat",
                       edgecolor="0.7", alpha=0.8))

    fig_dc.suptitle(
        f"Density contrast  $\\Delta\\rho = \\rho_m - (1-\\phi)\\rho_c$  "
        f"($\\rho_m$={best_rho_m:.0f} kg m$^{{-3}}$, $\\phi$={best_por})",
        fontsize=8, fontweight="semibold", y=1.01)
    fig_dc.savefig("density_contrast.png", dpi=300,
                   bbox_inches="tight", facecolor="white")
    plt.close("all")

print("密度对比度图已保存至 density_contrast.png")
