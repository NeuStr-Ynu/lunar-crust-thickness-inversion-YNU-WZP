"""
月球莫霍面反演 — 实战数据
==========================
数据说明：
    CSV 里的 lon/lat 是单元格中心坐标（0.5, 1.5, ..., 359.5 / -89.5, ..., 89.5）
    直接读取使用，不需要额外插值或取中点。

文件：
    data/boueguer_frenquency_11km_d??km_1deg.csv   布格重力异常（mGal），列名 deltaN
    data/density_1deg.csv                           密度数据，列名 deltaN
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from mohoinv_development_7 import MohoLayer, MohoInversion


# ==============================================================================
# 1. 读取数据
# ==============================================================================
PROJECT_ROOT = Path.cwd().parent

df         = pd.read_csv(PROJECT_ROOT/'data/boueguer_frenquency_11km_d35km_1deg.csv')
df_density = pd.read_csv(PROJECT_ROOT/'data/density_1deg.csv')

# 单元格中心坐标（CSV 里直接存的就是中心）
lon = np.sort(df['lon'].unique())
lat = np.sort(df['lat'].unique())

print(f"lon: {lon[0]:.2f} ~ {lon[-1]:.2f},  n={len(lon)}")
print(f"lat: {lat[0]:.2f} ~ {lat[-1]:.2f},  n={len(lat)}")

# pivot 成 2D 数组，行=lat（升序），列=lon（升序）
go           = df.pivot(index='lat', columns='lon', values='deltaN') \
                 .loc[lat, lon].values.astype(float)          # 观测重力 (mGal)
density_raw  = df_density.pivot(index='lat', columns='lon', values='deltaN') \
                          .loc[lat, lon].values.astype(float) # 原始密度数据
# density_raw = 700*np.ones_like(go)

LON, LAT = np.meshgrid(lon, lat)
shape    = LON.shape

print(f"Grid shape: {shape}  ({shape[0]*shape[1]} parameters)")
print(f"Observed gravity   min={go.min():.2f}  max={go.max():.2f}  "
      f"std={go.std():.2f} mGal")


# ==============================================================================
# 2. 密度对比度（逐点变化）
#    地壳密度  = density_raw * (1 - porosity)
#    地幔密度  = 3220 kg/m³（固定）
#    密度对比度 = 地幔密度 - 地壳密度（正值，莫霍面以下更重）
# ==============================================================================

porosity         = 0.12
crust_density    = density_raw * (1.0 - porosity)
mantle_density   = 3220.0
density_contrast = mantle_density - crust_density   # 2D array，逐点

print(f"Density contrast   min={density_contrast.min():.1f}  "
      f"max={density_contrast.max():.1f}  "
      f"mean={density_contrast.mean():.1f} kg/m³")


# ==============================================================================
# 3. 构造初始 MohoLayer
# ==============================================================================

reference    = -35e3 * np.ones(shape)   # 参考深度 -40 km（固定）
height_obs   = 11e3                      # 观测高度 11 km
surface_init = np.zeros(shape)           # 初始猜测：无起伏

layer = MohoLayer(
    lon=lon,
    lat=lat,
    height=height_obs,
    reference=reference,
    surface=surface_init,
    density_contrast=density_contrast,
)


# ==============================================================================
# 4. 反演
# ==============================================================================

inv = MohoInversion(
    layer=layer,
    go=go,
    max_iter=20,
    mu=1e-10,    # 先不加正则化；结果噪点多可以试 mu=0.05~0.5
    quite=True,
)

print("\nStarting inversion ...")
surface_inv = inv.inversion_gn(gtol=0.05)


# ==============================================================================
# 5. 统计
# ==============================================================================

stats = inv.residual_stats()
print("\nFinal residual stats:")
for k, v in stats.items():
    print(f"  {k:5s} = {v:.4f} mGal")


# ==============================================================================
# 6. 绘图
# ==============================================================================

# 莫霍面深度（正值向下，km）
moho_depth = -(reference + surface_inv) / 1e3

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
im = ax.pcolormesh(LON, LAT, go, cmap="viridis", shading="auto")
plt.colorbar(im, ax=ax, label="mGal")
ax.set_title("Observed Bouguer gravity")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

ax = axes[0, 1]
im = ax.pcolormesh(LON, LAT, moho_depth, cmap="RdBu_r", shading="auto")
plt.colorbar(im, ax=ax, label="km")
ax.set_title("Inverted Moho depth (km, positive down)")
ax.set_xlabel("Longitude")

inv.plot_convergence(ax=axes[1, 0])
inv.plot_residual_histogram(bins=40, ax=axes[1, 1])

plt.suptitle("Lunar Moho Inversion", fontsize=14)
plt.tight_layout()
plt.savefig("moho_inversion_result_1deg_35km.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved: moho_inversion_result_1deg_35km.png")


# ==============================================================================
# 7. 保存结果
# ==============================================================================
import pickle

print("calculating residuals")
# 最终残差（定义：residual = forward - observe）
residuals = inv.residual_2d()
print("calculating finished")
# 最终预测场
predict = go + residuals

# 收敛曲线
iters, rms = inv.convergence_data()

# 如果你要单个最终 RMS，也可以这样取
final_rms = inv.residual_stats()["rms"]

# 莫霍面深度（正值向下，km）
moho_depth = -(reference + surface_inv) / 1e3

# 保存为 pkl
result = dict()
result['LON'] = LON
result['LAT'] = LAT
result['observe'] = go
result['predict'] = predict
result['moho_grid'] = moho_depth
result['residuals'] = residuals
result['rms'] = rms              # 保存整个收敛序列
result['final_rms'] = final_rms  # 可选：再保存一个最终 RMS

print("Saving ...")
pkl_name = PROJECT_ROOT/"result/moho_inversion_result_1deg_35km.pkl"
with open(pkl_name, "wb") as f:
    pickle.dump(result, f)

print(f"Saved: {pkl_name}")