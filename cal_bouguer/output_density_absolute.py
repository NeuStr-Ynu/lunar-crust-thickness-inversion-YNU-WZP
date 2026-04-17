"""
从球谐系数文件输出全月面绝对密度数据。

输出文件：certaincertain/density_absolute.csv
  列：lat, lon, density   （单位 kg/m³，绝对值，非异常）

输出全网格和1deg网格的数据
"""

import numpy as np
import pyshtools as pysh
import boule as bl
import verde as vd
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent

df         = pd.read_csv(PROJECT_ROOT/'data/boueguer_frenquency_11km_d35km_1deg.csv')

DENSITY_FILE  = PROJECT_ROOT/"data/density_no_mare_n3000_f3050_719.sh"
BOUGUER_CSV   = PROJECT_ROOT/"data/boueguer_frenquency_11km.csv"
OUTPUT_CSV    = PROJECT_ROOT/"data/density_absolute.csv"
LMAX          = 719

# ── 读取并展开密度球谐系数
print(f"读取密度文件：{DENSITY_FILE}")
density_coeffs = pysh.SHCoeffs.from_file(DENSITY_FILE, lmax=LMAX)
density_grid   = density_coeffs.expand(grid='DH2', lmax=LMAX, extend=False)
density_xr     = density_grid.to_xarray()   # 绝对密度，lat 从 90→-90
density_xr     = density_xr.sortby("lat")   # 转为 -90 → 90

print(f"密度网格：lat {density_xr.lat.values[0]:.2f}~{density_xr.lat.values[-1]:.2f}  "
      f"lon {density_xr.lon.values[0]:.2f}~{density_xr.lon.values[-1]:.2f}")
print(f"密度范围：{float(density_xr.min()):.1f} ~ {float(density_xr.max()):.1f} kg/m³")

# ── 读取目标网格（与 Bouguer CSV 相同的经纬度）
print(f"\n读取目标网格：{BOUGUER_CSV}")
df_ref = pd.read_csv(BOUGUER_CSV)
# 经度统一转换为 -180~180
df_ref['lon'] = np.where(df_ref['lon'] > 180, df_ref['lon'] - 360, df_ref['lon'])

lat_target = np.sort(df_ref['lat'].unique())
lon_target = np.sort(df_ref['lon'].unique())
print(f"目标网格：lat n={len(lat_target)}, lon n={len(lon_target)}")

# ── 将密度插值到目标网格
# density_xr 的 lon 是 0~360，需要转换后才能与 -180~180 的目标网格对齐
density_xr_180 = density_xr.assign_coords(
    lon=(density_xr.lon.values + 180) % 360 - 180
)
density_xr_180 = density_xr_180.sortby("lon")

density_interp = density_xr_180.interp(
    lat=lat_target, lon=lon_target,
    kwargs={"fill_value": "extrapolate"}
)

print(f"插值后密度范围：{float(density_interp.min()):.1f} ~ "
      f"{float(density_interp.max()):.1f} kg/m³")

# ── 输出 CSV
df_out = (
    density_interp
    .stack(points=('lat', 'lon'))
    .reset_index('points')
    .to_dataframe(name='density')
    .reset_index(drop=True)
)

df_out.to_csv(OUTPUT_CSV, index=False)
print(f"\n输出完成：{OUTPUT_CSV}")
print(f"行数：{len(df_out)}  列：{list(df_out.columns)}")
print(df_out.describe())

# —— 输出1deg网格，供全球反演用

PROJECT_ROOT = Path.cwd().parent
data_filename=PROJECT_ROOT/"data/density_no_mare_n3000_f3050_719.sh"

region=(0, 359.99999, -90, 90)
grid_longitude, grid_latitude = vd.grid_coordinates(region=region, shape=(90*2+1,180*2+1))
lon_o=np.sort(np.unique(grid_longitude))
lat_o=np.sort(np.unique(grid_latitude))
lon = 0.5 * (lon_o[:-1] + lon_o[1:])
lat = 0.5 * (lat_o[:-1] + lat_o[1:])
density_xr_sub=density_xr.interp(lat=lat,lon=lon, kwargs={"fill_value": "extrapolate"})

df = (
    density_xr_sub
    .stack(points=('lat','lon'))
    .reset_index('points')
    .to_dataframe(name='deltaN')
    .reset_index(drop=True)
)

df.to_csv(PROJECT_ROOT/"data/density_1deg.csv", index=False)