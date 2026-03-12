import matplotlib
matplotlib.use("Agg")

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from pathlib import Path

import cartopy.crs as ccrs


# =========================
# Nature + 中文风：全局样式
# =========================
def set_nature_cn_style():
    """
    尽量贴近 Nature 风格（简洁、细线、紧凑） + 中文标注友好。
    若系统无中文字体，会自动 fallback 到默认字体。
    """
    plt.rcParams.update({
        # 字体：优先中文常见字体（不保证每台机器都有）
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei",
                            "PingFang SC", "SimHei", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,

        # Nature-ish 视觉
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,

        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,

        "lines.linewidth": 1.2,
        "lines.markersize": 5,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "grid.linewidth": 0.5,
        "grid.alpha": 0.25,
    })


def _wrap_sort_lon(lon, data_2d):
    """
    如果 lon 是 0..360，把它 wrap 到 [-180,180) 并重排数据列，避免地图压缩/接缝问题。
    lon: (nlon,)
    data_2d: (nlat, nlon)
    """
    lon = lon.astype(float)
    if np.nanmax(lon) > 180:
        lon_wrapped = ((lon + 180) % 360) - 180
        idx = np.argsort(lon_wrapped)
        return lon_wrapped[idx], data_2d[:, idx]
    return lon, data_2d


def plot_map_cartopy(
    da: xr.DataArray,
    save_path: str,
    title: str,
    cbar_label: str,
    cmap: str = "jet",
    annotation_text: str | None = None,
):
    """
    用 cartopy + matplotlib 画最简单经纬度图（PlateCarree）并保存。
    da: dims=("lat","lon"), coords 包含 lat/lon
    """
    lon = da["lon"].values
    lat = da["lat"].values
    data = da.values

    lon_plot, data_plot = _wrap_sort_lon(lon, data)

    fig = plt.figure(figsize=(7.2, 3.8))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # extent
    lon_min, lon_max = float(np.nanmin(lon_plot)), float(np.nanmax(lon_plot))
    lat_min, lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    # 网格线（论文图常用：淡）
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, linestyle="--", alpha=0.25)
    gl.top_labels = False
    gl.right_labels = False

    # 主图
    mesh = ax.pcolormesh(
        lon_plot, lat, data_plot,
        transform=ccrs.PlateCarree(),
        shading="auto",
        cmap=cmap,
    )

    ax.set_title(title, pad=6)
    cbar = plt.colorbar(mesh, ax=ax, shrink=0.88, pad=0.03)
    cbar.set_label(cbar_label)

    # 右上角小标注（类似你原 pygmt text）
    if annotation_text:
        ax.text(
            0.99, 0.98, annotation_text,
            transform=ax.transAxes,
            ha="right", va="top",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", lw=0.6, alpha=0.9),
        )

    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_results(test: bool = False):
    """
    绘制并保存 Moho 反演结果的所有图件。

    参数:
        test (bool):
            True  → 使用 test_run_approach.pkl，图片保存到 test/ 文件夹
            False → 使用 run_approach.pkl，图片保存到 fig/ 文件夹（默认）
    """
    set_nature_cn_style()

    # ===== 确定数据文件路径和图片保存目录 =====
    PROJECT_ROOT = Path.cwd().parent
    if test:
        pkl_file = PROJECT_ROOT / "result/test_run_approach.pkl"
        save_dir = PROJECT_ROOT / "test"
    else:
        pkl_file = PROJECT_ROOT / "result/run_approach.pkl"
        save_dir = PROJECT_ROOT / "fig"

    os.makedirs(save_dir, exist_ok=True)

    # ===== 读取 pickle 数据 =====
    with open(pkl_file, "rb") as f:
        obj = pickle.load(f, encoding="latin1")

    # 提取所需变量并转为 numpy array
    density = np.asarray(obj["densities"])
    reference_levels = np.asarray(obj["reference_levels"])
    score_refden = np.asarray(obj["scores_refden"])
    score_regul = np.asarray(obj["scores_regul"])
    regul = np.asarray(obj["regul_params"])
    regul_residual = np.asarray(obj["regul_residuals"])
    refden_residual = np.asarray(obj["refden_residuals"])

    refden_moho_grid = obj["best_solutions_refden_moho_grid"]
    refden_predict_grid = obj["best_solutions_refden_predict_grid"]
    observe = obj["observe"]
    lon = obj["lon"]
    lat = obj["lat"]
    lon_sub = obj["lon_sub"]
    lat_sub = obj["lat_sub"]

    print("数据形状：")
    print(np.shape(density))
    print(np.shape(reference_levels))
    print(np.shape(score_refden))

    # ====================== 1. Score vs Density & Reference Level ======================
    print("1. Score vs Density & Reference Level")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    pcm = ax.pcolormesh(
        reference_levels, density, score_refden,
        shading="auto", cmap="viridis"
    )
    ax.set_xlabel("参考界面高度 Reference level (km)")
    ax.set_ylabel("密度差 Density contrast (kg m$^{-3}$)")
    ax.grid(True, which="both")

    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label("评分 Score")

    # 找最小值并标注
    imin = np.unravel_index(np.nanargmin(score_refden), score_refden.shape)
    i_den, i_ref = imin
    den_best = density[i_den]
    ref_best = reference_levels[i_ref]
    score_best = score_refden[i_den, i_ref]

    ax.plot(ref_best, den_best, marker="*", color="red",
            markersize=12, markeredgecolor="k", zorder=10, label="最小值 Minimum")

    ax.annotate(
        f"Min = {score_best:.2e}",
        xy=(ref_best, den_best),
        xycoords="data",
        xytext=(0.02, 0.95),
        textcoords="axes fraction",
        fontsize=10,
        ha="left", va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.6),
        zorder=11
    )

    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "score_density_reference_minimum.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("finished")

    # ====================== 2. Score vs Regularization Parameter ======================
    print("2. Score vs Regularization Parameter")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(regul, score_regul, "-o", color="k")
    ax.set_xscale("log")
    ax.set_xlabel("正则化参数 Regularization parameter")
    ax.set_ylabel("评分 Score")
    ax.grid(True, which="both")

    i_min = np.nanargmin(score_regul)
    regul_best = regul[i_min]
    score_best2 = score_regul[i_min]

    ax.plot(regul_best, score_best2, marker="*", color="red",
            markersize=12, markeredgecolor="k", zorder=10, label="最小值 Minimum")

    ax.annotate(
        f"Min = {score_best2:.2e}\n$\\lambda$ = {regul_best:.2e}",
        xy=(regul_best, score_best2),
        xytext=(10, 10),
        textcoords="offset points",
        fontsize=10,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", lw=0.6),
        arrowprops=dict(arrowstyle="->", lw=0.8)
    )

    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "score_vs_regularization.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("finished")

    # ====================== 3. Regularization Residual Histogram ======================
    print("3. Regularization Residual Histogram")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    n_bins = 30
    ax.hist(regul_residual, bins=n_bins, color="0.6", edgecolor="k", alpha=0.85)

    mean_val = np.mean(regul_residual)
    std_val = np.std(regul_residual)
    ax.axvline(mean_val, color="r", linestyle="--", lw=1.2, label=f"均值 Mean = {mean_val:.2e}")
    ax.axvline(mean_val - std_val, color="k", linestyle=":", lw=1.0, label="-1σ")
    ax.axvline(mean_val + std_val, color="k", linestyle=":", lw=1.0, label="+1σ")

    ax.set_xlabel("残差 Residual")
    ax.set_ylabel("计数 Count")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "regul_residual_histogram.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("finished")

    # ====================== 4. Refden Residual Histogram ======================
    print("4. Refden Residual Histogram")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(refden_residual, bins=n_bins, color="0.6", edgecolor="k", alpha=0.85)

    mean_val = np.mean(refden_residual)
    std_val = np.std(refden_residual)
    ax.axvline(mean_val, color="r", linestyle="--", lw=1.2, label=f"均值 Mean = {mean_val:.2e}")
    ax.axvline(mean_val - std_val, color="k", linestyle=":", lw=1.0, label="-1σ")
    ax.axvline(mean_val + std_val, color="k", linestyle=":", lw=1.0, label="+1σ")

    ax.set_xlabel("残差 Residual")
    ax.set_ylabel("计数 Count")
    ax.grid(True, axis="y")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, "refden_residual_histogram.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("finished")

    # ====================== 5. Moho Depth Map (Cartopy) ======================
    print("5. Moho Depth Map")
    nlat_moho, nlon_moho = refden_moho_grid.shape
    lon_moho = np.linspace(lon.min(), lon.max(), nlon_moho)
    lat_moho = np.linspace(lat.min(), lat.max(), nlat_moho)

    moho_grid = xr.DataArray(
        refden_moho_grid,
        dims=("lat", "lon"),
        coords={"lat": lat_moho, "lon": lon_moho}
    )

    plot_map_cartopy(
        moho_grid,
        save_path=os.path.join(save_dir, "map_moho_depth.png"),
        title="Moho 深度分布图（反演结果）",
        cbar_label="Moho depth (m)",
        cmap="jet",
        annotation_text="Moho"
    )
    print("finished")

    # ====================== 6. Predicted Gravity Map (Cartopy) ======================
    print("6. Predicted Gravity Map")
    nlat_p, nlon_p = refden_predict_grid.shape
    lon_grid = np.linspace(lon.min(), lon.max(), nlon_p)
    lat_grid = np.linspace(lat.min(), lat.max(), nlat_p)

    predict_grid = xr.DataArray(
        refden_predict_grid,
        dims=("lat", "lon"),
        coords={"lon": lon_grid, "lat": lat_grid}
    )

    plot_map_cartopy(
        predict_grid,
        save_path=os.path.join(save_dir, "map_predict.png"),
        title="重力异常（预测）",
        cbar_label="Predicted gravity (mGal)",
        cmap="viridis",
        annotation_text="预测 Predicted"
    )
    print("finished")

    # ====================== 7. Observed Gravity Map (Cartopy) ======================
    print("7. Observed Gravity Map")
    nlat_sub, nlon_sub = observe.shape
    lon_sub_grid = np.linspace(lon_sub.min(), lon_sub.max(), nlon_sub)
    lat_sub_grid = np.linspace(lat_sub.min(), lat_sub.max(), nlat_sub)

    observe_grid = xr.DataArray(
        observe,
        dims=("lat", "lon"),
        coords={"lon": lon_sub_grid, "lat": lat_sub_grid}
    )

    plot_map_cartopy(
        observe_grid,
        save_path=os.path.join(save_dir, "map_observe.png"),
        title="重力异常（观测）",
        cbar_label="Observed gravity (mGal)",
        cmap="viridis",
        annotation_text="观测 Observed"
    )

    print(f"所有图件已成功保存至文件夹：{save_dir}")


# ===== 脚本入口 =====
if __name__ == "__main__":
    print("当前工作目录 =", os.getcwd())

    test = False
    if len(sys.argv) > 1 and sys.argv[1].lower() == "test":
        test = True

    plot_results(test=test)