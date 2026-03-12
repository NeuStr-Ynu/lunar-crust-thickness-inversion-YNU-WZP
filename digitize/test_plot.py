import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from plot_boundary import plot_boundaries_folder_on_ax
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False
def plot_two_hemispheres(folder_path):
    fig = plt.figure(figsize=(12, 6))

    # 正面：中心(0,0)
    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    ax1.set_global()
    ax1.gridlines(draw_labels=False, linewidth=0.5)
    ax1.set_title("正射投影：正面 (0°, 0°)")

    # 背面：中心(180,0)
    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=180, central_latitude=0))
    ax2.set_global()
    ax2.gridlines(draw_labels=False, linewidth=0.5)
    ax2.set_title("正射投影：背面 (180°, 0°)")

    # 把同一批边界画到两个轴上
    plot_boundaries_folder_on_ax(ax1, folder_path, thick_keywords=("SPATU", "MTU"))
    plot_boundaries_folder_on_ax(ax2, folder_path, thick_keywords=("SPATU", "MTU"))

    plt.tight_layout()
    plt.show()

# 用法：
plot_two_hemispheres(folder_path="E:/Moho/digitize/data")