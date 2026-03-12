import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib as mpl
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

def plot_one(ax, df, center_lon, center_lat=0, title=""):
    proj = ccrs.Orthographic(central_longitude=center_lon, central_latitude=center_lat)
    ax = plt.axes(projection=proj)

    # 可选：画一个经纬网，方便对齐
    ax.gridlines(draw_labels=False, linewidth=0.5)

    # 只显示一个半球
    ax.set_global()

    # 你的点是经纬度数据，所以 transform 用 PlateCarree
    ax.plot(
        df["lon_deg"].values,
        df["lat_deg"].values,
        transform=ccrs.PlateCarree(),
        linewidth=1.5
    )

    ax.set_title(title)
    return ax


def main():
    df = pd.read_csv("digitized_boundary.csv")

    # 正面 / 背面：用不同 central_longitude
    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.Orthographic(central_longitude=0, central_latitude=0))
    ax1.set_global()
    ax1.gridlines(draw_labels=False, linewidth=0.5)
    ax1.plot(df["lon_deg"], df["lat_deg"], transform=ccrs.PlateCarree(), linewidth=1.5)
    ax1.set_title("正射投影：正面（0°, 0°）")

    ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(central_longitude=180, central_latitude=0))
    ax2.set_global()
    ax2.gridlines(draw_labels=False, linewidth=0.5)
    ax2.plot(df["lon_deg"], df["lat_deg"], transform=ccrs.PlateCarree(), linewidth=1.5)
    ax2.set_title("正射投影：背面（180°, 0°）")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()