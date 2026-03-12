import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.widgets import Cursor
from PIL import Image
import csv
import math

# ===== 中文显示（Windows 常用）=====
mpl.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False

# -------------------------
# Mollweide forward/inverse
# -------------------------
SQRT2 = math.sqrt(2.0)

def mollweide_forward(lon_deg, lat_deg):
    """(lon, lat) degrees -> (x, y) Mollweide projected coords (central meridian 0°)."""
    lam = math.radians(lon_deg)
    phi = math.radians(lat_deg)

    # Solve: 2θ + sin(2θ) = π sin(phi)
    target = math.pi * math.sin(phi)
    theta = phi
    for _ in range(30):
        f = 2 * theta + math.sin(2 * theta) - target
        fp = 2 + 2 * math.cos(2 * theta)
        if abs(fp) < 1e-12:
            break
        step = f / fp
        theta -= step
        if abs(step) < 1e-12:
            break

    x = (2 * SQRT2 / math.pi) * lam * math.cos(theta)
    y = SQRT2 * math.sin(theta)
    return x, y

def mollweide_inverse(x, y):
    """(x, y) Mollweide -> (lon, lat) degrees."""
    y = max(-SQRT2, min(SQRT2, y))
    theta = math.asin(y / SQRT2)

    lat = math.asin((2 * theta + math.sin(2 * theta)) / math.pi)

    c = math.cos(theta)
    if abs(c) < 1e-12:
        lon = 0.0
    else:
        lon = (x * math.pi) / (2 * SQRT2 * c)

    return math.degrees(lon), math.degrees(lat)

# -------------------------
# Pixel <-> projected affine (separable scale+offset)
# Fit:
#   px = a*x + b
#   py = c*y + d
# -------------------------
def fit_affine_xy(xy, pxpy):
    x = xy[:, 0]
    y = xy[:, 1]
    px = pxpy[:, 0]
    py = pxpy[:, 1]

    A1 = np.column_stack([x, np.ones_like(x)])
    a, b = np.linalg.lstsq(A1, px, rcond=None)[0]

    A2 = np.column_stack([y, np.ones_like(y)])
    c, d = np.linalg.lstsq(A2, py, rcond=None)[0]

    return a, b, c, d

def pixel_to_projected(px, py, a, b, c, d):
    x = (px - b) / a
    y = (py - d) / c
    return x, y

def wrap_lon(lon_deg, mode="180"):
    if mode == "360":
        lon = lon_deg % 360.0
        if lon < 0:
            lon += 360.0
        return lon
    return ((lon_deg + 180.0) % 360.0) - 180.0

# -------------------------
# Interactive digitizer:
# - Left click: add point
# - Delete/Backspace: undo last point
# - Enter: finish
# - Crosshair cursor across the whole axes
# -------------------------
class LiveDigitizer:
    def __init__(self, ax):
        self.ax = ax
        self.points = []     # list of (px, py)
        self.done = False

        self.scatter = None
        self.line = None

        # Crosshair cursor
        self.cursor = Cursor(ax, useblit=True, linewidth=1)

        self.cid_click = ax.figure.canvas.mpl_connect("button_press_event", self.on_click)
        self.cid_key = ax.figure.canvas.mpl_connect("key_press_event", self.on_key)

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        self.points.append((event.xdata, event.ydata))
        self.redraw()

    def on_key(self, event):
        if event.key in ("delete", "backspace"):
            if self.points:
                self.points.pop()
                self.redraw()
        elif event.key in ("enter", "return"):
            self.done = True
            plt.close(self.ax.figure)

    def redraw(self):
        # Remove old artists to fully refresh after undo
        if self.scatter is not None:
            self.scatter.remove()
            self.scatter = None
        if self.line is not None:
            self.line.remove()
            self.line = None

        if self.points:
            xs = [p[0] for p in self.points]
            ys = [p[1] for p in self.points]
            self.scatter = self.ax.scatter(xs, ys, s=20)
            if len(self.points) >= 2:
                self.line, = self.ax.plot(xs, ys, linewidth=1)

        self.ax.figure.canvas.draw_idle()

# -------------------------
# Plot with Cartopy (ccrs) for final check
# -------------------------
def plot_with_cartopy(lons, lats, central_longitude=0):
    try:
        import cartopy.crs as ccrs
    except Exception as e:
        print("\n[警告] cartopy 未安装，无法用 ccrs 绘图。")
        print("建议：conda install -c conda-forge cartopy\n")
        print("原始错误：", e)
        return

    fig = plt.figure(figsize=(12, 6))
    ax = plt.axes(projection=ccrs.Mollweide(central_longitude=central_longitude))
    ax.gridlines(draw_labels=False, linewidth=0.5)
    ax.plot(lons, lats, transform=ccrs.PlateCarree(), linewidth=1.5)
    ax.set_title("Digitize 结果（Mollweide 投影显示）")
    plt.show()

# -------------------------
# Main
# -------------------------
def main(
    image_path="your_map.png",
    out_csv="digitized_boundary.csv",
    lon_wrap_mode="180",
    central_longitude_for_plot=0
):
    img = Image.open(image_path)

    # ===== Calibration stage =====
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(img)
    ax.axis("off")

    cal_points = [
        ("(0°,0°) 赤道与中央经线交点", 0, 0),
        ("(90°,0°) 赤道与90E交点", 90, 0),
        ("(-90°,0°) 赤道与90W交点", -90, 0),
        ("(0°,60°) 60N与中央经线交点", 0, 60),
        ("(0°,-60°) 60S与中央经线交点", 0, -60),
        ("(180°,0°) 赤道与180度边界（右端）", 180, 0),
        ("(-180°,0°) 赤道与-180度边界（左端）", -180, 0),
    ]

    print("\n=== 校准阶段 ===")
    print("按标题提示依次点击经纬网交点（点错直接关窗口重来最快）。\n")

    proj_list, pix_list = [], []
    for label, lon, lat in cal_points:
        ax.set_title(f"校准：请点击 {label}", fontsize=12)
        plt.draw()
        pts = plt.ginput(1, timeout=-1)
        if not pts:
            raise RuntimeError("未点击点（可能关闭了窗口）。")
        px, py = pts[0]
        x, y = mollweide_forward(lon, lat)
        proj_list.append([x, y])
        pix_list.append([px, py])

        ax.plot(px, py, marker="x")
        ax.text(px + 6, py + 6, f"{lon},{lat}", fontsize=9)
        plt.draw()

    proj_arr = np.array(proj_list, dtype=float)
    pix_arr = np.array(pix_list, dtype=float)
    a, b, c, d = fit_affine_xy(proj_arr, pix_arr)

    # calibration error report
    px_pred = a * proj_arr[:, 0] + b
    py_pred = c * proj_arr[:, 1] + d
    err = np.sqrt((px_pred - pix_arr[:, 0])**2 + (py_pred - pix_arr[:, 1])**2)

    print("校准点像素误差（越小越好）：")
    for i, e in enumerate(err):
        print(f"  {cal_points[i][0]}  error ≈ {e:.2f} px")
    print(f"平均误差 ≈ {err.mean():.2f} px\n")

    # ===== Digitize stage (live, with crosshair + undo) =====
    ax.set_title("Digitize：左键沿边界打点；Delete/Backspace 删除上一个点；Enter 结束", fontsize=12)
    plt.draw()

    print("=== Digitize阶段 ===")
    print("操作：左键打点；Delete/Backspace 撤销上一个点；Enter 结束并保存。\n")

    dig = LiveDigitizer(ax)
    plt.show()  # block until Enter closes figure

    clicked = dig.points
    if len(clicked) < 2:
        print("点太少，未保存。")
        return

    # Convert to lon/lat and save CSV
    lons, lats = [], []
    records = []
    for px, py in clicked:
        x, y = pixel_to_projected(px, py, a, b, c, d)
        lon, lat = mollweide_inverse(x, y)
        lon = wrap_lon(lon, mode=lon_wrap_mode)
        lons.append(lon)
        lats.append(lat)
        records.append((lon, lat, px, py))

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["lon_deg", "lat_deg", "pixel_x", "pixel_y"])
        w.writerows(records)

    print(f"完成！已导出：{out_csv}")

    # ===== Final check with Cartopy =====
    plot_with_cartopy(lons, lats, central_longitude=central_longitude_for_plot)


if __name__ == "__main__":
    main(
        image_path="your_map.png",          # 改成你的图片名
        out_csv="digitized_boundary.csv",
        lon_wrap_mode="180",                # 或 "360"
        central_longitude_for_plot=0        # 画最终 Mollweide 的中央经线
    )