# 用地形和重力数据反演月壳厚度 代码库
本库为本人毕业论文的使用的所有代码，包括反演与成图。
## 略览
本论文主要研究月球月壳与其反演问题，尝试利用Uieda等（2017）发展的在地球上广泛使用的基于球面棱柱体重力正演的莫霍面反演方法与Wieczorek等（1998）发展的球谐域的全球莫霍面反演的方法，利用LDEM高精度全月面地形数据与GEGM1200B高精度月球重力场模型展开对月壳厚度的反演，分别得到空间域反演与频率域反演的月壳厚度模型，对比与讨论二者的优劣性，并展开对月球相关地质问题的讨论，如月球不对称性、关键地质单元和月球演化历史，并对未来可能的探测任务提供一定的支持。
本文数据处理与反演计算主要通过Python进行。球谐域计算布格重力异常使用Pyshtools（https://shtools.github.io/SHTOOLS/）（Wieczorek 等 2018）进行；空间域的地形引起的重力场的建模与正演计算使用开源库Fatiando a Terra（https://www.fatiando.org/）（Uieda 等 2013）；数据成图使用python库cartpy（https://cartopy.readthedocs.io/）（Elson P, Sales de Andrade E, Lucas G, et al. SciTools/cartopy: REL: v0. 24.1[J]. Zenodo, 2024.）与matplotlib（https://matplotlib.org/）（Hunter J D. Matplotlib: A 2D graphics environment[J]. Computing in science & engineering, 2007, 9(3): 90-95.）；数据读取输入和计算使用numpy（https://numpy.org/）（Harris 等 2020）、xarray（https://docs.xarray.dev/en/stable/）（Hoyer 等 2017）和pandas（https://pandas.pydata.org/）（McKinney 2010）；球谐域反演使用软件ctplanet（https://markwieczorek.github.io/ctplanet/index.html），空间域反演使用软件Fast non-linear gravity inversion in spherical coordinates with application to the South American Moho（https://github.com/pinga-lab/paper-moho-inversion-tesseroids）（Uieda 等 2017）我们修改了软件中的相关参数，使其能够在月球的环境下稳定运行。

### 参考文献

Uieda L, Barbosa V C F. Fast nonlinear gravity inversion in spherical coordinates with application to the South American Moho[J]. Geophysical Journal International, 2017, 208(1): 162-176.

Uieda L, Oliveira Jr V C, Barbosa V C F. Modeling the earth with Fatiando a Terra[C]//Proceedings of the 12th Python in Science Conference. 2013: 91-98. DOI:10.25080/Majora-8b375195-010.

Wieczorek M A, Phillips R J. Potential anomalies on a sphere: Applications to the thickness of the lunar crust[J]. Journal of Geophysical Research: Planets, 1998, 103(E1): 1715-1724.

Wieczorek M A, Meschede M. SHTools: Tools for working with spherical harmonics[J]. Geochemistry, Geophysics, Geosystems, 2018, 19(8): 2574-2592.

Harris C R, Millman K J, Van Der Walt S J, et al. Array programming with NumPy[J]. nature, 2020, 585(7825): 357-362.

Hoyer S, Hamman J. xarray: ND labeled arrays and datasets in Python[J]. 2017.

McKinney W. Data structures for statistical computing in Python[J]. scipy, 2010, 445(1): 51-56.

## 文件组成

本库中文件主要有三种：代码文件（.py .ipynb）；图片文件（.png）；数据文件（.csv .xlsx .sh）

各个文件夹的各个文件内容如下：

**approaches:**

​	内部包含所有运行的结果报告，但是只有最后一次是正确的，因为前期一直是使用的错误的布格重力异常（没有乘以孔隙度）

**data:**

​	外层文件的数据存放处，计算布格重力异常的结果和月球密度数据与月球主要盆地的表格也在这里

**digitize:**

​	一个在Mollweide投影的图中扣除线的坐标点的小程序。

​	**data:** 存放了月球主要构造单元边界的数据

​	`mollweide_digitize.py`: 主程序

​	`plot_boundary.py`: 画图边界的函数程序

​	`plot_ortho_front_back.py`: 画图的程序

​	`test_plot.py`: 画图的测试程序

​	`your_map.png`: 要数字化的图

**plotting**:

​	主要的画图程序位置

​	**data**: 存放了月球主要构造单元边界的数据

​	**figure**: 图件输出的主要位置

​	**plot_area**: 画一个固定区域的图

​		**fig**: 输出存放位置

​		`lonlat_labels.py`: cartpy在正射投影时候不能很好的画出经纬度的标记，这是一个补丁函数

​		`plot_basins.py`: 在cartpy的正射投影图中标划出主要盆地的位置的函数

​		`plot_boundary.py`: 画出主要地质构造单元边界的函数

​		`plot_certain_area.py`: 画出固定区域的函数们

​		`plot_certain_area_main.ipynb`: 该部分画图主文件

​	**tesseroid_test**: 补充文件图件的

​		`lonlat_labels.py`: 同上

​		`mohoinv.py`: 空间域反演使用软件Fast non-linear gravity inversion in spherical coordinates with application to the South American Moho（https://github.com/pinga-lab/paper-moho-inversion-tesseroids）（Uieda 等 2017）的主函数文件

​		`plot_basins.py`: 同上（在这里没有用上）

​		`tesseroid_py2.ipynb`: 老软件重力正演模拟程序

​		`tesseroid_py3.ipynb`: 新软件重力正演模拟程序

​	`add_text_lonlat.py`: 在经纬度自动添加文字

​	`lonlat_labels.py`: 同上











