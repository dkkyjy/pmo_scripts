# Mazongshan StarShape 数据分析

GRAND GP300 **StarShape** 布局 ZHAireS 模拟数据的探索性分析脚本。

## 数据

- 路径（pmo001）：`/home/mapx/mapx/GP300_StarShape_ROOTFiles/`
  - `Proton/Proton_EeV_Mazongshan_stshp_2100Events.root`（12 GB，2099 events）
  - 同级还有 `Iron/`、`Gamma/`、`Event/`（类定义）、`READ.ME`、`_Readfile.cpp`
- 每个 ROOT 文件含 3 棵树，各含一个 `Sim**Branch`：

| Tree | Branch | 内容 |
|---|---|---|
| `SimuCollection` | `SimShowerBranch` | 原初/簇射真值：能量、天顶/方位角、Xmax、磁场、注入点、大气… |
| `EfieldCollection` | `SimEfieldBranch` | 星形阵列电场波形（176 天线） |
| `SignalCollection` | `SimSignalBranch` | 天线电压 p2p / t_0 及电压波形 |

> 参考同目录 `READ.ME` 与 `_Readfile.cpp`（Ma PengXiong, 2021，原作者用它做 zenith/azimuth 的 3D 扫描重建）。

## 脚本

`analyze_starshape.py` —— 轻量探索（只读标量分支 + p2p，不加载 12 GB 波形）。

```bash
# pmo001, py310 环境
/home/dkk/anaconda3/envs/py310/bin/python analyze_starshape.py \
    --file /home/mapx/mapx/GP300_StarShape_ROOTFiles/Proton/Proton_EeV_Mazongshan_stshp_2100Events.root \
    --out  /home/dkk/grand/mazongshan_out
```

输出 `report.txt`（统计）与 `proton_mazongshan_overview.png`（能量/角度/Xmax 分布、星形布局、p2p 分布）。

## 首轮结果（Proton_EeV_Mazongshan_stshp_2100Events.root）

- events = 2099；`prim_energy` 0.75–3.98 EeV（median 1.73）
- `shower_zenith` 105.9°–136.5°（median 118°，即全部为**上仰** showers）
- `shower_azimuth` 36°–324°（median 180°）
- `xmax_grams` 629–1190 g/cm²（median 757）；`xmax_alt` 2.5 km–8.7 km（1 个哨兵值 >1e5 m）
- `|xmax_pos_shc|` 1.7–39 km（median 15.7 km）
- 天线阵列（event 0）：176 天线，|pos_xy| 17–1043 m（星形布局）
- `p2p`（每事件 176 天线）：`p2p_max` median 8.0e3、max 1.2e5 ADC；`p2p>75` 的天线数 median 176、min 147
