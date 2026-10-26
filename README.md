# 信用卡违约风险评分卡 · Credit Scorecard

基于 UCI「Default of Credit Card Clients」公开数据集的**工业级评分卡完整流水线**：数据清洗 → 分箱与 WOE/IV → 逻辑回归 → AUC/KS 评估 → 评分卡转换 → PSI 稳定性验证。

**在线报告**：<https://cloud1-d4gsgj705497df85d-1501559031.tcloudbaseapp.com/scorecard/>

## 核心结果

| 指标 | 数值 | 说明 |
|---|---|---|
| 测试集 AUC | **0.759** | 5 折交叉验证 0.772 ± 0.005，无明显过拟合 |
| KS 统计量 | **0.403** | 评分卡实务中区分度较好（>0.3 可用，>0.4 良好） |
| PSI（train vs test） | **0.002** | 分布高度一致，稳定性极好 |
| 建模样本 | 29,965 | 违约率 22.1%，8:2 分层抽样 |

入模特征 14 个，预测能力排序与业务直觉一致：**还款历史（IV 0.3~0.9）＞ 授信额度与还款金额（0.1~0.3）＞ 人口属性（<0.02，剔除）**。

## 方法要点

- **分箱与 WOE**：数值变量等频分箱 + 低频箱相邻合并；train/test 严格共用同一套**数值边界**（避免区间字符串匹配导致的 WOE 错配——这是初版实现踩过的真实坑，修复后 test AUC 0.748→0.759，PSI 0.64→0.002）。
- **特征筛选**：IV ≥ 0.02 保留；WOE 化后相关矩阵 |r|>0.7 时保留 IV 更高者。
- **评分卡转换**：`Score = A + B·ln(好坏比)`，PDO=40、600 分对应 20:1 好坏比，输出每个特征每个分箱的单变量分值表。
- **诚实声明**：数据集无时间戳，PSI 验证的是随机划分下的分布一致性而非时间外推；分箱未强制 WOE 单调；与 GBDT 基线的对比是明确的改进方向。

## 目录结构

```
credit-scorecard/
├── data/                # UCI 原始数据（xlsx）
├── src/
│   ├── analysis.py      # 完整分析流水线（输出图表 PNG + metrics.json）
│   └── build_report.py  # 由 metrics+图表生成在线 HTML 报告
└── public/              # 部署产物（在线报告页）
```

## 复现

```bash
pip install pandas numpy scikit-learn matplotlib xlrd
python3 src/analysis.py       # 跑完整流水线，输出 output/
python3 src/build_report.py   # 生成 public/index.html 报告页
```

数据来源：[UCI Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)（Yeh & Lien, 2009）。

## License

MIT
