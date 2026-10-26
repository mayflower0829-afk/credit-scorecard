# -*- coding: utf-8 -*-
"""把 output/ 下的 metrics.json 与图表组装为可部署的单页 HTML 报告。

用法：python3 src/build_report.py
产物：credit-scorecard/public/（index.html + charts/*.png）
"""
import json
import pathlib
import shutil

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
PUB = ROOT / "public"
(CHARTS := PUB / "charts").mkdir(parents=True, exist_ok=True)

GBDT_M = json.loads((OUT / "gbdt_metrics.json").read_text(encoding="utf-8")) if (OUT / "gbdt_metrics.json").exists() else None

M = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))

for png in OUT.glob("*.png"):
    shutil.copy(png, CHARTS / png.name)

CHART_NAMES = {
    "01_eda_overview.png": "图1 · 数据概览",
    "02_pay0_defaultrate.png": "图2 · 还款状态与违约率",
    "03_iv_ranking.png": "图3 · 特征 IV 排序",
    "04_model_eval.png": "图4 · 模型评估（ROC 与概率分布）",
    "05_score_grade.png": "图5 · 评分分组与实际违约率",
}
figcap04 = CHART_NAMES['04_model_eval.png']


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def metric_card(v, label, note=""):
    return f'<div class="kpi"><b>{v}</b><span>{label}</span><em>{note}</em></div>'


iv_rows = "".join(
    f"<tr><td>{esc(r['feature'])}</td><td>{r['iv']:.3f}</td></tr>"
    for r in M["iv_table"][:10]
)
coef_rows = "".join(
    f"<tr><td>{esc(k)}</td><td>{v:+.3f}</td></tr>" for k, v in M["coef"].items()
)

# 评分卡分值表：只展示 IV 最高的 5 个特征
top5 = [r["feature"] for r in M["iv_table"] if r["feature"] in M["coef"]][:5]
score_tables = ""
for feat in top5:
    rows = "".join(
        f"<tr><td>{esc(k)}</td><td>{v:+.1f}</td></tr>" for k, v in M["scorecard_table"][feat].items()
    )
    score_tables += f"""
    <div class="card"><h4>{esc(feat)}</h4>
    <table class="tbl"><thead><tr><th>分箱</th><th>得分</th></tr></thead><tbody>{rows}</tbody></table></div>"""

grade = " → ".join(f"{v:.1f}%" for v in M["grade_badrate"])

GBDT_ROWS = ""
GBDT_IMG = ""
if GBDT_M:
    GBDT_ROWS = "".join(
        f"<tr><td>{esc(r['model'])}</td><td>{r['cv_auc']:.3f} ± {r['cv_std']:.3f}</td><td>{r['test_auc']:.3f}</td></tr>"
        for r in GBDT_M["models"]
    )
    GBDT_IMG = '<img src="charts/09_gbdt_compare.png" alt="GBDT对比"><p class="figcap">图4b · 与 GBDT 基线的 ROC 对比</p>'

kpi = (
    metric_card(f"{M['auc_test']:.3f}", "测试集 AUC", f"5折CV {M['cv_auc_mean']:.3f}±{M['cv_auc_std']:.3f}")
    + metric_card(f"{M['ks']:.3f}", "KS 统计量", "风控常用区分度指标")
    + metric_card(f"{M['psi_train_vs_test']:.3f}", "PSI 稳定性", "&lt; 0.1 视为稳定")
    + metric_card(f"{M['n_samples']:,}", "建模样本", f"违约率 {M['default_rate']}%")
)

html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>信用卡违约风险评分卡 · 数据分析报告</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><rect width='100' height='100' rx='22' fill='%23B4540A'/><text x='50' y='68' font-size='46' text-anchor='middle' fill='white' font-family='sans-serif' font-weight='bold'>卡</text></svg>">
<style>
:root {{ --bg:#FAF8F5; --card:#fff; --text:#23201C; --muted:#8A8378; --border:#EAE5DD; --accent:#B4540A; }}
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{ font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.75; }}
header {{ background:linear-gradient(135deg,#B4540A,#D97E2B); color:#fff; padding:26px 22px 20px; border-radius:0 0 22px 22px; }}
header h1 {{ font-size:22px; }} header p {{ font-size:13px; opacity:.9; }}
main {{ max-width:900px; margin:0 auto; padding:18px 16px 50px; }}
.kpis {{ display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin-top:-6px; }}
@media (max-width:640px) {{ .kpis {{ grid-template-columns:repeat(2,1fr); }} }}
.kpi {{ background:var(--card); border-radius:14px; box-shadow:0 2px 10px rgba(35,32,28,.07); padding:14px 10px; text-align:center; margin-top:14px; }}
.kpi b {{ display:block; font-size:24px; color:var(--accent); }}
.kpi span {{ font-size:12px; font-weight:700; }} .kpi em {{ display:block; font-size:11px; font-style:normal; color:var(--muted); }}
section {{ margin-top:22px; }}
.card {{ background:var(--card); border-radius:14px; box-shadow:0 2px 10px rgba(35,32,28,.07); padding:18px 20px; margin-top:12px; }}
h2 {{ font-size:17px; border-left:4px solid var(--accent); padding-left:10px; }}
h3 {{ font-size:15px; margin:14px 0 6px; }} h4 {{ font-size:14px; margin-bottom:6px; }}
p {{ font-size:14px; margin-top:8px; }}
ul {{ font-size:14px; padding-left:20px; }} li {{ margin-top:4px; }}
img {{ max-width:100%; border-radius:10px; border:1px solid var(--border); margin-top:10px; }}
.figcap {{ font-size:12px; color:var(--muted); margin-top:4px; }}
table.tbl {{ width:100%; border-collapse:collapse; font-size:13px; margin-top:8px; }}
.tbl th, .tbl td {{ border-bottom:1px solid var(--border); padding:7px 10px; text-align:left; }}
.tbl th {{ color:var(--muted); font-weight:600; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
@media (max-width:640px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
.steps {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:10px; }}
.step {{ background:#FFF4EA; color:var(--accent); border-radius:999px; padding:5px 13px; font-size:12.5px; font-weight:600; }}
.note {{ background:#FFF9EF; border:1px solid #F2E3C9; border-radius:10px; padding:10px 14px; font-size:13px; margin-top:10px; }}
footer {{ text-align:center; font-size:12px; color:var(--muted); padding:26px 0 34px; }}
a {{ color:var(--accent); }}
</style>
</head>
<body>
<header>
  <h1>信用卡违约风险评分卡</h1>
  <p>基于 UCI「Default of Credit Card Clients」公开数据集（台湾信用卡，2005年4-9月账单）· Python 完整流水线</p>
</header>
<main>

<div class="kpis">{kpi}</div>

<section>
  <h2>1 · 业务问题与数据</h2>
  <div class="card">
    <p>信用风控的核心问题：<b>在放款之前，判断客户未来违约的概率</b>。本项目使用 UCI 公开数据集（{M['n_samples']:,} 个客户样本、23 个特征），目标变量为「下月是否违约」（违约率 {M['default_rate']}%）。特征涵盖授信额度、人口属性、6 个月的还款状态 / 账单金额 / 还款金额。</p>
    <p>完整流程：<span class="steps"><span class="step">数据清洗</span><span class="step">EDA</span><span class="step">分箱与 WOE/IV</span><span class="step">共线性筛选</span><span class="step">逻辑回归</span><span class="step">AUC/KS 评估</span><span class="step">评分卡转换</span><span class="step">PSI 稳定性</span></span></p>
    <p>清洗动作：剔除重复样本 {M['n_duplicates_removed']} 条；学历/婚姻字段中的未知类别码（0、5、6）归并为「其他」，避免哑变量陷阱；训练/测试按 8:2 分层抽样（{M['train_n']:,}/{M['test_n']:,}）。</p>
  </div>
</section>

<section>
  <h2>2 · 探索性分析</h2>
  <div class="card">
    <img src="charts/01_eda_overview.png" alt="EDA">
    <img src="charts/02_pay0_defaultrate.png" alt="PAY_0">
    <p class="figcap">{CHART_NAMES['01_eda_overview.png']}；{CHART_NAMES['02_pay0_defaultrate.png']}</p>
    <p><b>结论：</b>还款状态是最强的违约信号——最近一个月无逾期的客户违约率在 10%~15%，而逾期 2 个月及以上的客户违约率跃升至 50% 以上，梯度清晰，业务含义直白。</p>
  </div>
</section>

<section>
  <h2>3 · 特征分箱与 WOE/IV 筛选</h2>
  <div class="card">
    <img src="charts/03_iv_ranking.png" alt="IV">
    <p class="figcap">{CHART_NAMES['03_iv_ranking.png']}</p>
    <p><b>方法：</b>数值变量等频分箱（低频箱自动向邻箱合并），按 {esc(M['score_formula'].split('，')[0] if M.get('score_formula') else '标准定义')}计算各箱 WOE（证据权重）与整列 IV（信息值）；IV &lt; 0.02 的弱特征剔除。WOE 变换同时完成了两件事：把非线性关系线性化（对逻辑回归友好），并把所有特征统一到同一量纲。</p>
    <p><b>结论：</b>预测能力排序为「还款状态（IV 0.3~0.9，强）＞ 授信额度与还款金额（0.1~0.3，中）＞ 人口属性（&lt;0.02，剔除）」——与行业经验一致：历史行为比人口画像更能预测违约。</p>
  </div>
</section>

<section>
  <h2>4 · 逻辑回归建模与评估</h2>
  <div class="card">
    <img src="charts/04_model_eval.png" alt="模型评估">
    <p class="figcap">{figcap04}</p>
    {GBDT_IMG}
    {f'''<div class="grid2"><div>
      <h3>与 GBDT 基线的定量对比</h3>
      <table class="tbl"><thead><tr><th>模型</th><th>5折CV AUC</th><th>测试集 AUC</th></tr></thead><tbody>{GBDT_ROWS}</tbody></table>
    </div><div>
      <h3>为什么最终选择逻辑回归</h3>
      <p>默认超参的 HistGradientBoosting 只把测试集 AUC 提高约 <b>1 个百分点</b>（{GBDT_M['models'][0]['test_auc']:.3f} → {GBDT_M['models'][1]['test_auc']:.3f}），代价是失去系数可解释性——而评分卡的业务价值恰恰在于每个分数都能拆解到特征与分箱层面。此外注意：LR 使用 IV 筛选后的 14 个特征即可达到接近全特征的表现（口径差异见两表 AUC），说明筛选后信息几乎无损。</p>
      <p>这个对比是「可解释性 vs 性能」权衡的定量证据：当复杂模型只带来边际提升时，简单模型是更负责任的选择。</p>
    </div></div>''' if GBDT_M else ''}
    <div class="grid2">
      <div>
        <h3>入模特征与系数（WOE 尺度）</h3>
        <table class="tbl"><thead><tr><th>特征</th><th>系数</th></tr></thead><tbody>{coef_rows}</tbody></table>
      </div>
      <div>
        <h3>关键指标</h3>
        <table class="tbl">
          <tbody>
          <tr><th>5折交叉验证 AUC</th><td>{M['cv_auc_mean']:.3f} ± {M['cv_auc_std']:.3f}</td></tr>
          <tr><th>测试集 AUC</th><td>{M['auc_test']:.3f}</td></tr>
          <tr><th>训练集 AUC</th><td>{M['auc_train']:.3f}（与测试集差距小，无明显过拟合）</td></tr>
          <tr><th>KS 统计量</th><td>{M['ks']:.3f}</td></tr>
          </tbody>
        </table>
        <p>KS &gt; 0.4 在评分卡实务中属于区分度较好；共线性处理剔除了 {len(M['dropped_collinear'])} 个特征（|r|&gt;0.7 保留 IV 高者）。</p>
      </div>
    </div>
  </div>
</section>

<section>
  <h2>5 · 评分卡转换</h2>
  <div class="card">
    <p><b>{esc(M.get('score_formula', ''))}</b>。逻辑回归的对数几率被线性映射为直观的信用分数：分数越高风险越低。</p>
    <div class="grid2">{score_tables}</div>
    <p class="figcap">评分卡分值表（部分特征；基础分均摊到各特征，各箱得分 = 基础分 − 缩放系数 × 系数 × WOE）</p>
    <img src="charts/05_score_grade.png" alt="评分分组">
    <p class="figcap">{CHART_NAMES['05_score_grade.png']}</p>
    <p><b>结论：</b>按评分十分位分组后，实际违约率从低分组的约 {M['grade_badrate'][0]:.1f}% 递减至高分组的约 {M['grade_badrate'][-1]:.1f}%，排序能力与业务预期一致。各组违约率：{grade}。</p>
  </div>
</section>

<section>
  <h2>6 · 稳定性与模型局限（诚实声明）</h2>
  <div class="card">
    <p><b>稳定性：</b>训练/测试集评分分布的 PSI = {M['psi_train_vs_test']:.3f}（&lt;0.1 视为稳定），两集平均分几乎一致（{M['score_mean_train']:.0f} / {M['score_mean_test']:.0f}）。</p>
    <ul>
      <li>本数据集无时间戳，PSI 验证的是随机划分下的分布一致性，<b>不是真正的时间外推稳定性</b>；上线场景应按时间切分验证并按月监控 PSI。</li>
      <li>分箱未强制 WOE 单调（个别箱可能轻微U型），生产实践中应做单调性约束。</li>
      <li>违约率 {M['default_rate']}% 为中度不平衡，逻辑回归对此较稳健，但阈值选择会影响精确率/召回率的权衡，本报告以排序能力（AUC/KS）为主指标。</li>
      <li>可改进方向：与梯度提升树（GBDT）基线对比、引入特征交叉、按时间外推验证。</li>
    </ul>
  </div>
</section>

<footer>
  数据来源：<a href="https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients" target="_blank" rel="noopener">UCI Default of Credit Card Clients</a> ·
  Yeh &amp; Lien (2009) · 分析代码见项目仓库
 · <a href="/">← 返回作品集主页</a></footer>
</main>
</body>
</html>"""

(PUB / "index.html").write_text(html, encoding="utf-8")
print(f"报告已生成: {PUB / 'index.html'}")
