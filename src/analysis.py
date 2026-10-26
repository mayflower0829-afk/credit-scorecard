# -*- coding: utf-8 -*-
"""信用卡违约风险评分卡 — 完整分析流水线

数据：UCI "Default of Credit Card Clients"（台湾信用卡 2005 年，30000 样本，23 特征）
流程：数据清洗 → EDA → 分箱与 WOE/IV → 共线性筛选 → 逻辑回归 → AUC/KS 评估 → 评分卡转换 → PSI 稳定性
输出：output/ 下的图表 PNG 与 metrics.json（报告页由此生成）

分箱实现说明：数值变量用「内部边界数组 + 箱号」方式，train/test 严格共用同一套边界，
避免字符串匹配区间标签导致的 WOE 错配；分类变量对 test 中未见过类别单独给 WOE。
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

metrics = {}

# ========== 1. 数据准备与清洗 ==========
COLS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "target"]
df = pd.read_excel(ROOT / "data" / "uci_default.xls", header=1)
df.columns = COLS
df = df.drop(columns=["ID"]).reset_index(drop=True)

n_total = len(df)
dup = int(df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# 未知类别归并：EDUCATION 的 0/5/6 归为 4=其他，MARRIAGE 的 0 归为 3=其他
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

metrics["n_samples"] = int(len(df))
metrics["n_duplicates_removed"] = dup
metrics["default_rate"] = round(float(df["target"].mean()) * 100, 2)

NUM_COLS = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
            "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
            "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
CAT_COLS = ["SEX", "EDUCATION", "MARRIAGE", "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

train, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)
metrics["train_n"], metrics["test_n"] = int(len(train)), int(len(test))

# ========== 2. EDA 图 ==========
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
axes[0].hist(df["LIMIT_BAL"] / 10000, bins=40, color="#4F6EF2")
axes[0].set_title("授信额度分布（万元）")
axes[1].hist(df["AGE"], bins=40, color="#10B981")
axes[1].set_title("年龄分布")
axes[2].bar(["正常", "违约"], [int((df["target"] == 0).sum()), int(df["target"].sum())],
            color=["#10B981", "#E5484D"])
axes[2].set_title(f"违约分布（违约率 {metrics['default_rate']}%）")
for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "01_eda_overview.png", bbox_inches="tight")
plt.close(fig)

pay0_rate = df.groupby("PAY_0")["target"].agg(["mean", "count"])
fig, ax = plt.subplots(figsize=(7.5, 3.6))
ax.bar(pay0_rate.index.astype(str), pay0_rate["mean"] * 100, color="#4F6EF2")
for x, (rate, cnt) in enumerate(zip(pay0_rate["mean"], pay0_rate["count"])):
    ax.text(x, rate * 100 + 1, f"{rate * 100:.0f}%\n(n={cnt})", ha="center", fontsize=8)
ax.set_ylabel("违约率 %")
ax.set_title("最近一个月还款状态（PAY_0）与违约率")
ax.set_xlabel("PAY_0：-2/-1/0=无逾期，1~8=逾期月数")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "02_pay0_defaultrate.png", bbox_inches="tight")
plt.close(fig)

# ========== 3. 分箱与 WOE/IV ==========
def agg_bad_total(bin_idx: np.ndarray, y: np.ndarray):
    d = pd.DataFrame({"b": bin_idx, "y": y})
    g = d.groupby("b")["y"].agg(bad="sum", total="count")
    g["good"] = g["total"] - g["bad"]
    return g

def woe_from_agg(g: pd.DataFrame):
    gd = g["good"].sum()
    bd = g["bad"].sum()
    dist_g = g["good"] / gd
    dist_b = g["bad"] / bd
    woe = np.log(dist_g / dist_b)
    iv = float(((dist_g - dist_b) * woe).sum())
    return woe, iv

def bin_numeric(x: pd.Series, y: pd.Series, n_bins: int = 6, min_count: int = 30):
    """等频分箱 + 低频相邻合并。返回 edges(内部边界)、箱号、WOE 数组、IV"""
    x = x.astype(float)
    yv = np.asarray(y)
    qs = np.unique(np.quantile(x, np.linspace(0, 1, n_bins + 1)))
    edges = qs[1:-1] if len(qs) > 2 else np.array([])
    if edges.size == 0:
        b = np.zeros(len(x), dtype=int)
        g = agg_bad_total(b, yv)
        woe, iv = woe_from_agg(g)
        return edges, b, woe, iv, g
    for _ in range(40):
        b = np.searchsorted(edges, x, side="left")
        g = agg_bad_total(b, yv)
        if len(g) <= 2:
            break
        small = g[g["total"] < min_count]
        if small.empty:
            break
        i = int(small["total"].idxmin())
        if i == 0:
            edges = np.delete(edges, 0)          # 首箱并入第二箱
        elif i == len(g) - 1:
            edges = np.delete(edges, -1)         # 末箱并入前箱
        else:
            edges = np.delete(edges, i - 1)      # 中间箱并入左箱
    b = np.searchsorted(edges, x, side="left")
    g = agg_bad_total(b, yv)
    woe, iv = woe_from_agg(g)
    return edges, b, np.asarray(woe), iv, g

def bin_categorical(x: pd.Series, y: pd.Series, min_count: int = 30):
    """低频类合并为 __RARE__。返回映射表、train 箱标签、WOE dict、IV、未见类 WOE"""
    s = x.astype(str)
    yv = np.asarray(y)
    d = pd.DataFrame({"b": s, "y": yv})
    cnt = d["b"].value_counts()
    rare = cnt[cnt < min_count].index
    mp = {k: ("__RARE__" if k in set(rare) else k) for k in cnt.index}
    d["b"] = d["b"].map(mp)
    g = agg_bad_total(d["b"].values, yv)
    woe, iv = woe_from_agg(g)
    gd = g["good"].sum()
    bd = g["bad"].sum()
    unseen_woe = float(np.log(gd / bd))
    return mp, d["b"].values, {k: float(v) for k, v in woe.items()}, iv, unseen_woe, g

def bin_labels_numeric(edges: np.ndarray):
    if edges.size == 0:
        return ["全部"]
    bounds = [-np.inf, *edges, np.inf]
    return [f"({bounds[i]:.0f}, {bounds[i + 1]:.0f}]" for i in range(len(bounds) - 1)]

binners = {}          # feature -> 分箱信息
iv_rows = []
bin_tr = {}           # feature -> train 箱标签(数组)
bin_te = {}

for col in NUM_COLS + CAT_COLS:
    is_num = col in NUM_COLS
    if is_num:
        edges, b_tr, woe, iv, g = bin_numeric(train[col], train["target"])
        b_te = np.searchsorted(edges, test[col].astype(float), side="left")
        labels = bin_labels_numeric(edges)
        woe_map = {labels[i]: float(woe[i]) for i in range(len(woe))}
        binners[col] = {"type": "numeric", "edges": edges.tolist(), "woe_by_bin": woe.tolist()}
        bin_tr[col], bin_te[col] = b_tr, np.clip(b_te, 0, len(woe) - 1)
    else:
        mp, b_tr, woe_map, iv, unseen_woe, g = bin_categorical(train[col], train["target"])
        b_te = test[col].astype(str).map(lambda v: mp.get(v, "__UNSEEN__")).values
        binners[col] = {"type": "categorical", "woe_map": woe_map, "unseen_woe": unseen_woe}
        bin_tr[col], bin_te[col] = b_tr, b_te
    iv_rows.append({"feature": col, "iv": round(iv, 4)})

iv_table = pd.DataFrame(iv_rows).sort_values("iv", ascending=False).reset_index(drop=True)
fig, ax = plt.subplots(figsize=(7.5, 5.2))
top = iv_table[::-1]
colors = ["#E5484D" if v >= 0.3 else "#4F6EF2" if v >= 0.1 else "#9AA5B1" for v in top["iv"]]
ax.barh(top["feature"], top["iv"], color=colors)
ax.axvline(0.02, ls="--", c="#F59E0B", lw=1)
ax.axvline(0.1, ls="--", c="#10B981", lw=1)
ax.text(0.022, -0.6, "0.02 弱", fontsize=8, color="#F59E0B")
ax.text(0.102, -0.6, "0.1 中", fontsize=8, color="#10B981")
ax.set_xlabel("信息值 IV")
ax.set_title("特征 IV 排序（预测能力）")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "03_iv_ranking.png", bbox_inches="tight")
plt.close(fig)
metrics["iv_table"] = iv_table.to_dict("records")

# ========== 4. 特征筛选（IV 阈值 + WOE 后共线性） ==========
keep = [r["feature"] for r in metrics["iv_table"] if r["iv"] >= 0.02]

def to_woe(cols, bin_index: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=range(len(next(iter(bin_index.values())))))
    for c in cols:
        info = binners[c]
        idx = np.asarray(bin_index[c])
        if info["type"] == "numeric":
            w = np.asarray(info["woe_by_bin"])
            out[c] = w[np.clip(idx, 0, len(w) - 1)]
        else:
            out[c] = [info["woe_map"].get(v, info["unseen_woe"]) for v in idx]
    return out

woe_tr_full = to_woe(keep, bin_tr)
woe_te_full = to_woe(keep, bin_te)

corr = woe_tr_full.corr().abs()
iv_lookup = {r["feature"]: r["iv"] for r in metrics["iv_table"]}
cols_sorted = sorted(keep, key=lambda c: -iv_lookup[c])
drop_set = set()
for i, a in enumerate(cols_sorted):
    if a in drop_set:
        continue
    for b in cols_sorted[i + 1:]:
        if b in drop_set:
            continue
        if corr.loc[a, b] > 0.7:
            drop_set.add(b)
selected = [c for c in cols_sorted if c not in drop_set]
metrics["selected_features"] = selected
metrics["dropped_collinear"] = sorted(drop_set)

X_tr, y_tr = woe_tr_full[selected], train["target"].values
X_te, y_te = woe_te_full[selected], test["target"].values

# ========== 5. 逻辑回归建模与评估 ==========
model = LogisticRegression(max_iter=2000, C=1.0)
cv_auc = cross_val_score(model, X_tr, y_tr, cv=5, scoring="roc_auc")
model.fit(X_tr, y_tr)

proba_te = model.predict_proba(X_te)[:, 1]
auc_te = roc_auc_score(y_te, proba_te)
auc_tr = roc_auc_score(y_tr, model.predict_proba(X_tr)[:, 1])
fpr, tpr, _ = roc_curve(y_te, proba_te)
ks = float(np.max(tpr - fpr))

metrics["cv_auc_mean"] = round(float(cv_auc.mean()), 4)
metrics["cv_auc_std"] = round(float(cv_auc.std()), 4)
metrics["auc_train"] = round(float(auc_tr), 4)
metrics["auc_test"] = round(float(auc_te), 4)
metrics["ks"] = round(ks, 4)
metrics["coef"] = {c: round(float(v), 4) for c, v in zip(selected, model.coef_[0])}
metrics["intercept"] = round(float(model.intercept_[0]), 4)

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(fpr, tpr, color="#4F6EF2", lw=2, label=f"ROC (AUC={auc_te:.3f}, KS={ks:.3f})")
axes[0].plot([0, 1], [0, 1], "--", c="#9AA5B1", lw=1)
axes[0].set_xlabel("假阳率 FPR")
axes[0].set_ylabel("真阳率 TPR")
axes[0].set_title(f"测试集 ROC（5折CV AUC={cv_auc.mean():.3f}±{cv_auc.std():.3f}）")
axes[0].legend()
axes[0].spines[["top", "right"]].set_visible(False)

bins_s = np.linspace(proba_te.min(), proba_te.max(), 30)
axes[1].hist(proba_te[y_te == 0], bins=bins_s, alpha=0.7, label="正常客户", color="#10B981")
axes[1].hist(proba_te[y_te == 1], bins=bins_s, alpha=0.7, label="违约客户", color="#E5484D")
axes[1].set_xlabel("模型预测违约概率")
axes[1].set_title("测试集预测概率分布")
axes[1].legend()
axes[1].spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "04_model_eval.png", bbox_inches="tight")
plt.close(fig)

# ========== 6. 评分卡转换 ==========
PDO = 40.0
BASE_SCORE = 600.0
BASE_ODDS = 20.0  # 600 分对应 20:1 的好坏比
B = PDO / np.log(2)
A = BASE_SCORE - B * np.log(BASE_ODDS)

def to_score(p):
    odds = (1 - p) / np.clip(p, 1e-9, None)
    return A + B * np.log(odds)

scores_te = to_score(proba_te)
metrics["score_formula"] = f"Score = {A:.1f} + {B:.2f} × ln(好坏比)，PDO=40，600分=20:1"

score_bins = pd.qcut(pd.Series(scores_te), 10, duplicates="drop")
grad = pd.DataFrame({"bin": score_bins, "bad": y_te}).groupby("bin", observed=True)["bad"].agg(["mean", "count"])
grad["mean"] = grad["mean"] * 100
fig, ax = plt.subplots(figsize=(8.5, 3.8))
labels = [str(i + 1) for i in range(len(grad))]
ax.bar(labels, grad["mean"], color="#4F6EF2")
for x, (m, c) in enumerate(zip(grad["mean"], grad["count"])):
    ax.text(x, m + 0.4, f"{m:.1f}%", ha="center", fontsize=9)
ax.set_xlabel("评分十分位组（1=低分高风险 … 10=高分低风险）")
ax.set_ylabel("实际违约率 %")
ax.set_title("测试集：评分分组与实际违约率（应大体单调）")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "05_score_grade.png", bbox_inches="tight")
plt.close(fig)
metrics["grade_badrate"] = [round(float(v), 2) for v in grad["mean"]]

# 单变量分值表（基础分均摊 + 特征各箱分值）
score_table = {}
for c in selected:
    info = binners[c]
    beta = float(model.coef_[0][selected.index(c)])
    base_part = A / len(selected)
    if info["type"] == "numeric":
        w = info["woe_by_bin"]
        score_table[c] = {bin_labels_numeric(np.asarray(info["edges"]))[i]:
                          round(float(base_part - B * beta * w[i]), 1)
                          for i in range(len(w))}
    else:
        score_table[c] = {k: round(float(base_part - B * beta * v), 1)
                          for k, v in info["woe_map"].items()}
metrics["scorecard_table"] = score_table

# ========== 7. PSI 稳定性 ==========
def psi(expected, actual, bins=10):
    qs = np.quantile(expected, np.linspace(0, 1, bins + 1))
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e = np.clip(e, 1e-4, None)
    a = np.clip(a, 1e-4, None)
    return float(np.sum((a - e) * np.log(a / e)))

scores_tr = to_score(model.predict_proba(X_tr)[:, 1])
psi_val = psi(scores_tr, scores_te)
metrics["psi_train_vs_test"] = round(psi_val, 4)
metrics["score_mean_train"] = round(float(np.mean(scores_tr)), 1)
metrics["score_mean_test"] = round(float(np.mean(scores_te)), 1)

# ========== 8. 保存 ==========
(OUT / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({k: metrics[k] for k in
                  ["n_samples", "default_rate", "selected_features", "dropped_collinear",
                   "cv_auc_mean", "auc_test", "ks", "psi_train_vs_test",
                   "score_mean_train", "score_mean_test"]},
                 ensure_ascii=False, indent=2))
print("图表与 metrics.json 已输出到", OUT)
