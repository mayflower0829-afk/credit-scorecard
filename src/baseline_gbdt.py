# -*- coding: utf-8 -*-
"""GBDT 基线对比：回答"评分卡为什么不用复杂模型"的定量版本。

对比三个模型（同一 train/test 划分，random_state=42）：
  1. Logistic Regression on WOE（本项目主模型）
  2. HistGradientBoosting on 原始特征（现代 GBDT 基线）
  3. HistGradientBoosting on WOE 特征
输出 output/gbdt_metrics.json 与 09_gbdt_compare.png，供报告页引用。
"""
import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import cross_val_score, train_test_split

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "Arial Unicode MS", "STHeiti"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

# ---- 数据准备（与 analysis.py 完全一致的清洗与划分） ----
COLS = ["ID", "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
        "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
        "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
        "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6", "target"]
df = pd.read_excel(ROOT / "data" / "uci_default.xls", header=1)
df.columns = COLS
df = df.drop(columns=["ID"]).drop_duplicates().reset_index(drop=True)
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})

train, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=42)

# ---- 复用 analysis.py 的 WOE 管线（重跑分箱，保证同口径） ----
import subprocess
import sys
sys.path.insert(0, str(ROOT / "src"))
# 直接运行 analysis.py 会重复输出图表；改为 import 其中的函数过于耦合，
# 这里以最小方式重建 WOE 特征（与 analysis.py 相同的 bin_numeric / bin_categorical）。
from analysis import NUM_COLS, CAT_COLS, bin_numeric, bin_categorical  # noqa: E402

binners, bin_tr, bin_te = {}, {}, {}
for col in NUM_COLS + CAT_COLS:
    if col in NUM_COLS:
        edges, b_tr, woe, iv, g = bin_numeric(train[col], train["target"])
        b_te = np.clip(np.searchsorted(edges, test[col].astype(float), side="left"), 0, len(woe) - 1)
        binners[col] = {"type": "numeric", "woe_by_bin": np.asarray(woe)}
        bin_tr[col], bin_te[col] = b_tr, b_te
    else:
        mp, b_tr, woe_map, iv, unseen_woe, g = bin_categorical(train[col], train["target"])
        b_te = test[col].astype(str).map(lambda v: mp.get(v, "__UNSEEN__")).values
        binners[col] = {"type": "categorical", "woe_map": woe_map, "unseen_woe": unseen_woe}
        bin_tr[col], bin_te[col] = b_tr, b_te

def to_woe(cols, bin_index):
    out = pd.DataFrame(index=range(len(next(iter(bin_index.values())))))
    for c in cols:
        info = binners[c]
        idx = np.asarray(bin_index[c])
        if info["type"] == "numeric":
            w = info["woe_by_bin"]
            out[c] = w[np.clip(idx, 0, len(w) - 1)]
        else:
            out[c] = [info["woe_map"].get(v, info["unseen_woe"]) for v in idx]
    return out

woe_cols = [c for c in NUM_COLS + CAT_COLS]
X_woe_tr, X_woe_te = to_woe(woe_cols, bin_tr), to_woe(woe_cols, bin_te)
X_raw_tr = train[NUM_COLS + CAT_COLS].reset_index(drop=True)
X_raw_te = test[NUM_COLS + CAT_COLS].reset_index(drop=True)
y_tr, y_te = train["target"].values, test["target"].values

models = {
    "LR(WOE) 主模型": LogisticRegression(max_iter=2000).fit(X_woe_tr, y_tr),
    "HistGB(原始特征)": HistGradientBoostingClassifier(random_state=42).fit(X_raw_tr, y_tr),
    "HistGB(WOE特征)": HistGradientBoostingClassifier(random_state=42).fit(X_woe_tr, y_tr),
}

rows, curves = [], {}
for name, mdl in models.items():
    Xte = X_woe_te if "WOE" in name else X_raw_te
    Xtr = X_woe_tr if "WOE" in name else X_raw_tr
    cv = cross_val_score(mdl, Xtr, y_tr, cv=5, scoring="roc_auc")
    proba = mdl.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(y_te, proba)
    fpr, tpr, _ = roc_curve(y_te, proba)
    curves[name] = (fpr, tpr, auc)
    rows.append({"model": name, "cv_auc": round(float(cv.mean()), 4),
                 "cv_std": round(float(cv.std()), 4), "test_auc": round(float(auc), 4)})

M = {"models": rows,
     "note": "同一 8:2 分层划分（random_state=42）；GBDT 为默认超参未调优"}
(OUT / "gbdt_metrics.json").write_text(json.dumps(M, ensure_ascii=False, indent=2), encoding="utf-8")

fig, ax = plt.subplots(figsize=(7.5, 4.6))
styles = {"LR(WOE) 主模型": (C_MAIN := "#4F6EF2", "-"),
          "HistGB(原始特征)": ("#E5484D", "--"),
          "HistGB(WOE特征)": ("#D97706", ":")}
for name, (fpr, tpr, auc) in curves.items():
    c, ls = styles[name]
    ax.plot(fpr, tpr, ls, color=c, lw=2, label=f"{name}（AUC {auc:.3f}）")
ax.plot([0, 1], [0, 1], "--", c="#9AA5B1", lw=1)
ax.set_xlabel("假阳率 FPR")
ax.set_ylabel("真阳率 TPR")
ax.set_title("测试集 ROC：LR(WOE) vs GBDT 基线")
ax.legend(fontsize=9, loc="lower right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig(OUT / "09_gbdt_compare.png", bbox_inches="tight")
plt.close(fig)

print(json.dumps(rows, ensure_ascii=False, indent=2))
