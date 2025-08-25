# 项目背景

本项目基于 2021 年以来 A 股全市场日度数据（5000+ 股票，约 1300 个交易日），研究一个基于主动买卖金额的因子，并检验其对未来收益率的预测能力。
该项目来源于量化实习笔试题，经过扩展整理后作为完整因子研究案例。

# 研究目标

1. 构造一个反映资金主动买卖行为的因子：
         $$
         a = \frac{主动买入金额 - 主动卖出金额}{主动买入金额 + 主动卖出金额}
         $$

2. 在 20 日滚动窗口内，筛选最低 20% 收益率的日期（低收益日），对这些日期的 𝑎 取均值，得到最终因子值。

3. 使用 IC 分析 和 分组收益回测 验证该因子是否具有选股价值。

# 方法论

1.数据清洗

- 处理停牌 / 无成交数据（buy/sell 金额为 0）；

- 删除缺失收益率行；

2. 因子构造

- 每只股票逐日计算 a;

- 滚动 20 日，取最低 20% 收益日，均值作为因子值。

3. IC 分析

- 计算每日横截面 Spearman Rank IC：
  $$
IC_t = corr(factor_{i,t}, \; return_{i,t+1})
  $$        
- 输出均值、标准差、t 检验和信息比率（IR）。

4. 分组收益回测

- 每日按因子值将股票分为 5 组；

- 计算组内未来收益率，构建等权组合；

- 绘制累计收益曲线和多空组合（Q5–Q1）。

# 结果展示

1. IC 时间序列与分布：评估因子与未来收益的相关性和稳定性。

2. 分组累计收益图：验证因子是否具有单调性与超额收益。

3. 多空组合表现：展示因子是否能构建盈利策略。

# 技术实现

1. 语言：Python

2. 库：pandas, numpy, scipy, matplotlib

3. 结构：

    - factor_calc.py → 数据处理与因子构造
    - ic_analysis.py → IC 计算与统计检验
    - group_backtest.py → 分组收益回测
    - report.ipynb → 可视化与结果分析

# 总结与展望

1. 本项目展示了一个完整的 量化因子研究流程：数据清理 → 因子设计 → IC 检验 → 分组回测 → 可视化。

2. 验证了主动买卖行为因子与未来收益的关系，为后续策略组合与多因子研究提供了思路。

3. 未来可扩展方向：

         - 引入更多市场因子进行对比；
         
         - 加入行业/市值中性化处理；
         
         - 考虑更长持有期的因子检验。

# 代码解读

本 Notebook 主要完成三件事：
原始数据检查与清洗（读取、列名/唯一标的、异常值与覆盖率、连续性检查）；


因子评估之 IC 分析（基于 data_with_factor.csv 中已计算好的 factor 与 fwd_ret1）；


因子分层收益（按日分组 Q1~Q5，计算下一期组收益与累计净值/多空）。


说明：factor 的计算在前置流程完成，本 Notebook 直接使用 data_with_factor.csv 中的结果做评估。

## 1.读取数据与基本查看
目的：读入 data.csv，查看字段、预览数据，以确定后续清洗口径。
关键片段：
import pandas as pd

### ---- 读数：按需调整 encoding / sep ----
df = pd.read_csv("data.csv", sep=",", encoding="utf-8", low_memory=False)

### ---- 展示前 10 行 ----
from IPython.display import display
display(df.head(10))

### ---- 打印列名列表 ----
print("\nColumns list:")
print(list(df.columns))


读入 CSV，保留原始列类型；
预览前 10 行，打印列名，为后续字段对齐做准备。


## 2. 标的列表与抽样核对
目的：确认样本覆盖的股票数量与具体代码，便于抽检与问题复现。
关键片段：
df = pd.read_csv("data.csv")

unique_ids = df['order_book_id'].unique()
print(f"共有 {len(unique_ids)} 个唯一的 order_book_id")
print("具体列表:")
print(unique_ids)

抽样核对（按标的与日期）：
### 抽查某标的在特定日期区间的数据
df = pd.read_csv("data.csv")
stock = "688683.XSHG"
tmp = df[(df['order_book_id'] == stock) &
         (df['date'].between("2021-04-21", "2021-04-22"))]
print(tmp[['date', 'pct_rate']])


## 3. 覆盖率与连续性检查
目的：统计各标的在样本期内的观测覆盖度与缺口，识别数据质量风险点。
覆盖率（相对全局交易日集合）：
df = pd.read_csv("data_clean.csv")
ticker_date_sets = df.groupby("order_book_id")["date"].apply(set)
global_date_set = set(df["date"].unique())
cover_ratio = ticker_date_sets.apply(lambda s: len(s) / len(global_date_set))

threshold = 0.5
bad_tickers = cover_ratio[cover_ratio < threshold].index
print("\n>>> 各股票覆盖率统计："); print(cover_ratio.describe())
print(f"\n>>> 覆盖率低于 {threshold:.0%} 的股票数量：{len(bad_tickers)}")

期内连续性（按各自起止区间）：
df = pd.read_csv("data.csv", parse_dates=["date"])
df["date"] = df["date"].dt.normalize()
df = df.drop_duplicates(subset=["order_book_id", "date"])

master_dates = pd.Index(df["date"].sort_values().unique())
g = df.groupby("order_book_id")["date"]
minmax = g.agg(["min", "max"])

### 每个 ticker 在 [min, max] 内用 master_dates 检查缺失天数
res = pd.DataFrame(rows, columns=["order_book_id", "missing_count"]) \
        .sort_values("missing_count", ascending=False) \
        .reset_index(drop=True)

print(f"数据不连续（期内有缺口）的 ticker 数: {(res['missing_count']>0).sum()}")
print(res.head(20))

计算每只股票相对全样本交易日的覆盖比；


在各自起止区间内检查“应该有但缺失”的交易日数量，输出缺口最大的标的。



## 4.删除空白交易日/空白数据
删除空白行，为下一步计算因子和分组准备干净且整齐的数据
### 先确保 (ticker, date) 唯一（防止重复）
df = df.drop_duplicates(subset=["order_book_id", "date"])

### 需要检查的字段
cols = ["buy_volume", "buy_value", "sell_volume", "sell_value", "pct_rate"]

### 条件：任意一个字段为 0
mask = (df[cols] == 0).any(axis=1)

### 清理：去掉满足条件的整行
df_clean = df.loc[~mask].copy()

## 5. 计算因子Factor
该脚本为每只股票构建一个 滚动因子（factor）：在每个日期，回溯最近 W 天，找到 pct_rate 最低的 k 天，取这些日期对应的 买卖失衡指标 a 的平均值。结果写入 CSV，作为新列 factor。

（1）核心逻辑（Numba 加速）
对每只股票、每个交易日 t：
回看最近 W 天数据。


找到其中 k = max(1, int(W * LAMBDA)) 个 pct_rate 最低的日期。

对这些日期对应的 a 取均值。

若窗口未满或无有效数据，则结果为 NaN。

核心计算函数（用 Numba JIT 加速）：
@njit
def rolling_factor_numba(pct_rate: np.ndarray, a: np.ndarray, window: int, k: int) -> np.ndarray:
    # 在 W 日窗口中，挑选 pct_rate 最低的 k 日，
    # 计算这些日期对应的 a 的平均值；支持 NaN 处理与窗口不足。
    ...
    idx = np.argpartition(pr_buf[:m], kk - 1)[:kk]  # 部分排序取前 k 个最小值
    ...

（2）输入与预处理
输入 CSV 应包含字段：
 date, order_book_id, buy_value, sell_value, pct_rate, ...
预处理步骤：
df = pd.read_csv("data_clean.csv", parse_dates=["date"])
df["date"] = df["date"].dt.normalize()
df = df.drop_duplicates(subset=["order_book_id", "date"]).sort_values(["order_book_id", "date"])

计算买卖失衡 a：
denom = (df["buy_value"].astype(float) + df["sell_value"].astype(float))
df["a"] = (df["buy_value"] - df["sell_value"]) / denom.replace(0, np.nan)

当 buy_value + sell_value == 0 时，a 设为 NaN。
（3）参数设置
文件顶部定义：
W = 20          # 回溯窗口（天）
LAMBDA = 0.2    # “最差 k 日”比例
K = max(1, int(W * LAMBDA))  # 例如：20% × 20 = 4

增大 W → 平滑因子。
增大 LAMBDA → 纳入更多“最差日”。
K 至少为 1。


（4）分组计算
按股票逐组计算：
for _, g in df.groupby("order_book_id", sort=False):
    pr = g["pct_rate"].to_numpy(dtype=np.float64)
    av = g["a"].to_numpy(dtype=np.float64)
    fac = rolling_factor_numba(pr, av, W, K)
    factors.append(pd.Series(fac, index=g.index))

df["factor"] = pd.concat(factors).sort_index()

窗口 NaN 处理规则：
窗口内 pct_rate 为 NaN → 忽略。


若选出的日期 a 全为 NaN → 因子结果为 NaN。


每只股票前 W-1 天因窗口不足 → NaN。


（5）输出与日志
最终结果写入 CSV：
out_path = "data_with_factor.csv"
df.to_csv(out_path, index=False)
print(f"OK -> {out_path}")
print(f"rows: {len(df):,}, factor NaN (前 {W-1} 天+无效窗口): {df['factor'].isna().sum():,}")

生成 data_with_factor.csv，新增 factor 列，可直接用于后续分组回测。

## 6.IC 分析（因子信息含量）
目的：评估 factor 与下一期收益 fwd_ret1 的横截面秩相关（Spearman）。
关键片段：
import pandas as pd, numpy as np, matplotlib.pyplot as plt

CSV_PATH = "data_with_factor.csv"
DATE_COL, FACTOR_COL, RET_COL = "date", "factor", "fwd_ret1"
METHOD, ROLL_WIN = "spearman", 126

df = pd.read_csv(CSV_PATH, parse_dates=[DATE_COL]).dropna(subset=[FACTOR_COL, RET_COL])

def _daily_ic(g: pd.DataFrame) -> float:
    if g[FACTOR_COL].count() < 3 or g[RET_COL].count() < 3:
        return np.nan
    return g[FACTOR_COL].corr(g[RET_COL], method=METHOD)

ic_series = (df.groupby(DATE_COL, sort=True).apply(_daily_ic)
               .rename(f"IC_{METHOD}").astype(float))

### 汇总指标
ic = ic_series.dropna()
n = ic.shape[0]
ic_mean, ic_std = ic.mean(), ic.std(ddof=1)
se = ic_std / np.sqrt(n); t_stat = ic_mean / se if se > 0 else np.nan
icir = ic_mean / ic_std if ic_std > 0 else np.nan
pos_ratio = (ic > 0).mean()

print({"N_days": n, "IC_mean": ic_mean, "IC_std": ic_std,
       "t_stat": t_stat, "ICIR": icir, "Pos_ratio(IC>0)": pos_ratio})

### 时序与滚动均值
plt.figure(figsize=(10,4))
ic_series.plot(linewidth=1, label="Daily IC")
ic_series.rolling(ROLL_WIN, min_periods=max(5, ROLL_WIN//5)).mean() \
         .plot(linewidth=2, alpha=0.9, label=f"Rolling mean ({ROLL_WIN})")
plt.axhline(0.0, linewidth=1); plt.legend(); plt.tight_layout(); plt.show()

按交易日分组，计算当日横截面 Spearman 相关作为 IC_t；

汇总输出 均值、标准差、t 值、ICIR、IC>0 占比；

绘制 IC_t 时间序列与 126 日滚动均值，观察稳定性与阶段性。

## 7. 因子分层收益（Q1~Q5）
该代码实现了一个基于因子的分组收益分析与可视化工具，主要流程如下：
数据预处理：
 从 data_with_factor.csv 读取数据，仅保留含有因子值和次日收益的数据：

 df = df.dropna(subset=["factor", "fwd_ret1"])

分组逻辑：
 每个交易日将股票按照因子值进行秩排序（相同值用 method="first" 打破并列），再划分为 N_GROUPS 组：

 rank_pct = g["factor"].rank(method="first", pct=True)
grp = np.ceil(rank_pct * N_GROUPS).astype(int)


收益计算：
 对每组计算次日收益的等权平均，并生成多空组合：

 long_short = group_returns[f"Q{N_GROUPS}"] - group_returns["Q1"]
group_returns["LS"] = long_short


可视化：

每日分组收益时序图。

多空每日收益时序图。

各组累计净值曲线。

多空累计净值曲线。








