#!/usr/bin/env python3
"""一次性补丁：为 final_improve.ipynb 写入 SMOKE_TEST 轻量跑通逻辑。"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
NB = ROOT / "final_improve.ipynb"

CELL8_NEW = r'''import os

# 【关键修复】处理 macOS 下可能出现的 OpenMP 冲突报错，必须在 import torch 等依赖之前设置！
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import gc
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

# 检查是否有 MPS（Metal Performance Shaders）加速
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

OUTPUT_DIR = '../outputs'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =============================================================================
# SMOKE_TEST：极速冒烟 —— 只验证管线能否跑通（无数值/论文意义）
# 正式复现前请改为 False；为 True 时使用独立 cache/models 子目录，避免覆盖全样本缓存
# =============================================================================
SMOKE_TEST = True

if SMOKE_TEST:
    # 冒烟仍快于全量，但显著提高子样本规模，避免测试窗/因子回归因观测过少报错
    SMOKE_READ_CSV_NROWS = 6_000_000
    SMOKE_DATE_START = "2008-01-01"
    SMOKE_DATE_END = "2019-12-31"
    SMOKE_MAX_PERMNOS = 350
    # 只跑 (L, F) 任务列表；全量请 [(5, 5), (20, 20), (60, 60)]
    SMOKE_TASKS = [(5, 5)]
    SMOKE_MAX_EPOCHS = 18
    SMOKE_SEEDS = [42]
    SMOKE_BATCH_SIZE = 128
    SMOKE_NORM_SAMPLES = 10_240
    SMOKE_LIQUIDITY_TOP_N = 700
    SMOKE_FORCE_REBUILD = True
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models_smoke')
    CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache_smoke')
else:
    SMOKE_READ_CSV_NROWS = None
    SMOKE_DATE_START = None
    SMOKE_DATE_END = None
    SMOKE_MAX_PERMNOS = None
    SMOKE_TASKS = [(5, 5), (20, 20), (60, 60)]
    SMOKE_MAX_EPOCHS = 50
    SMOKE_SEEDS = [42, 123, 456, 789, 101112]
    SMOKE_BATCH_SIZE = 128
    SMOKE_NORM_SAMPLES = 50_000
    SMOKE_LIQUIDITY_TOP_N = None
    SMOKE_FORCE_REBUILD = False
    MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
    CACHE_DIR = os.path.join(OUTPUT_DIR, 'cache')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

if SMOKE_TEST:
    print("[SMOKE_TEST] ON — 使用", CACHE_DIR, "|", MODEL_DIR)
else:
    print("[SMOKE_TEST] OFF — 全样本复现模式")
'''

CELL10_NEW = r'''import ast
from datetime import datetime
import os
import gc
import matplotlib.pyplot as plt

# ==========================================
# 第一部分：数据预处理与特征构建 (Step 1 & 2)
# ==========================================
def load_and_preprocess_data(
    data_path,
    cache_dir='../outputs/cache',
    force_rebuild=False,
    liquidity_top_n=None,
    nrows=None,
    cache_extra_tag='',
):
    """读取、清洗数据，并计算复权价、因变量标签。

    liquidity_top_n:
        None — 不做月度市值筛选（全样本，与作者默认 stocks_for_train=\"all\" 一致）
        正整数 — 每月按 dlycap 保留市值前 N 的股票（历史 notebook 的 top1000 行为）
    nrows:
        传给 pd.read_csv 的最大行数；冒烟测试用大幅缩短读盘时间
    cache_extra_tag:
        附加在缓存文件名上，避免与全样本 pkl 冲突（如 '_smoke'）
    """
    liq_core = 'all' if liquidity_top_n is None else f'top{int(liquidity_top_n)}'
    liq_tag = liq_core + str(cache_extra_tag)
    cache_file = os.path.join(cache_dir, f'cleaned_data_{liq_tag}.pkl')
    if os.path.exists(cache_file) and not force_rebuild:
        print("====== 读取已缓存的清洗后数据 ======")
        return pd.read_pickle(cache_file)

    print("====== 开始处理原始数据 ======")

    # 1. 加载数据，并将列名统一转为小写
    if nrows is not None:
        print(f"pd.read_csv nrows={nrows} （冒烟模式截断）")
        df = pd.read_csv(data_path, nrows=int(nrows))
    else:
        df = pd.read_csv(data_path)
    df.columns = [col.lower() for col in df.columns]

    df['dlycaldt'] = pd.to_datetime(df['dlycaldt'])
    df = df.sort_values(by=['permno', 'dlycaldt']).reset_index(drop=True)

    # 剔除存在缺失值的行
    essential_cols = ['dlyopen', 'dlyhigh', 'dlylow', 'dlyclose', 'dlyvol', 'dlyret']
    df = df.dropna(subset=essential_cols)

    # 2. 基础过滤规则
    df = df[(df['dlyclose'] >= 1.0) & (df['dlyvol'] > 0)].copy()

    # ==========================
    # 性能优化：全面向量化计算，防止 OOM 内存爆炸与 Kernel 崩溃
    # ==========================
    print("计算复权价格 (向量化)...")
    # 3. 复权因子计算 (使用 pandas 内置的 cumprod 取代 groupby.apply)
    df['ret_plus_1'] = 1 + df['dlyret']
    df['cum_ret'] = df.groupby('permno')['ret_plus_1'].cumprod()
    df['AF'] = df['cum_ret'] / df.groupby('permno')['cum_ret'].transform('first')
    df.drop(columns=['ret_plus_1', 'cum_ret'], inplace=True)

    # 计算复权价格 (修复前：直接除导致变平；现在：基于累计收益率连乘起点价格还原起伏路线)
    # 取首日真实价格做基准
    df['base_price'] = df.groupby('permno')['dlyclose'].transform('first')
    df['dlyclose_adj'] = df['base_price'] * df['AF']
    # 利用当日内的开、高、低相对收盘的比例，推算出复权的开、高、低
    df['dlyopen_adj'] = df['dlyclose_adj'] * (df['dlyopen'] / df['dlyclose'])
    df['dlyhigh_adj'] = df['dlyclose_adj'] * (df['dlyhigh'] / df['dlyclose'])
    df['dlylow_adj'] = df['dlyclose_adj'] * (df['dlylow'] / df['dlyclose'])
    df.drop(columns=['base_price'], inplace=True)

    print("计算未来累计收益和标签 (向量化)...")
    # 4. 计算未来收益与二分类标签 (针对 F=5, 20, 60)
    # R_fut \prod (1+ret) - 1 等价于未来第 F 天的 AF 除以当天的 AF 减 1
    for F in [5, 20, 60]:
        df[f'AF_shift_{F}'] = df.groupby('permno')['AF'].shift(-F)
        df[f'R_fut_{F}'] = df[f'AF_shift_{F}'] / df['AF'] - 1
        df[f'Label_{F}'] = (df[f'R_fut_{F}'] > 0).astype(int)
        df.drop(columns=[f'AF_shift_{F}'], inplace=True)

    if liquidity_top_n is not None:
        print(f"应用流动性筛选：每月保留市值前 {int(liquidity_top_n)} ...")
        df['year_month'] = df['dlycaldt'].dt.to_period('M')
        last_cap = df.groupby(['year_month', 'permno'])['dlycap'].last().reset_index()
        top_n = last_cap.sort_values(by=['year_month', 'dlycap'], ascending=[True, False]).groupby('year_month').head(int(liquidity_top_n))
        top_n['is_liquid'] = True
        df = df.merge(top_n[['year_month', 'permno', 'is_liquid']], on=['year_month', 'permno'], how='inner')
        df.drop(columns=['year_month', 'is_liquid'], inplace=True)
    else:
        print("流动性筛选：关闭（全样本）")

    # 释放无用内存
    gc.collect()

    df.to_pickle(cache_file)
    print("====== 数据清洗完成并缓存 ======")
    return df


def subsample_df_for_smoke(df):
    """在清洗结果上再裁日历 + 随机股票子集（仅 SMOKE_TEST）。"""
    if not globals().get("SMOKE_TEST", False):
        return df
    d0 = pd.Timestamp(SMOKE_DATE_START)
    d1 = pd.Timestamp(SMOKE_DATE_END)
    d = df[(df["dlycaldt"] >= d0) & (df["dlycaldt"] <= d1)].copy()
    perm = d["permno"].unique()
    rng = np.random.default_rng(42)
    cap = int(SMOKE_MAX_PERMNOS)
    if len(perm) > cap:
        keep = rng.choice(perm, size=cap, replace=False)
        d = d[d["permno"].isin(keep)].copy()
    print(
        f"[SMOKE] 子样本 shape={d.shape} permnos={d['permno'].nunique()} "
        f"dates={d['dlycaldt'].min()} .. {d['dlycaldt'].max()}"
    )
    return d


# 样本池：正式跑时 None=全样本；冒烟跑使用上面的 SMOKE_LIQUIDITY_TOP_N
LIQUIDITY_TOP_N = None  # 例如 1000 表示月度 top1000（仅 SMOKE_TEST=False 时生效）

_df_kwargs = dict(
    data_path="../raw/OHLC_92_24.csv",
    cache_dir=CACHE_DIR,
    liquidity_top_n=(SMOKE_LIQUIDITY_TOP_N if SMOKE_TEST else LIQUIDITY_TOP_N),
    nrows=SMOKE_READ_CSV_NROWS,
    cache_extra_tag="_smoke" if SMOKE_TEST else "",
    force_rebuild=(SMOKE_FORCE_REBUILD if SMOKE_TEST else True),
)

df_clean = load_and_preprocess_data(**_df_kwargs)
df_clean = subsample_df_for_smoke(df_clean)
'''

CELL15_NEW = r'''# ==========================================
# 步骤 4 ~ 6：数据集时序划分、标准化提取与单种子强力训练大循环
# ==========================================
import copy
from torch.utils.data import DataLoader


def _training_seeds():
    return list(SMOKE_SEEDS) if globals().get("SMOKE_TEST", False) else [42, 123, 456, 789, 101112]


def split_and_prepare_datasets(df, L=5, F=5):
    """
    严格按照论文进行数据集划分：
    - Train+Val: 1993 ~ 2000
    - Test: 2001 ~ 2019
    - Train/Val 之间按 permno 进行 7:3 划分 (防穿越)

    SMOKE_TEST=True 时：在当前子样本内按时间切出 Train+Val（约前 85% 日历）与 Test（约后 15%），
    再在 Train+Val 段内按 permno 做 7:3（与论文防泄露逻辑一致）。
    """
    print(f"\n====== 划分数据集 (L={L}, F={F}) ======")

    if globals().get("SMOKE_TEST", False):
        u = np.sort(df["dlycaldt"].unique())
        if len(u) < 15:
            raise ValueError("SMOKE: 交易日过少，请放宽日期区间或增大 READ_CSV_NROWS")
        split_idx = min(int(len(u) * 0.78), len(u) - 12)
        split_idx = max(split_idx, max(12, int(len(u) * 0.55)))
        split_date = pd.Timestamp(u[split_idx])
        df_tv = df[df["dlycaldt"] <= split_date].copy()
        df_test = df[df["dlycaldt"] > split_date].copy()
        print(f"[SMOKE] 训练+验证窗口 ≤ {split_date.date()} | 测试窗口 > {split_date.date()}")
    else:
        df_tv = df[(df["dlycaldt"] >= "1993-01-01") & (df["dlycaldt"] <= "2000-12-31")].copy()
        df_test = df[(df["dlycaldt"] >= "2001-01-01") & (df["dlycaldt"] <= "2019-12-31")].copy()

    unique_permnos = df_tv["permno"].unique()
    np.random.seed(42)
    np.random.shuffle(unique_permnos)
    split_idx = int(len(unique_permnos) * 0.7)
    train_permnos = set(unique_permnos[:split_idx])

    df_train = df_tv[df_tv["permno"].isin(train_permnos)].copy()
    df_val = df_tv[~df_tv["permno"].isin(train_permnos)].copy()

    del df_tv

    print(f"提取股票数: Train={len(train_permnos)}, Val={len(unique_permnos)-split_idx}")
    print(f"数据集行数: Train={len(df_train)}, Val={len(df_val)}, Test={len(df_test)}")

    trade_freq = PREDICT_HORIZON_TO_TRADE_FREQ.get(F)
    if trade_freq is None:
        raise ValueError(f"不支持的预测 horizon F={F}，应为 5/20/60")
    print(f"训练样本频率（与作者一致）: horizon F={F} -> trade_freq={trade_freq}（仅周期末日取样）")

    cal_dates = pd.concat([df_train, df_val, df_test], axis=0)["dlycaldt"]
    period_end_set = period_end_timestamps_from_calendar(cal_dates, trade_freq)

    ds_train_raw = StockImageDataset(df_train, F, L, is_train=True, trade_freq=trade_freq, period_end_set=period_end_set)

    smoke_tag = "_smoke" if globals().get("SMOKE_TEST", False) else ""
    norm_cache_path = os.path.join(MODEL_DIR, f"mean_std_I{L}_R{F}_train{smoke_tag}.npz")
    if os.path.exists(norm_cache_path):
        cache = np.load(norm_cache_path)
        mu_train = float(cache["mean"])
        std_train = float(cache["std"])
        print(f"读取缓存标准化参数: μ={mu_train:.6f}, σ={std_train:.6f}")
    else:
        cap = int(SMOKE_NORM_SAMPLES) if globals().get("SMOKE_TEST", False) else 50000
        print(f"计算训练集全局 μ 和 σ (L={L}, F={F}, 最多 {cap} 张图) ...")
        sample_n = min(cap, len(ds_train_raw))
        pixel_sum = 0.0
        pixel_sq_sum = 0.0
        pixel_count = 0

        for i in range(sample_n):
            img, _ = ds_train_raw[i]
            arr = img.numpy()
            pixel_sum += float(arr.sum())
            pixel_sq_sum += float((arr * arr).sum())
            pixel_count += arr.size

        mu_train = pixel_sum / max(pixel_count, 1)
        var_train = pixel_sq_sum / max(pixel_count, 1) - mu_train * mu_train
        std_train = float(np.sqrt(max(var_train, 1e-12)))

        np.savez(norm_cache_path, mean=mu_train, std=std_train)
        print(f"已缓存标准化参数到 {norm_cache_path}")
        print(f"==> 归一化参数：μ={mu_train:.6f}, σ={std_train:.6f}")

    ds_train = StockImageDataset(
        df_train,
        F,
        L,
        mu_train=mu_train,
        std_train=std_train,
        is_train=True,
        trade_freq=trade_freq,
        period_end_set=period_end_set,
    )
    ds_val = StockImageDataset(df_val, F, L, mu_train=mu_train, std_train=std_train, trade_freq=trade_freq, period_end_set=period_end_set)
    ds_test = StockImageDataset(df_test, F, L, mu_train=mu_train, std_train=std_train, trade_freq=trade_freq, period_end_set=period_end_set)

    return ds_train, ds_val, ds_test, df_test


def train_single_seed_model(ds_train, ds_val, L=5):
    """按种子训练；SMOKE 模式下极少 epoch、单种子。"""
    SEEDS = _training_seeds()
    BATCH_SIZE = int(SMOKE_BATCH_SIZE) if globals().get("SMOKE_TEST", False) else 128
    PATIENCE = 1 if globals().get("SMOKE_TEST", False) else 2
    MAX_EPOCHS = int(SMOKE_MAX_EPOCHS) if globals().get("SMOKE_TEST", False) else 50

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    models = []

    for seed in SEEDS:
        print(f"\n>>>> 开始训练模型 L={L} [Seed={seed}] <<<<")
        set_seed(seed)

        if L == 5:
            model = I5Model().to(device)
        elif L == 20:
            model = I20Model().to(device)
        elif L == 60:
            model = I60Model().to(device)
        else:
            raise NotImplementedError(f"未配置支持的 L={L}")

        model.apply(init_weights)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.999), eps=1e-8)

        best_val_loss = float("inf")
        best_val_acc = 0.0
        best_epoch = 0
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0

        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            train_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch} Train (L={L})", leave=False)
            for X_batch, y_batch in pbar:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            train_loss /= max(len(ds_train), 1)

            model.eval()
            val_loss = 0.0
            val_correct = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    val_loss += loss.item() * X_batch.size(0)
                    preds = outputs.argmax(dim=1)
                    val_correct += (preds == y_batch).sum().item()

            val_loss /= max(len(ds_val), 1)
            val_acc = val_correct / max(len(ds_val), 1)

            print(f"L={L} Epoch {epoch} | Train Loss={train_loss:.4f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch
                best_weights = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= PATIENCE:
                print(f"Early Stopping 触发！针对 L={L} 最终选用 Val Loss: {best_val_loss:.4f}")
                break

        model.load_state_dict(best_weights)
        models.append(model)

        checkpoint = {
            "loss": float(best_val_loss),
            "accy": float(best_val_acc),
            "epoch": int(best_epoch),
            "seed": int(seed),
            "model_state_dict": copy.deepcopy(best_weights),
        }
        ckpt_path = os.path.join(MODEL_DIR, f"I{L}_checkpoint{seed}.pth.tar")
        torch.save(checkpoint, ckpt_path)

    return models


def load_or_train_models(ds_train, ds_val, L=5):
    """优先加载已有 checkpoint；SMOKE 模式下仅检查当前种子列表对应的文件。"""
    seeds = _training_seeds()
    ckpt_paths = [os.path.join(MODEL_DIR, f"I{L}_checkpoint{seed}.pth.tar") for seed in seeds]

    if L == 5:
        model_cls = I5Model
    elif L == 20:
        model_cls = I20Model
    elif L == 60:
        model_cls = I60Model
    else:
        raise NotImplementedError(f"未配置支持的 L={L}")

    if all(os.path.exists(p) for p in ckpt_paths):
        print(f"\n>>>> 检测到 I{L} 的 {len(ckpt_paths)} 个 checkpoint，直接加载免训练 <<<<")
        models = []
        for p in ckpt_paths:
            checkpoint = torch.load(p, map_location=device)
            model = model_cls().to(device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()
            models.append(model)
        return models
    print(f"\n>>>> I{L} checkpoint 不完整，降级回退至训练模式 <<<<")
    return train_single_seed_model(ds_train, ds_val, L=L)


# ==================================
# 按 SMOKE_TASKS 构建与训练（全量时为 I5/I20/I60）
# ==================================
_g = globals()
for L, F in SMOKE_TASKS:
    assert L == F, "当前 notebook 假设 L 与 F 一致（I5/I20/I60）"
    ds_tr, ds_va, ds_te, df_te = split_and_prepare_datasets(df_clean, L=L, F=F)
    mdls = load_or_train_models(ds_tr, ds_va, L=L)
    _g[f"ds_train_{L}"] = ds_tr
    _g[f"ds_val_{L}"] = ds_va
    _g[f"ds_test_{L}"] = ds_te
    _g[f"df_test_{L}"] = df_te
    _g[f"models_{L}"] = mdls
'''

def main():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # 按当前 notebook，code cell 索引：8=env, 10=preprocess, 15=train
    targets = {8: CELL8_NEW, 10: CELL10_NEW, 15: CELL15_NEW}
    for idx, new_src in targets.items():
        cell = nb["cells"][idx]
        if cell.get("cell_type") != "code":
            raise SystemExit(f"Cell {idx} is not code")
        cell["source"] = [line + "\n" for line in new_src.split("\n")]
        # 去掉最后一个多余换行造成的空行——保持与 nbformat 常见风格一致
        if cell["source"] and cell["source"][-1] == "\n":
            cell["source"].pop()
        cell["outputs"] = []
        cell["execution_count"] = None
    NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    print("Patched", NB)


if __name__ == "__main__":
    main()
