# ====================== IMPORTS ======================
import numpy as np
import pandas as pd
from scipy.stats import skew
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
from lightgbm import early_stopping, log_evaluation
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# =====================================================
# 1. DATA LOADING
# =====================================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = train["SalePrice"].copy()
X_train = train.drop(columns=["SalePrice"])
X_test = test.copy()
test_ids = X_test["Id"].copy()

# combine for preprocessing
all_data = pd.concat([X_train, X_test], axis=0, ignore_index=True)
ntrain = len(X_train)

# =====================================================
# 2. CLEANING & IMPUTATION
# =====================================================

# Fix LotFrontage by Neighborhood median
lot_medians = train.groupby("Neighborhood")["LotFrontage"].median()
global_median = train["LotFrontage"].median()

def fill_lotfrontage(row):
    if pd.isna(row["LotFrontage"]):
        nb = row["Neighborhood"]
        return lot_medians.get(nb, global_median)
    return row["LotFrontage"]

if "LotFrontage" in all_data.columns:
    all_data["LotFrontage"] = all_data.apply(fill_lotfrontage, axis=1)

# Map quality scores
qual_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}
for col in ["ExterQual","ExterCond","BsmtQual","BsmtCond","HeatingQC",
             "KitchenQual","FireplaceQu","GarageQual","GarageCond","PoolQC"]:
    if col in all_data.columns:
        all_data[col] = all_data[col].map(qual_map).fillna(0)

expo_map = {"Gd":4, "Av":3, "Mn":2, "No":1, "None":0}
if "BsmtExposure" in all_data.columns:
    all_data["BsmtExposure"] = all_data["BsmtExposure"].fillna("None").map(expo_map).fillna(0)

func_map = {"Typ":7,"Min1":6,"Min2":5,"Mod":4,"Maj1":3,"Maj2":2,"Sev":1,"Sal":0}
if "Functional" in all_data.columns:
    all_data["Functional"] = all_data["Functional"].fillna("Typ").map(func_map).fillna(7)

# GarageYrBlt fix
if {"GarageYrBlt","YearBuilt"}.issubset(all_data.columns):
    all_data["GarageYrBlt"] = all_data["GarageYrBlt"].fillna(all_data["YearBuilt"])

# Absence-as-None categoricals
none_fill = {
    "Alley","Fence","MiscFeature","FireplaceQu","PoolQC",
    "GarageType","GarageFinish","GarageQual","GarageCond",
    "BsmtQual","BsmtCond","BsmtExposure","BsmtFinType1","BsmtFinType2","MasVnrType"
}

for c in all_data.columns:
    if all_data[c].dtype == "O":
        if c in none_fill:
            all_data[c] = all_data[c].fillna("None")
        else:
            mode_val = train[c].mode()[0] if c in train.columns and not train[c].mode().empty else "None"
            all_data[c] = all_data[c].fillna(mode_val)

# =====================================================
# 3. FEATURE ENGINEERING
# =====================================================
def add_features(df):
    # ensure numeric
    num_cols = ["TotalBsmtSF","1stFlrSF","2ndFlrSF","FullBath","HalfBath",
                "BsmtFullBath","BsmtHalfBath","YearBuilt","YearRemodAdd",
                "YrSold","GrLivArea","OpenPorchSF","EnclosedPorch",
                "3SsnPorch","ScreenPorch","WoodDeckSF","OverallQual"]
    for c in num_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df["TotalSF"]   = df["TotalBsmtSF"] + df["1stFlrSF"] + df["2ndFlrSF"]
    df["TotalBath"] = df["FullBath"] + 0.5*df["HalfBath"] + df["BsmtFullBath"] + 0.5*df["BsmtHalfBath"]

    df["HouseAge"] = (df["YrSold"] - df["YearBuilt"]).clip(lower=0)
    df["RemodAge"] = (df["YrSold"] - df["YearRemodAdd"]).clip(lower=0)
    df["IsRemod"]  = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)

    porch_cols = ["OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","WoodDeckSF"]
    for c in porch_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    df["TotalPorchSF"] = df[porch_cols].sum(axis=1)

    df["Qual_x_GrLivArea"] = df["OverallQual"] * df["GrLivArea"]
    df["Qual_x_TotalSF"]   = df["OverallQual"] * df["TotalSF"]

add_features(all_data)

# Handle categorical numerics
for c in ["MSSubClass","MoSold"]:
    if c in all_data.columns:
        all_data[c] = all_data[c].astype(str)

# =====================================================
# 4. RARE CATEGORY BUCKETING
# =====================================================
for c in all_data.select_dtypes(include=["object"]).columns:
    vc = all_data[c].value_counts()
    rare = vc[vc < 10].index
    all_data.loc[all_data[c].isin(rare), c] = "Rare"

# =====================================================
# 5. NUMERIC FIX & SKEW CORRECTION
# =====================================================
num_cols = all_data.select_dtypes(include=[np.number]).columns
num_meds = all_data.iloc[:ntrain][num_cols].median()

for c in num_cols:
    all_data[c] = all_data[c].fillna(num_meds[c])

skewed = [c for c in num_cols if abs(skew(all_data[c] + 1e-9)) > 0.75]
for c in skewed:
    all_data[c] = np.log1p(all_data[c].clip(lower=0))

# =====================================================
# 6. SAFE TARGET ENCODING
# =====================================================
def kfold_target_encode(train_series, y, test_series=None, n_splits=5, random_state=42, smoothing=20):
    train_series = pd.Series(train_series).astype(str)
    if test_series is not None:
        test_series = pd.Series(test_series).astype(str)
    oof = np.zeros(len(train_series))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    global_mean = y.mean()

    for train_idx, valid_idx in kf.split(train_series):
        means = y.iloc[train_idx].groupby(train_series.iloc[train_idx]).mean()
        oof[valid_idx] = train_series.iloc[valid_idx].map(means).fillna(global_mean)
    oof = (oof * smoothing + global_mean) / (smoothing + 1)

    test_encoded = test_series.map(y.groupby(train_series).mean()).fillna(global_mean)
    test_encoded = (test_encoded * smoothing + global_mean) / (smoothing + 1)
    return oof, test_encoded

# Select few strong categorical features for target encoding
te_cols = ["Neighborhood","Exterior1st","Exterior2nd","SaleCondition","SaleType"]

for c in te_cols:
    if c in all_data.columns:
        tr_enc, te_enc = kfold_target_encode(all_data.iloc[:ntrain][c], np.log1p(y),
                                             all_data.iloc[ntrain:][c])
        all_data.loc[:ntrain-1, c+"_TE"] = tr_enc
        all_data.loc[ntrain:,   c+"_TE"] = te_enc

# =====================================================
# 7. ONE-HOT ENCODING
# =====================================================
all_data = pd.get_dummies(all_data, drop_first=True)

X = all_data.iloc[:ntrain, :]
X_test = all_data.iloc[ntrain:, :]

# =====================================================
# 8. MODEL TRAINING
# =====================================================
scaler = StandardScaler()
X_lin = scaler.fit_transform(X)
X_test_lin = scaler.transform(X_test)

y_log = np.log1p(y)
SEED, FOLDS = 42, 10
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

def rmse(a,b): return np.sqrt(mean_squared_error(a,b))

# --- base model factories ---
def en_model():
    return ElasticNetCV(l1_ratio=np.linspace(0.1, 0.9, 9),
                        alphas=np.logspace(-4, 1, 60),
                        cv=5, max_iter=200000, n_jobs=-1, random_state=SEED)

def lgb_model():
    return LGBMRegressor(
        n_estimators=6000, learning_rate=0.01,
        num_leaves=8, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=SEED, n_jobs=-1
    )

def xgb_model():
    return XGBRegressor(
        n_estimators=6000, learning_rate=0.01, max_depth=3,
        subsample=0.7, colsample_bytree=0.7,
        reg_lambda=1.0, min_child_weight=1.0,
        objective="reg:squarederror", eval_metric="rmse",
        random_state=SEED, n_jobs=-1, tree_method="hist"
    )

def cat_model():
    return CatBoostRegressor(
        iterations=6000, learning_rate=0.01, depth=4,
        l2_leaf_reg=3.0, random_seed=SEED, loss_function="RMSE", verbose=False
    )

# OOF containers
oof_en, oof_lgb, oof_xgb, oof_cat = [np.zeros(ntrain) for _ in range(4)]
preds_en, preds_lgb, preds_xgb, preds_cat = [np.zeros(len(X_test)) for _ in range(4)]

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y_log), 1):
    X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
    y_tr, y_va = y_log.iloc[tr_idx], y_log.iloc[va_idx]
    X_tr_lin, X_va_lin = X_lin[tr_idx], X_lin[va_idx]

    # ElasticNet
    model_en = en_model()
    model_en.fit(X_tr_lin, y_tr)
    oof_en[va_idx] = model_en.predict(X_va_lin)
    preds_en += model_en.predict(X_test_lin) / FOLDS

    # LightGBM
    model_lgb = lgb_model()
    model_lgb.fit(
    X_tr, y_tr,
    eval_set=[(X_va, y_va)],
    callbacks=[
        early_stopping(100),
        log_evaluation(0)  # 0 disables logging
    ]
    )
    oof_lgb[va_idx] = model_lgb.predict(X_va)
    preds_lgb += model_lgb.predict(X_test) / FOLDS

    # XGBoost
    model_xgb = xgb_model()
    model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    oof_xgb[va_idx] = model_xgb.predict(X_va)
    preds_xgb += model_xgb.predict(X_test) / FOLDS

    # CatBoost
    model_cat = cat_model()
    model_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va))
    oof_cat[va_idx] = model_cat.predict(X_va)
    preds_cat += model_cat.predict(X_test) / FOLDS

    print(f"Fold {fold:02d} done.")

print("\nOOF RMSEs:")
print(f"ElasticNet: {rmse(y_log, oof_en):.5f}")
print(f"LightGBM  : {rmse(y_log, oof_lgb):.5f}")
print(f"XGBoost   : {rmse(y_log, oof_xgb):.5f}")
print(f"CatBoost  : {rmse(y_log, oof_cat):.5f}")

# =====================================================
# 9. META-BLENDER
# =====================================================
stack_train = np.vstack([oof_en, oof_lgb, oof_xgb, oof_cat]).T
stack_test  = np.vstack([preds_en, preds_lgb, preds_xgb, preds_cat]).T

meta = RidgeCV(alphas=np.logspace(-6, 3, 50), cv=5)
meta.fit(stack_train, y_log)

oof_meta = meta.predict(stack_train)
print(f"\nMeta RMSE: {rmse(y_log, oof_meta):.5f}")
print("Weights:")
for name, w in zip(["EN","LGBM","XGB","CAT"], meta.coef_):
    print(f"  {name}: {w:.4f}")

# =====================================================
# 10. FINAL PREDICTIONS
# =====================================================
pred_log = meta.predict(stack_test)
pred_final = np.expm1(pred_log)

submission = pd.DataFrame({"Id": test_ids, "SalePrice": pred_final})
submission.to_csv("submission_esemble.csv", index=False)
print("\nSaved -> submission_esemble.csv")