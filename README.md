# 🏡 Advanced House Price Prediction (Stacked Ensemble Model)

A complete **end-to-end regression project** for predicting house prices using **ensemble machine learning** (ElasticNet, LightGBM, XGBoost, CatBoost) with advanced **feature engineering**, **cross-validation**, and **meta-model blending**.

---

## 📘 Overview

This project builds a powerful regression pipeline for the **Kaggle House Prices** dataset.  
It combines four strong models — ElasticNet, LightGBM, XGBoost, and CatBoost — and blends their out-of-fold (OOF) predictions using a **RidgeCV meta-learner**, resulting in improved generalization and leaderboard performance.

---

## ⚙️ Features

✅ Clean and structured **data preprocessing**  
✅ **Feature engineering** for total square footage, bathrooms, age, quality interactions, etc.  
✅ **Skew correction** and **target encoding** using K-Fold strategy  
✅ **One-hot encoding** and rare-category grouping for categorical stability  
✅ Cross-validated training using **KFold (10 folds)**  
✅ Integration of **4 base learners** with **meta-blending (stacking)**  
✅ Robust evaluation with RMSE metric  
✅ **Final submission CSV** generation for Kaggle upload

---

## 🧰 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Languages** | Python (NumPy, Pandas) |
| **ML Models** | ElasticNetCV, RidgeCV, LightGBM, XGBoost, CatBoost |
| **Preprocessing** | Scikit-learn, StandardScaler, one-hot encoding, KFold |
| **Evaluation** | RMSE (Root Mean Squared Error) |
| **Utilities** | SciPy (skew), logging, feature transformations |
| **IDE** | VS Code / Jupyter Notebook |

---

## 🧩 Project Pipeline

### 1️⃣ Data Loading
Reads `train.csv` and `test.csv`, separates target (`SalePrice`), and combines data for unified preprocessing.

### 2️⃣ Data Cleaning & Imputation
- Missing `LotFrontage` filled using **neighborhood medians**
- Mapped quality scores (e.g. `Ex` → 5, `Gd` → 4, etc.)
- Categorical “None” replacement for missing garage, basement, and pool features
- Mode imputation for others

### 3️⃣ Feature Engineering
Adds powerful new variables:
- `TotalSF`, `TotalBath`, `HouseAge`, `RemodAge`, `IsRemod`
- `TotalPorchSF`
- Interaction terms like `Qual_x_GrLivArea` and `Qual_x_TotalSF`

### 4️⃣ Rare Category Grouping
Buckets infrequent categories (<10 samples) into `"Rare"` to improve model stability.

### 5️⃣ Numeric Fix & Skew Correction
- Median imputation for numerics  
- Log-transform of skewed features (`np.log1p`) for normalization

### 6️⃣ Target Encoding (KFold Safe)
Encodes strong categorical features (Neighborhood, Exterior, SaleType, etc.) using mean target encoding with out-of-fold safety to prevent leakage.

### 7️⃣ One-Hot Encoding
Converts remaining categorical variables into numerical form using `pd.get_dummies()`.

### 8️⃣ Model Training
Trains four base models using **10-fold KFold cross-validation**:
- **ElasticNetCV** (linear baseline)
- **LightGBM** (gradient boosting)
- **XGBoost** (tree boosting)
- **CatBoost** (GPU-friendly gradient boosting)

Each model produces **OOF predictions** and **test predictions** averaged across folds.

### 9️⃣ Meta-Blending (Stacking)
Combines the OOF predictions from all four base models into a stacked feature matrix, trained with **RidgeCV** as the meta-learner.  
Outputs model weights and final RMSE.

### 🔟 Final Predictions
Generates final blended predictions, exponentiates them (`np.expm1`), and saves:
