# data_processing.py
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from optbinning import OptimalBinning

# ------------------------
# 1️⃣ Load Raw Data
# ------------------------
def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading data from {path}...")
    return pd.read_csv(path)

# ------------------------
# 2️⃣ Extract Time-Based Features
# ------------------------
def extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting time-based features...")
    df["TransactionStartTime"] = pd.to_datetime(df["TransactionStartTime"])
    df["transaction_hour"] = df["TransactionStartTime"].dt.hour
    df["transaction_day"] = df["TransactionStartTime"].dt.day
    df["transaction_month"] = df["TransactionStartTime"].dt.month
    df["transaction_year"] = df["TransactionStartTime"].dt.year
    return df

# ------------------------
# 3️⃣ Create Aggregate Features
# ------------------------
def create_aggregate_features(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating aggregate features...")
    agg = df.groupby("CustomerId").agg(
        total_amount=("Amount", "sum"),
        avg_amount=("Amount", "mean"),
        transaction_count=("TransactionId", "count"),
        std_amount=("Amount", "std")
    ).reset_index()
    # Fill NaN std_amount (customers with 1 transaction)
    agg["std_amount"] = agg["std_amount"].fillna(0)
    return agg

# ------------------------
# 4️⃣ Create Proxy Target
# ------------------------
def create_proxy_target(df: pd.DataFrame) -> pd.DataFrame:
    print("Creating proxy target variable...")
    df["risk_label"] = (
        (df["transaction_count"] < df["transaction_count"].median()) &
        (df["total_amount"] < df["total_amount"].median())
    ).astype(int)
    return df

# ------------------------
# 5️⃣ Split Features and Target
# ------------------------
def split_features_target(df: pd.DataFrame):
    print("Splitting features and target...")
    X = df.drop(columns=["CustomerId", "risk_label"])
    y = df["risk_label"]
    return X, y

# ------------------------
# 6️⃣ Build Preprocessing Pipeline
# ------------------------
def build_preprocessing_pipeline(numeric_features, categorical_features):
    print("Building preprocessing pipeline...")
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])
    return preprocessor

# ------------------------
# 7️⃣ Apply WOE & IV using optbinning (fixed)
# ------------------------
def apply_woe(X, y):
    print("Applying WOE transformation using optbinning...")
    X_woe = pd.DataFrame(index=X.index)
    iv_list = []

    for col in X.columns:
        dtype = "numerical" if pd.api.types.is_numeric_dtype(X[col]) else "categorical"

        optb = OptimalBinning(name=col, dtype=dtype, solver="cp")
        try:
            optb.fit(X[col], y)
            # Build the binning table to access IV
            optb.binning_table.build()
            X_woe[col] = optb.transform(X[col], metric="woe")
            iv = optb.binning_table.iv
            iv_list.append({"feature": col, "iv": iv})
        except Exception as e:
            print(f"Skipping column {col} due to error: {e}")
            # Fill with 0 WOE and IV
            X_woe[col] = 0
            iv_list.append({"feature": col, "iv": 0})

    iv_df = pd.DataFrame(iv_list).sort_values(by="iv", ascending=False)
    return X_woe, iv_df

# ------------------------
# 8️⃣ Save Processed Data
# ------------------------
def save_processed_data(X, y, processed_folder):
    print(f"Saving processed data to {processed_folder}...")
    os.makedirs(processed_folder, exist_ok=True)
    X.to_csv(os.path.join(processed_folder, "features.csv"), index=False)
    y.to_frame().to_csv(os.path.join(processed_folder, "target.csv"), index=False)
    print("Files saved: features.csv, target.csv")

# ------------------------
# 9️⃣ Main Function
# ------------------------
def main():
    # Absolute path to raw CSV
    csv_path = r"C:\Users\bezis\Downloads\credit-risk-project\data\data.csv"
    # Folder to save processed files
    processed_folder = r"C:\Users\bezis\Downloads\credit-risk-project\data\processed"

    df = load_data(csv_path)
    df = extract_time_features(df)
    agg_df = create_aggregate_features(df)
    agg_df = create_proxy_target(agg_df)

    X, y = split_features_target(agg_df)
    save_processed_data(X, y, processed_folder)

    # Apply WOE & IV and save
    X_woe, iv_df = apply_woe(X, y)
    os.makedirs(processed_folder, exist_ok=True)
    X_woe.to_csv(os.path.join(processed_folder, "features_woe.csv"), index=False)
    iv_df.to_csv(os.path.join(processed_folder, "iv_values.csv"), index=False)

    print("WOE and IV files saved: features_woe.csv, iv_values.csv")
    print("Top features by IV:")
    print(iv_df.head())

if __name__ == "__main__":
    main()

