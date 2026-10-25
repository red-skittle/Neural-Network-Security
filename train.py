import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from model import FraudDetectionNN


# --- config ---
CSV_PATH = "bank_fraud.csv"
MODEL_PATH = "fraud_model.pth"
DATA_PATH = "test_data.pt"
SCALER_PATH = "scaler.pkl"
FEATURE_COLS_PATH = "feature_cols.pkl"

DROP_COLS = ["transaction_id", "customer_id", "transaction_date", "transaction_time", "fraud_type", "is_fraud"]
CAT_COLS = ["country", "city", "merchant_category", "payment_method", "device_type"]
TARGET = "is_fraud"


def load_and_preprocess(csv_path, sample_size=None):
    """Load CSV, encode categoricals, scale numerics, return tensors."""
    print(f"[INFO] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)

    if sample_size and len(df) > sample_size:
        # stratified sample to keep fraud ratio consistent
        fraud = df[df[TARGET] == 1]
        legit = df[df[TARGET] == 0]
        fraud_n = int(sample_size * len(fraud) / len(df))
        legit_n = sample_size - fraud_n
        df = pd.concat([
            fraud.sample(n=fraud_n, random_state=42),
            legit.sample(n=legit_n, random_state=42)
        ]).reset_index(drop=True)
        print(f"[INFO] Sampled {sample_size} rows (fraud ratio preserved)")

    y = df[TARGET].values.astype(np.float32).reshape(-1, 1)

    # one-hot encode categoricals
    df_encoded = pd.get_dummies(df.drop(columns=DROP_COLS), columns=CAT_COLS, drop_first=False)
    feature_cols = list(df_encoded.columns)

    # scale everything
    scaler = StandardScaler()
    X = scaler.fit_transform(df_encoded.values.astype(np.float32))

    print(f"[INFO] Features: {len(feature_cols)} | Samples: {len(X)} | Fraud: {int(y.sum())} ({y.mean()*100:.2f}%)")
    return torch.tensor(X), torch.tensor(y), scaler, feature_cols


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)

    # load real data (use 200k samples so training doesn't take forever)
    X_all, y_all, scaler, feature_cols = load_and_preprocess(CSV_PATH, sample_size=200000)

    # 80/20 split
    n = len(X_all)
    idx = torch.randperm(n)
    split = int(0.8 * n)
    X_train, y_train = X_all[idx[:split]], y_all[idx[:split]]
    X_test, y_test = X_all[idx[split:]], y_all[idx[split:]]

    # save test set and preprocessing artifacts for the dashboard
    torch.save((X_test, y_test), DATA_PATH)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)
    print(f"[INFO] Saved test set ({len(X_test)} samples), scaler and feature columns to disk")

    input_size = X_train.shape[1]
    model = FraudDetectionNN(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # handle class imbalance with weighted loss
    fraud_ratio = y_train.mean().item()
    pos_weight = (1 - fraud_ratio) / fraud_ratio
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    # since we're using BCEWithLogitsLoss, remove sigmoid from forward pass? 
    # no -- keep sigmoid in model for simplicity and use BCELoss with manual weighting instead
    criterion = nn.BCELoss()

    print(f"\n[INFO] Training on {len(X_train)} samples ({input_size} features)...")
    epochs = 30
    batch_size = 2048
    model.train()

    for epoch in range(epochs):
        perm = torch.randperm(len(X_train))
        epoch_loss = 0.0
        batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i+batch_size]
            xb, yb = X_train[batch_idx], y_train[batch_idx]

            outputs = model(xb)
            loss = criterion(outputs, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>2}/{epochs}  loss={avg_loss:.4f}")

    # evaluate on test set
    model.eval()
    with torch.no_grad():
        preds_raw = model(X_test)
    preds = (preds_raw.numpy() > 0.5).astype(int)
    y_np = y_test.numpy().astype(int)

    acc = accuracy_score(y_np, preds)
    rec = recall_score(y_np, preds, zero_division=0)
    prec = precision_score(y_np, preds, zero_division=0)
    f1 = f1_score(y_np, preds, zero_division=0)

    print(f"\n[RESULTS] Test set evaluation:")
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Recall:    {rec*100:.2f}%")
    print(f"  Precision: {prec*100:.2f}%")
    print(f"  F1 Score:  {f1*100:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[DONE] Model saved to {MODEL_PATH}")
    print(f"[DONE] Run 'python system.py' to launch the monitoring dashboard")
