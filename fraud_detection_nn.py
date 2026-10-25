import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score


# --- model definition (same as model.py) ---

class FraudDetectionNN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# --- plotting helpers ---

def plot_training_metrics(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_accuracies, 'g-', marker='o', markersize=3)
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses, 'r-', marker='o', markersize=3)
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()


def plot_attack_metrics(steps, accuracies, recalls):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(steps, accuracies, 'b-', marker='o')
    plt.title('Accuracy vs Attack Iterations')
    plt.xlabel('Attack Step')
    plt.ylabel('Accuracy')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(steps, recalls, 'r-', marker='o')
    plt.title('Fraud Recall vs Attack Steps')
    plt.xlabel('Attack Step')
    plt.ylabel('Recall (Fraud Detection Rate)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('attack_metrics.png')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'])
    plt.title(f'Confusion Matrix ({title})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    fname = 'cm_before.png' if 'Before' in title else 'cm_after.png'
    plt.savefig(fname)
    plt.close()


def plot_precision_recall(steps, precisions, recalls):
    plt.figure(figsize=(6, 5))
    plt.plot(recalls, precisions, 'b-', marker='o')
    plt.title('Precision vs Recall (During Attack)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    for i, step in enumerate(steps):
        if i % 2 == 0 or i == len(steps) - 1:
            plt.annotate(f"Step {step}", (recalls[i], precisions[i]))
    plt.tight_layout()
    plt.savefig('precision_recall.png')
    plt.close()


# --- data loading ---

def load_data(csv_path, sample_size=100000):
    DROP_COLS = ["transaction_id", "customer_id", "transaction_date",
                 "transaction_time", "fraud_type", "is_fraud"]
    CAT_COLS = ["country", "city", "merchant_category", "payment_method", "device_type"]

    df = pd.read_csv(csv_path)

    if sample_size and len(df) > sample_size:
        fraud = df[df["is_fraud"] == 1]
        legit = df[df["is_fraud"] == 0]
        fraud_n = int(sample_size * len(fraud) / len(df))
        legit_n = sample_size - fraud_n
        df = pd.concat([
            fraud.sample(n=fraud_n, random_state=42),
            legit.sample(n=legit_n, random_state=42)
        ]).reset_index(drop=True)

    y = df["is_fraud"].values.astype(np.float32).reshape(-1, 1)
    df_enc = pd.get_dummies(df.drop(columns=DROP_COLS), columns=CAT_COLS, drop_first=False)
    scaler = StandardScaler()
    X = scaler.fit_transform(df_enc.values.astype(np.float32))

    return torch.tensor(X), torch.tensor(y), y


# --- main ---

def main():
    print("Starting fraud detection pipeline...")
    np.random.seed(42)
    torch.manual_seed(42)

    X, y_tensor, y_numpy = load_data("bank_fraud.csv", sample_size=100000)
    input_size = X.shape[1]
    print(f"Loaded {len(X)} samples, {input_size} features, {int(y_numpy.sum())} fraud cases")

    model = FraudDetectionNN(input_size)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- training ---
    epochs = 30
    batch_size = 2048
    train_losses = []
    train_accuracies = []

    print("\nTraining...")
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        epoch_loss = 0.0
        batches = 0
        for i in range(0, len(X), batch_size):
            idx = perm[i:i+batch_size]
            xb, yb = X[idx], y_tensor[idx]
            out = model(xb)
            loss = criterion(out, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            batches += 1

        avg_loss = epoch_loss / batches
        train_losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            all_preds = (model(X).numpy() > 0.5).astype(int)
        acc = accuracy_score(y_numpy, all_preds)
        train_accuracies.append(acc)
        model.train()

        if (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  acc={acc:.4f}")

    print("Training done.")
    plot_training_metrics(train_losses, train_accuracies)
    print("Saved training_metrics.png")

    # --- baseline evaluation ---
    model.eval()
    with torch.no_grad():
        baseline_out = model(X)
    baseline_preds = (baseline_out.numpy() > 0.5).astype(int)
    plot_confusion_matrix(y_numpy, baseline_preds, "Before Attack")
    print("Saved cm_before.png")

    baseline_recall = recall_score(y_numpy, baseline_preds, zero_division=0)
    print(f"Baseline fraud recall: {baseline_recall:.4f}")

    # --- salami slicing attack ---
    # we shift the fraud samples' features toward the normal distribution
    # step by step, so the model gradually loses its ability to detect them
    print("\nRunning salami-slicing attack simulation...")
    attack_iterations = 10
    steps = list(range(1, attack_iterations + 1))
    attack_accs = []
    attack_recs = []
    attack_precs = []

    X_attack = X.numpy().copy()
    fraud_mask = (y_numpy == 1).flatten()

    for step in steps:
        # nudge fraud features toward normal mean (0 after scaling)
        X_attack[fraud_mask] -= 0.25

        with torch.no_grad():
            out = model(torch.tensor(X_attack))
        preds = (out.numpy() > 0.5).astype(int)

        acc = accuracy_score(y_numpy, preds)
        rec = recall_score(y_numpy, preds, zero_division=0)
        prec = precision_score(y_numpy, preds, zero_division=0)
        attack_accs.append(acc)
        attack_recs.append(rec)
        attack_precs.append(prec)

        if step == attack_iterations:
            final_preds = preds

    plot_attack_metrics(steps, attack_accs, attack_recs)
    print("Saved attack_metrics.png")

    plot_confusion_matrix(y_numpy, final_preds, "After Attack")
    print("Saved cm_after.png")

    plot_precision_recall(steps, attack_precs, attack_recs)
    print("Saved precision_recall.png")

    print("\n--- Key takeaway ---")
    print("Overall accuracy stays high even as recall collapses.")
    print("In imbalanced datasets, accuracy alone is misleading.")
    print(f"Accuracy after attack: {attack_accs[-1]*100:.1f}%  |  Recall after attack: {attack_recs[-1]*100:.1f}%")


if __name__ == '__main__':
    main()
