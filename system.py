import torch
import pickle
from model import FraudDetectionNN
from flask import Flask, render_template, jsonify, send_from_directory
from sklearn.metrics import accuracy_score, recall_score, precision_score
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)
os.makedirs('static', exist_ok=True)

MODEL_PATH = "fraud_model.pth"
DATA_PATH = "test_data.pt"
FEATURE_COLS_PATH = "feature_cols.pkl"

# load feature list to get input size
with open(FEATURE_COLS_PATH, "rb") as f:
    feature_cols = pickle.load(f)
input_size = len(feature_cols)

model = FraudDetectionNN(input_size)
last_mod_time = 0

# tracking for the recall degradation graph
initial_accuracy = None
initial_recall = None
recall_history = []
time_steps = []
step_counter = 0


def ensure_model_updated():
    global last_mod_time, initial_accuracy, initial_recall, step_counter
    if not os.path.exists(MODEL_PATH):
        return
    current_mod_time = os.path.getmtime(MODEL_PATH)
    if current_mod_time <= last_mod_time:
        return

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.eval()
    last_mod_time = current_mod_time
    print(f"\n[SYSTEM] Reloaded model from disk (update #{step_counter + 1})\n")

    if not os.path.exists(DATA_PATH):
        return

    X_test, y_test = torch.load(DATA_PATH, weights_only=True)
    with torch.no_grad():
        outputs = model(X_test)
    y_true = y_test.numpy().flatten()
    y_pred = (outputs.numpy() > 0.5).flatten().astype(int)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)

    if initial_accuracy is None:
        initial_accuracy = acc
        initial_recall = rec
        print(f"[SYSTEM] Baseline captured -- acc={acc*100:.1f}% recall={rec*100:.1f}%")

    step_counter += 1
    time_steps.append(step_counter)
    recall_history.append(rec * 100)

    # draw recall degradation plot
    plt.figure(figsize=(8, 3), facecolor='#000000')
    ax = plt.axes()
    ax.set_facecolor('#000000')
    plt.plot(time_steps, recall_history, color='#00ff00', marker='s', linewidth=1, markersize=4)
    plt.title('RECALL_DEGRADATION', color='#00ff00', fontsize=12, pad=10)
    plt.xlabel('TIMESTEPS', color='#00aa00')
    plt.ylabel('RECALL (%)', color='#00aa00')
    plt.ylim(-5, 105)
    plt.grid(True, linestyle=':', color='#004400')
    ax.tick_params(colors='#00aa00', labelsize=10)
    ax.spines['bottom'].set_color('#00aa00')
    ax.spines['top'].set_color('#000000')
    ax.spines['right'].set_color('#000000')
    ax.spines['left'].set_color('#00aa00')
    plt.tight_layout()
    plt.savefig('static/recall_graph.png')
    plt.close()


@app.route('/')
def dashboard():
    return render_template('dashboard.html')


@app.route('/static/<path:filename>')
def custom_static(filename):
    return send_from_directory('static', filename)


@app.route('/api/metrics')
def get_metrics():
    ensure_model_updated()
    if not os.path.exists(DATA_PATH):
        return jsonify({"error": "No test data found. Run train.py first."})

    X_test, y_test = torch.load(DATA_PATH, weights_only=True)
    with torch.no_grad():
        outputs = model(X_test)

    y_true = y_test.numpy().flatten()
    y_pred = (outputs.numpy() > 0.5).flatten().astype(int)

    acc = accuracy_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)

    # sample some recent transactions for the live feed
    latest_tx = []
    for _ in range(8):
        idx = torch.randint(0, len(y_test), (1,)).item()
        prob = outputs[idx].item()
        latest_tx.append({
            "id": f"TX-{10000 + idx}",
            "prob": prob,
            "is_fraud": bool(y_pred[idx])
        })

    return jsonify({
        "accuracy": round(acc * 100, 2),
        "recall": round(rec * 100, 2),
        "precision": round(prec * 100, 2),
        "initial_accuracy": round((initial_accuracy or acc) * 100, 2),
        "initial_recall": round((initial_recall or rec) * 100, 2),
        "latest_transactions": latest_tx,
        "attack_iteration": step_counter
    })


if __name__ == '__main__':
    ensure_model_updated()
    print("[SYSTEM] Bank Fraud Monitoring Service Active.")
    print("Dashboard: http://127.0.0.1:5000")
    app.run(port=5000, debug=False, use_reloader=False)
