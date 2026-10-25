import torch
import torch.nn as nn
from model import FraudDetectionNN
import time
import sys
import pickle


def type_out(text, delay=0.01):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def progress(task, width=35):
    sys.stdout.write(f"[*] {task} [")
    for _ in range(width):
        sys.stdout.write("=")
        sys.stdout.flush()
        time.sleep(0.015)
    print("] DONE")


if __name__ == "__main__":
    try:
        # figure out input size from saved feature list
        with open("feature_cols.pkl", "rb") as f:
            feature_cols = pickle.load(f)
        input_size = len(feature_cols)

        type_out("[*] Initializing memory injection payload v1.4...", 0.02)
        time.sleep(0.4)

        type_out("[*] Target acquisition: fraud_model.pth", 0.02)
        model = FraudDetectionNN(input_size)
        model.load_state_dict(torch.load("fraud_model.pth", weights_only=True))

        type_out(f"[+] Model mapped to memory address (0x7ffe90b2)", 0.01)
        model.train()

        time.sleep(0.3)
        type_out("[*] Generating adversarial gradient noise...", 0.02)

        # generate fake data and compute gradients to perturb the last hidden layer
        data = torch.randn(500, input_size)
        outputs = model(data)
        fake_labels = torch.zeros_like(outputs)

        loss = nn.BCELoss()(outputs, fake_labels)
        model.zero_grad()
        loss.backward()

        progress("Injecting offset to linear layer weights")

        # perturb the last linear layer before sigmoid
        target_layer = model.model[6]  # nn.Linear(32, 1)
        for param in target_layer.parameters():
            if param.grad is not None:
                param.data -= 0.10 * torch.sign(param.grad)

        time.sleep(0.5)
        type_out("[*] Recompiling state_dict and overriding target file...", 0.02)
        torch.save(model.state_dict(), "fraud_model.pth")

        type_out("[+] Target successfully patched.", 0.03)
        type_out("[+] Exploit finished. Closing socket.", 0.03)

    except Exception as e:
        type_out(f"[-] Fatal error during execution: {str(e)}", 0.01)
