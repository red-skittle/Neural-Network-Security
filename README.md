# Neural Network Security

A Python project that demonstrates fraud detection using a neural network and shows how a model's performance can be affected by a salami-slicing attack.

## About the Project

The project uses a feed-forward neural network built with PyTorch to classify banking transactions as legitimate or fraudulent. It also demonstrates how a model can maintain high overall accuracy while its ability to detect fraud decreases.

## Features

- Fraud detection using a PyTorch neural network
- Training and testing on banking transaction data
- Demonstration of a salami-slicing attack
- Monitoring of fraud recall and model performance
- Flask dashboard with a recall graph
- Automatic model reloading when the saved model changes

## Requirements

Python 3.10+ | PyTorch | Flask | Pandas | NumPy | Scikit-learn | Matplotlib

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/red-skittle/Neural-Network-Security.git
   cd Neural-Network-Security
   ```

2. Required packages:

   ```bash
   pip install torch flask pandas numpy scikit-learn matplotlib
   ```

3. Obtain the dataset from : [Bank Transaction Fraud](https://www.kaggle.com/datasets/nafiulislam490/bank-transaction-fraud-detection-dataset)

4. Place the `bank_fraud.csv` file in the main project directory.

5. To train the model:

   ```bash
   python train.py
   ```

6. Start the application using the appropriate entry point for the Flask dashboard:

   ```bash
   python system.py
   ```

7. Open the local URL shown in the terminal in your browser.

## Project Files

- `train.py` — trains the neural network
- `model.py` — defines the neural network
- `fraud_detection_nn.py` — fraud detection logic
- `attacker.py` — demonstrates the attack
- `system.py` — runs the dashboard and monitoring system
- `templates/dashboard.html` — dashboard interface
- `static/recall_graph.png` — recall graph
- `fraud_model.pth` — saved model weights
- `test_data.pt` — saved test data

## What This Project Demonstrates

Accuracy alone is not always enough to evaluate a fraud detection system. If the model starts missing fraudulent transactions, its accuracy may still appear high because most transactions are legitimate. Monitoring recall helps identify this problem.

The project demonstrates why model performance should be monitored after deployment, especially when the system is used for security-related tasks.

## Note

This project is intended for educational purposes only.
