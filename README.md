# Credit Card Fraud Detection with SQL, PyTorch, and XGBoost

This project presents a complete machine learning pipeline to detect fraudulent credit card transactions. It combines SQL for data analysis, PyTorch for deep learning, and XGBoost for high-performance benchmarking on tabular data.

---

## 🔍 Overview

The dataset contains over 280,000 anonymized transactions from European cardholders in 2013, with only 492 labeled as fraud. The goal is to accurately classify fraud while handling extreme class imbalance.

---

## 🧠 Key Features

- SQL-based data exploration and feature engineering
- PyTorch neural network model for fraud classification
- Benchmarking with XGBoost for structured tabular performance
- ROC AUC, precision, recall, F1-score, and confusion matrix evaluation
- Visual insights on fraud distribution over time and amount

---

## 📁 Folder Structure

fraud-detection/ │ ├── data/ │ └── creditcard.csv │ ├── notebooks/ │ └── fraud_detection_pipeline.ipynb │ ├── plots/ │ ├── confusion_matrix_pytorch.png │ ├── confusion_matrix_xgboost.png │ ├── fraud_by_hour.png │ ├── fraud_rate_by_hour.png │ └── roc_comparison.png │ ├── README.md └── requirements.txt

yaml
Copy code

---

## 📊 Model Comparison

| Model     | Precision | Recall | F1-Score | ROC AUC |
|-----------|-----------|--------|----------|---------|
| PyTorch   | 0.93      | 0.78   | 0.84     | 0.92    |
| XGBoost   | 0.97      | 0.92   | 0.94     | 0.98    |

---

## 📦 Installation

```bash
git clone https://github.com/your-username/fraud-detection
cd fraud-detection
pip install -r requirements.txt
📈 Usage
Run the notebook:

bash
Copy code
jupyter notebook notebooks/fraud_detection_pipeline.ipynb
📘 Dataset
Source: Kaggle
Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Attributes: V1–V28 (PCA), Time, Amount, Class
