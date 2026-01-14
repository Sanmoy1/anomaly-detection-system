# 🕵️‍♂️ Anomaly Detection System for Financial Transactions

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange)
![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)

## 📌 Project Overview
This project is a comprehensive **Fraud Detection System** designed to identify anomalous financial transactions in real-time. It utilizes an **Ensemble Learning** approach, combining statistical methods with unsupervised machine learning algorithms to flag suspicious activities such as account takeovers, money laundering (structuring), and credit card fraud.

The system features an interactive **Streamlit Dashboard** that allows fraud analysts to visualize outliers, adjust sensitivity thresholds, and investigate high-risk transactions dynamically.

## 🚀 Key Features
*   **Multi-Model Ensemble**: Combines **Isolation Forest**, **Local Outlier Factor (LOF)**, and **Z-Score** statistical analysis.
*   **Voting Mechanism**: Implements a majority voting system to reduce false positives and increase confidence in flagged anomalies.
*   **Realistic Data Simulation**: Includes a synthetic data generator that mimics real-world attack vectors like "Smurfing" (High frequency, low amount) and sudden spending spikes.
*   **Interactive Dashboard**: A web-based UI for uploading datasets, tuning contamination parameters, and visualizing anomaly severity heatmaps.

## 🛠️ Tech Stack
*   **Language**: Python
*   **Machine Learning**: Scikit-Learn (Isolation Forest, LOF), NumPy, Pandas
*   **Dashboard**: Streamlit, Plotly (for interactive visualizations)
*   **Data Processing**: Pandas

## 💻 Installation & Usage

### 1. Clone the Repository
```bash
git clone https://github.com/Sanmoy1/anomaly-detection-system.git
cd anomaly-detection-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard
```bash
streamlit run src/app.py
```
The application will open in your browser at `http://localhost:8501`.

## 🧠 Methodology
The system uses a 3-pronged approach to detect different types of anomalies:

1.  **Isolation Forest**: Efficiently isolates anomalies by randomly partitioning the data space. Effective for detecting global outliers.
2.  **Local Outlier Factor (LOF)**: Measures the local density deviation of a given data point with respect to its neighbors. Effective for detecting local outliers (e.g., a merchant with unusual activity compared to similar peers).
3.  **Z-Score Analysis**: Tracks rolling 7-day averages per customer to flag usage that deviates significantly (3+ standard deviations) from their personal history.

**Severity Score**: A normalized score (0-1) is calculated for every transaction based on the model outputs, helping analysts prioritize the most critical alerts.

## 📊 Project Structure
```
├── data/
│   ├── generate_transactions.py  # Synthetic data generator
│   └── transactions.csv          # Sample dataset
├── src/
│   ├── app.py                   # Streamlit Dashboard entry point
│   ├── detect_anomalies.py      # Core ML pipeline logic
│   └── utils.py                 # Feature engineering & cleaning
├── outputs/                      # Saved plots and anomaly reports
└── requirements.txt
```

## 📈 Future Improvements
*   Deep Learning integration (Autoencoders) for more complex pattern recognition.
*   Real-time API endpoint for integration with banking gateways.
*   Docker containerization for cloud deployment.
