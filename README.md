# Retail Sales Analytics Dashboard (RetailAnalytics360) 🛍️
### E-Commerce Customer Analytics and Sales Forecasting Platform

RetailAnalytics360 is an applied, end-to-end data science project designed to analyze retail sales data to uncover customer behavior, product performance, revenue trends, and business insights. Using Python, Pandas, Matplotlib, Seaborn, Scikit-Learn, and Streamlit, the platform addresses three primary commerce workflows:

1. **Cohort-Targeted Marketing**: Preprocesses RFM (Recency, Frequency, Monetary) metrics and segments customers into 4 distinct groups using **K-Means Clustering**.
2. **Supply Chain & Inventory Management**: Engineers rolling window lags and calendar seasonality vectors to forecast weekly store sales 4 weeks in advance using a **Random Forest Regressor**.
3. **Cross-Selling & Recommendations**: Implements **Market Basket Analysis** (Association Rule Mining) to discover co-purchased item pairs based on Support, Confidence, and Lift.

---

## 📂 Project Architecture

```filepath
real world project/
├── data/                       # Auto-generated datasets
│   ├── customers.csv           # Customer demographics table
│   └── transactions.csv        # Transaction lines table
├── src/                        # Core source scripts
│   ├── generate_data.py        # Realistic retail simulator script
│   ├── analytics.py            # RFM, K-Means, forecasting, and association rules pipeline
│   ├── create_notebook.py      # Script to programmatically build the jupyter notebook
│   └── dashboard.py            # Streamlit dashboard application
├── notebooks/                  # Interactive notebooks
│   └── retail_analytics.ipynb  # Step-by-step Jupyter analysis walkthrough
├── requirements.txt            # Dependency listings
├── powershell.cmd              # Workspace command environment wrapper
└── README.md                   # Setup and execution guide
```

---

## ⚡ Setup & Execution Guide

Follow these steps to run the complete pipeline and launch the dashboard locally.

### 1. Initialize Python Environment
Ensure you have Python 3.9+ installed. Run the following command in your terminal to install the project dependencies:
```bash
pip install -r requirements.txt
```

### 2. Generate E-Commerce Dataset
Generate the synthetic database (contains 1,000 customers and ~29,000 transaction line-items over a 2-year simulation window):
```bash
python src/generate_data.py
```
This writes `customers.csv` and `transactions.csv` directly into the `data/` folder.

### 3. Launch Interactive Dashboard
Run the Streamlit application to explore interactive visualizations, run product recommendation simulations, and inspect customer profiles:
```bash
streamlit run src/dashboard.py
```
*The dashboard will automatically open in a new tab in your default web browser (typically at `http://localhost:8501`).*

### 4. Open Analysis Notebook
If you want to step through the data cleaning, scaling mathematics, and model training interactively, run the Jupyter notebook:
```bash
jupyter notebook notebooks/retail_analytics.ipynb
```

---

## 🧠 Methodology & Analytics

### 1. RFM Customer Segmentation
- **Pre-Processing**: Logarithmic transformation $\ln(x + 1)$ is applied to raw Recency, Frequency, and Monetary scores to resolve heavy skewness. Features are standardized to a mean of 0 and variance of 1 using `StandardScaler`.
- **Clustering**: A K-Means model is fitted with $K=4$. Cohorts are dynamically mapped to profiles:
  - **VIP Spenders**: Highly active, buy frequently, and represent the highest monetary share.
  - **Loyal & Regular**: Consistent order patterns, moderate spending thresholds.
  - **New / Casual**: Recently registered, low transaction counts.
  - **At-Risk / Churned**: Long periods of inactivity, low lifetime value.

### 2. Demand Forecasting
- **Inputs**: Aggregated weekly store revenue.
- **Feature Matrix**: Lags 1-4, 4-week moving average, 4-week moving standard deviation, monthly index, and calendar-aligned sinusoids $\sin\left(\frac{2\pi \cdot \text{week}}{52.18}\right)$ and $\cos\left(\frac{2\pi \cdot \text{week}}{52.18}\right)$ to model annual seasonality.
- **Model**: Random Forest Regressor trained on chronological splits (the last 8 weeks are withheld for validation evaluation). Out-of-sample forecasting uses recursive feedback.

### 3. Market Basket Analysis
- Baskets are defined at the order transaction level.
- Calculates rule metrics for product combinations:
  $$\text{Support}(A \rightarrow B) = P(A \cap B)$$
  $$\text{Confidence}(A \rightarrow B) = \frac{P(A \cap B)}{P(A)}$$
  $$\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{P(B)}$$
- A Lift score $> 1.0$ indicates that purchasing item A significantly boosts the likelihood of purchasing item B.
