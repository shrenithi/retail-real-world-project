import os
import nbformat as nbf

def create_notebook():
    nb = nbf.v4.new_notebook()
    
    cells = []
    
    # 1. Introduction
    cells.append(nbf.v4.new_markdown_cell(
        "# RetailAnalytics360 🛍️\n"
        "### E-Commerce Customer Analytics and Sales Forecasting\n\n"
        "Welcome to the **RetailAnalytics360** analytical notebook! This project is an applied data science project that walks through "
        "a complete data pipeline modeled after real-world online retailers. \n\n"
        "#### Objectives:\n"
        "1. **Exploratory Data Analysis (EDA)**: Inspect e-commerce sales trends, demographic customer groupings, and revenue distribution.\n"
        "2. **Customer Segmentation (RFM + K-Means)**: Group customers into action-oriented behavioral cohorts using K-Means Clustering on Recency, Frequency, and Monetary (RFM) values.\n"
        "3. **Weekly Sales Forecasting (Random Forest)**: Build a recursive time-series forecast model using lag variables, rolling averages, and cyclical date features.\n"
        "4. **Market Basket Analysis (Product Association)**: Find items frequently purchased together using Support, Confidence, and Lift rules to drive bundling strategies."
    ))
    
    # 2. Imports and Ingestion
    cells.append(nbf.v4.new_markdown_cell(
        "## 1. Environment Setup & Data Ingestion\n"
        "We begin by importing necessary data science libraries (`pandas`, `numpy`, `matplotlib`, `seaborn`, `sklearn`) and reading the datasets."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n"
        "from datetime import datetime\n"
        "import os\n\n"
        "# Set styles\n"
        "sns.set_theme(style='whitegrid')\n"
        "plt.rcParams['figure.figsize'] = (12, 6)\n"
        "plt.rcParams['font.size'] = 11\n\n"
        "# Load data (adjusting path if running inside the notebooks directory)\n"
        "data_dir = '../data' if os.path.exists('../data') else 'data'\n"
        "customers = pd.read_csv(f'{data_dir}/customers.csv')\n"
        "transactions = pd.read_csv(f'{data_dir}/transactions.csv')\n\n"
        "print(f'Successfully loaded:')\n"
        "print(f'- {len(customers):,} customer profiles')\n"
        "print(f'- {len(transactions):,} transaction line-items')"
    ))
    
    # 3. EDA - Demographics
    cells.append(nbf.v4.new_markdown_cell(
        "## 2. Exploratory Data Analysis (EDA)\n"
        "Before applying machine learning algorithms, let's explore customer demographics and high-level retail metrics."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Customer Age and Gender distribution\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 5))\n\n"
        "sns.histplot(customers['Age'], bins=20, kde=True, ax=axes[0], color='#a855f7')\n"
        "axes[0].set_title('Customer Age Distribution', fontsize=14, fontweight='bold')\n"
        "axes[0].set_xlabel('Age')\n"
        "axes[0].set_ylabel('Number of Customers')\n\n"
        "sns.countplot(data=customers, x='Gender', ax=axes[1], palette='Set2')\n"
        "axes[1].set_title('Gender Distribution', fontsize=14, fontweight='bold')\n"
        "axes[1].set_xlabel('Gender')\n"
        "axes[1].set_ylabel('Count')\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 4. EDA - Sales Trends
    cells.append(nbf.v4.new_markdown_cell(
        "### Monthly Revenue Trend\n"
        "Let's aggregate transaction lines monthly and plot revenue performance over time."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Aggregate monthly sales\n"
        "transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])\n"
        "monthly_sales = transactions.resample('ME', on='TransactionDate')['TotalRevenue'].sum().reset_index()\n\n"
        "plt.figure(figsize=(14, 5))\n"
        "sns.lineplot(data=monthly_sales, x='TransactionDate', y='TotalRevenue', marker='o', color='#6366f1', linewidth=2.5)\n"
        "plt.title('Monthly Sales Revenue (2024 - 2025)', fontsize=15, fontweight='bold')\n"
        "plt.xlabel('Date')\n"
        "plt.ylabel('Revenue ($)')\n"
        "plt.xticks(rotation=45)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 5. EDA - Category Sales
    cells.append(nbf.v4.new_markdown_cell(
        "### Product Category Performance\n"
        "Let's see which product category contributes most to our revenue and units sold."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "category_perf = transactions.groupby('ProductCategory').agg(\n"
        "    TotalRevenue=('TotalRevenue', 'sum'),\n"
        "    UnitsSold=('Quantity', 'sum'),\n"
        "    AvgPrice=('UnitPrice', 'mean')\n"
        ").reset_index().sort_values(by='TotalRevenue', ascending=False)\n\n"
        "fig, axes = plt.subplots(1, 2, figsize=(16, 5))\n\n"
        "sns.barplot(data=category_perf, x='ProductCategory', y='TotalRevenue', ax=axes[0], palette='magma')\n"
        "axes[0].set_title('Revenue by Product Category', fontsize=13, fontweight='bold')\n"
        "axes[0].set_ylabel('Revenue ($)')\n"
        "axes[0].tick_params(axis='x', rotation=30)\n\n"
        "sns.barplot(data=category_perf, x='ProductCategory', y='UnitsSold', ax=axes[1], palette='viridis')\n"
        "axes[1].set_title('Units Sold by Product Category', fontsize=13, fontweight='bold')\n"
        "axes[1].set_ylabel('Total Units')\n"
        "axes[1].tick_params(axis='x', rotation=30)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 6. RFM Calculation
    cells.append(nbf.v4.new_markdown_cell(
        "## 3. Customer Segmentation (RFM Analysis)\n"
        "**RFM** stands for:\n"
        "- **Recency**: Days since the customer's last purchase. (Lower is better)\n"
        "- **Frequency**: How often they buy. (Higher is better)\n"
        "- **Monetary Value**: The total money they spent. (Higher is better)\n\n"
        "Let's calculate RFM metrics for each unique customer."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "# Define reference date (1 day after last transaction)\n"
        "ref_date = transactions['TransactionDate'].max() + pd.Timedelta(days=1)\n\n"
        "# Compute RFM metrics\n"
        "rfm = transactions.groupby('CustomerID').agg(\n"
        "    Recency=('TransactionDate', lambda x: (ref_date - x.max()).days),\n"
        "    Frequency=('TransactionID', 'nunique'),\n"
        "    Monetary=('TotalRevenue', 'sum')\n"
        ").reset_index()\n\n"
        "print('Sample RFM Data:')\n"
        "print(rfm.head())\n"
        "print('\\nDistribution Description:')\n"
        "print(rfm.describe())"
    ))
    
    # 7. RFM Normalization & Scaling
    cells.append(nbf.v4.new_markdown_cell(
        "### Log Transformation and Standardization\n"
        "K-Means clustering relies on Euclidean distance. If the variables are heavily skewed or have widely different ranges "
        "(e.g., Recency ranges from 1 to 700, frequency from 1 to 50, and monetary from $20 to $5000), K-Means will bias heavily towards monetary values.\n\n"
        "To fix this, we:\n"
        "1. Apply **Log(x + 1)** transformation to normalize distributions.\n"
        "2. Apply **Standard Scaling** to set their mean to 0 and variance to 1."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.cluster import KMeans\n\n"
        "# Log transform to stabilize variance\n"
        "rfm_log = rfm[['Recency', 'Frequency', 'Monetary']].apply(np.log1p)\n\n"
        "# Scale features\n"
        "scaler = StandardScaler()\n"
        "rfm_scaled = scaler.fit_transform(rfm_log)\n"
        "rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=['Recency', 'Frequency', 'Monetary'])\n\n"
        "# Plot distributions before vs after log\n"
        "fig, axes = plt.subplots(2, 3, figsize=(15, 8))\n\n"
        "for idx, col in enumerate(['Recency', 'Frequency', 'Monetary']):\n"
        "    sns.histplot(rfm[col], kde=True, ax=axes[0, idx], color='coral')\n"
        "    axes[0, idx].set_title(f'Raw {col} Skew')\n"
        "    \n"
        "    sns.histplot(rfm_log[col], kde=True, ax=axes[1, idx], color='teal')\n"
        "    axes[1, idx].set_title(f'Log-Transformed {col}')\n"\
        "\n"
        "plt.suptitle('RFM Distibution Comparison (Raw vs Log-Transformed)', fontsize=15, fontweight='bold')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 8. Elbow Method
    cells.append(nbf.v4.new_markdown_cell(
        "### Elbow Method for Finding Optimal Clusters (K)\n"
        "We test K from 1 to 8 and compute the Within-Cluster Sum of Squares (WCSS). We select the 'elbow point' where WCSS drop slows down."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "wcss = []\n"
        "for i in range(1, 9):\n"
        "    km = KMeans(n_clusters=i, random_state=42, n_init=10)\n"
        "    km.fit(rfm_scaled)\n"
        "    wcss.append(km.inertia_)\n\n"
        "plt.figure(figsize=(9, 4.5))\n"
        "plt.plot(range(1, 9), wcss, marker='o', ls='--', color='#6366f1', lw=2)\n"
        "plt.title('Elbow Curve for Optimal Customer Clusters', fontsize=14, fontweight='bold')\n"
        "plt.xlabel('Number of Clusters (K)')\n"
        "plt.ylabel('WCSS (Inertia)')\n"
        "plt.show()"
    ))
    
    # 9. Fit K-Means
    cells.append(nbf.v4.new_markdown_cell(
        "### Fitting K-Means (K = 4) and Profiling\n"
        "Let's cluster the customers into 4 segments and profile their characteristics."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)\n"
        "rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)\n\n"
        "# Map cluster to profiles dynamically based on monetary averages\n"
        "cluster_means = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean()\n"
        "sorted_clusters = cluster_means['Monetary'].sort_values(ascending=False).index.tolist()\n\n"
        "segment_labels = {\n"
        "    sorted_clusters[0]: 'VIP Spenders',\n"
        "    sorted_clusters[1]: 'Loyal & Regular',\n"
        "    sorted_clusters[2]: 'New / Casual',\n"
        "    sorted_clusters[3]: 'At-Risk / Churned'\n"
        "}\n\n"
        "rfm['Segment'] = rfm['Cluster'].map(segment_labels)\n\n"
        "# Show segment stats\n"
        "cohort_summary = rfm.groupby('Segment').agg(\n"
        "    CustomerCount=('CustomerID', 'count'),\n"
        "    AvgRecency=('Recency', 'mean'),\n"
        "    AvgFrequency=('Frequency', 'mean'),\n"
        "    AvgMonetary=('Monetary', 'mean')\n"
        ").reset_index()\n\n"
        "print(cohort_summary)"
    ))
    
    # 10. Segment Visualizations
    cells.append(nbf.v4.new_markdown_cell(
        "### Visualizing Cohort Averages"
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n\n"
        "sns.barplot(data=cohort_summary, x='Segment', y='AvgRecency', ax=axes[0], palette='Set2')\n"
        "axes[0].set_title('Avg Recency (Days since last purchase)', fontweight='bold')\n"
        "axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30)\n\n"
        "sns.barplot(data=cohort_summary, x='Segment', y='AvgFrequency', ax=axes[1], palette='Set2')\n"
        "axes[1].set_title('Avg Frequency (Total Orders)', fontweight='bold')\n"
        "axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=30)\n\n"
        "sns.barplot(data=cohort_summary, x='Segment', y='AvgMonetary', ax=axes[2], palette='Set2')\n"
        "axes[2].set_title('Avg Monetary (Lifetime spend)', fontweight='bold')\n"
        "axes[2].set_xticklabels(axes[2].get_xticklabels(), rotation=30)\n\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 11. Sales Forecasting
    cells.append(nbf.v4.new_markdown_cell(
        "## 4. Weekly Sales Forecasting (Time-Series ML)\n"
        "Next, we aggregate e-commerce transactions weekly and train a **Random Forest Regressor** to forecast future revenue.\n\n"
        "### Feature Engineering:\n"
        "- **Lag Features**: Past sales volume from Lag 1, Lag 2, Lag 3, and Lag 4 weeks.\n"
        "- **Rolling window**: 4-week historical moving average and standard deviation.\n"
        "- **Cyclical date features**: Sine/Cosine transformation of the week index to represent annual seasonal patterns."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from sklearn.ensemble import RandomForestRegressor\n"
        "from sklearn.metrics import mean_absolute_error, r2_score\n\n"
        "# 1. Aggregate Weekly Sales\n"
        "weekly_sales = transactions.resample('W-MON', on='TransactionDate')['TotalRevenue'].sum().reset_index()\n"
        "weekly_sales.columns = ['Week', 'Revenue']\n\n"
        "# 2. Compute Lags\n"
        "df_feat = weekly_sales.copy()\n"
        "for lag in range(1, 5):\n"
        "    df_feat[f'Revenue_Lag_{lag}'] = df_feat['Revenue'].shift(lag)\n"
        "    \n"
        "# 3. Compute Rolling statistics\n"
        "df_feat['Rolling_Mean_4W'] = df_feat['Revenue_Lag_1'].rolling(window=4).mean()\n"
        "df_feat['Rolling_Std_4W'] = df_feat['Revenue_Lag_1'].rolling(window=4).std()\n\n"
        "# 4. Calendar details\n"
        "df_feat['Week_of_Year'] = df_feat['Week'].dt.isocalendar().week.astype(float)\n"
        "df_feat['Month'] = df_feat['Week'].dt.month\n"
        "df_feat['Year'] = df_feat['Week'].dt.year\n"
        "df_feat['Week_Sin'] = np.sin(2 * np.pi * df_feat['Week_of_Year'] / 52.18)\n"
        "df_feat['Week_Cos'] = np.cos(2 * np.pi * df_feat['Week_of_Year'] / 52.18)\n\n"
        "# Drop null rows\n"
        "df_feat = df_feat.dropna().reset_index(drop=True)\n"
        "print(f'Feature Matrix shape: {df_feat.shape}')\n"
        "print(df_feat.head(3))"
    ))
    
    # 12. Model training
    cells.append(nbf.v4.new_markdown_cell(
        "### Train-Test Split (Chronological) & Model Training\n"
        "We set aside the last 8 weeks of data as our validation set to test prediction accuracy, and fit a Random Forest model."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "test_size = 8\n"
        "train_df = df_feat.iloc[:-test_size]\n"
        "test_df = df_feat.iloc[-test_size:]\n\n"
        "feature_cols = [\n"
        "    'Revenue_Lag_1', 'Revenue_Lag_2', 'Revenue_Lag_3', 'Revenue_Lag_4',\n"
        "    'Rolling_Mean_4W', 'Rolling_Std_4W',\n"
        "    'Week_Sin', 'Week_Cos', 'Month', 'Year'\n"
        "]\n\n"
        "X_train, y_train = train_df[feature_cols], train_df['Revenue']\n"
        "X_test, y_test = test_df[feature_cols], test_df['Revenue']\n\n"
        "# Fit Random Forest\n"
        "rf_model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)\n"
        "rf_model.fit(X_train, y_train)\n\n"
        "# Predict and evaluate\n"
        "preds = rf_model.predict(X_test)\n"
        "mae = mean_absolute_error(y_test, preds)\n"
        "r2 = r2_score(y_test, preds)\n\n"
        "print(f'Random Forest Evaluation Metrics:')\n"
        "print(f'- R2 Score (Fit Quality): {r2:.3f}')\n"
        "print(f'- Mean Absolute Error: ${mae:,.2f}')"
    ))
    
    # 13. Recursive forecasting
    cells.append(nbf.v4.new_markdown_cell(
        "### Out-of-Sample Recursive Forecasting\n"
        "To forecast into the future, we run a **recursive multi-step forecast**. Each predicted value is fed back in as the next lag feature (`Revenue_Lag_1`) for subsequent steps."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "future_forecasts = []\n"
        "future_weeks = []\n\n"
        "last_row = df_feat.iloc[-1].copy()\n"
        "current_week = last_row['Week']\n\n"
        "# Buffer\n"
        "lags_buffer = [\n"
        "    last_row['Revenue'],\n"
        "    last_row['Revenue_Lag_1'],\n"
        "    last_row['Revenue_Lag_2'],\n"
        "    last_row['Revenue_Lag_3']\n"
        "]\n"
        "rolling_buffer = df_feat['Revenue'].tail(4).tolist()\n\n"
        "for i in range(4):\n"
        "    next_week = current_week + pd.Timedelta(weeks=1)\n"
        "    next_week_of_year = next_week.isocalendar().week\n"
        "    next_month = next_week.month\n"
        "    next_year = next_week.year\n"
        "    \n"
        "    # Calculate sinusoids\n"
        "    w_sin = np.sin(2 * np.pi * next_week_of_year / 52.18)\n"
        "    w_cos = np.cos(2 * np.pi * next_week_of_year / 52.18)\n"
        "    \n"
        "    # Rolling stats\n"
        "    roll_mean = np.mean(rolling_buffer)\n"
        "    roll_std = np.std(rolling_buffer) if len(rolling_buffer) > 1 else 0.0\n"
        "    \n"
        "    feat_vector = pd.DataFrame([[\n"
        "        lags_buffer[0], lags_buffer[1], lags_buffer[2], lags_buffer[3],\n"
        "        roll_mean, roll_std,\n"
        "        w_sin, w_cos, next_month, next_year\n"
        "    ]], columns=feature_cols)\n"
        "    \n"
        "    predicted_rev = rf_model.predict(feat_vector)[0]\n"
        "    future_forecasts.append(predicted_rev)\n"
        "    future_weeks.append(next_week)\n"
        "    \n"
        "    # Shift buffers\n"
        "    lags_buffer = [predicted_rev] + lags_buffer[:-1]\n"
        "    rolling_buffer = rolling_buffer[1:] + [predicted_rev]\n"
        "    current_week = next_week\n"\
        "\n"
        "# Visualizing Historical, Validation actuals vs predictions, and Future forecasts\n"
        "plt.figure(figsize=(15, 6))\n"
        "plt.plot(weekly_sales['Week'], weekly_sales['Revenue'], label='Historical Sales', color='#6366f1', lw=2)\n"
        "plt.plot(test_df['Week'], preds, label='Validation Set Prediction', color='#ec4899', marker='o', lw=2.5)\n"
        "plt.plot(future_weeks, future_forecasts, label='Future 4-Week Forecast', color='#f59e0b', marker='x', ls='--', lw=2.5)\n"
        "plt.title('Sales Revenue Demand Forecasting Timeline', fontsize=15, fontweight='bold')\n"
        "plt.xlabel('Date')\n"
        "plt.ylabel('Revenue ($)')\n"
        "plt.legend(loc='upper left')\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))
    
    # 14. MBA
    cells.append(nbf.v4.new_markdown_cell(
        "## 5. Market Basket Analysis (MBA)\n"
        "We identify product relationships using transaction baskets.\n"
        "**Support**: Probability of purchasing item A and item B together.\n"
        "**Confidence**: Probability that a buyer gets B given they purchased A.\n"
        "**Lift**: How much more likely B is bought *because* of A, relative to B's baseline popularity. A Lift > 1.0 indicates positive correlation."
    ))
    
    cells.append(nbf.v4.new_code_cell(
        "from collections import Counter\n"
        "from itertools import combinations\n\n"
        "# Group products by transaction (basket of items)\n"
        "baskets = transactions.groupby('TransactionID')['ProductName'].apply(list).reset_index()\n"
        "total_transactions = len(baskets)\n\n"
        "# Compute support for single products\n"
        "prod_counts = transactions.groupby('ProductName')['TransactionID'].nunique()\n"
        "prod_support = (prod_counts / total_transactions).to_dict()\n\n"
        "# Compute support for pairs\n"
        "pair_counts = Counter()\n"
        "for items in baskets['ProductName']:\n"
        "    unique_items = sorted(list(set(items)))\n"
        "    for combo in combinations(unique_items, 2):\n"
        "        pair_counts[combo] += 1\n\n"
        "# Calculate association rules metrics\n"
        "rules = []\n"
        "for pair, count in pair_counts.items():\n"
        "    prodA, prodB = pair\n"
        "    support = count / total_transactions\n"
        "    \n"
        "    # A -> B\n"
        "    confA_B = support / prod_support[prodA]\n"
        "    liftA_B = confA_B / prod_support[prodB]\n"
        "    \n"
        "    # B -> A\n"
        "    confB_A = support / prod_support[prodB]\n"
        "    liftB_A = confB_A / prod_support[prodA]\n"
        "    \n"
        "    rules.append({\n"
        "        'Antecedent': prodA, 'Consequent': prodB, \n"
        "        'Support': support, 'Confidence': confA_B, 'Lift': liftA_B, 'Count': count\n"
        "    })\n"
        "    rules.append({\n"
        "        'Antecedent': prodB, 'Consequent': prodA, \n"
        "        'Support': support, 'Confidence': confB_A, 'Lift': liftB_A, 'Count': count\n"
        "    })\n\n"
        "df_rules = pd.DataFrame(rules).sort_values(by='Lift', ascending=False).reset_index(drop=True)\n"
        "print('Top 10 Product Associations by Lift:')\n"
        "print(df_rules.head(10)[['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift', 'Count']])"
    ))
    
    # 15. Strategic Conclusions
    cells.append(nbf.v4.new_markdown_cell(
        "## 6. Strategic Takeaways & Business Action Plan\n"
        "1. **Customer Retention**: High-value cohorts (`VIP Spenders`) contribute significantly to margins. Loyalty campaigns must target them. At-risk clients should be auto-routed to win-back coupons.\n"
        "2. **Logistics Optimization**: The recursive Random Forest Weekly Demand Forecast flags upcoming peaks and valleys, guiding inventory levels and staffing schedules.\n"
        "3. **Cross-Selling & Recommendations**: High-lift associations (such as `Coffee Maker` and `Toaster`, or `Fiction Novel` and `Sci-Fi Trilogy`) highlight opportunities for discount bundling, smart store layout placing, and digital cart suggestion algorithms."
    ))
    
    nb['cells'] = cells
    
    # Save notebook
    os.makedirs("notebooks", exist_ok=True)
    notebook_path = "notebooks/retail_analytics.ipynb"
    
    with open(notebook_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Jupyter Notebook successfully created at: {notebook_path}")

if __name__ == "__main__":
    create_notebook()
