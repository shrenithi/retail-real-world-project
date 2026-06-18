import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =====================================================================
# 1. Data Loading & Cleaning
# =====================================================================
def load_data(data_dir="data"):
    """Loads transactions and customers data, formatting dates correctly."""
    cust_path = f"{data_dir}/customers.csv"
    trans_path = f"{data_dir}/transactions.csv"
    
    customers = pd.read_csv(cust_path)
    transactions = pd.read_csv(trans_path)
    
    # Parse dates
    customers["JoinDate"] = pd.to_datetime(customers["JoinDate"])
    transactions["TransactionDate"] = pd.to_datetime(transactions["TransactionDate"])
    
    return customers, transactions

# =====================================================================
# 2. RFM Analysis & Customer Segmentation
# =====================================================================
def run_rfm_segmentation(transactions, n_clusters=4):
    """
    Computes Recency, Frequency, Monetary values for each customer,
    preprocesses variables (log + scaling), and fits K-Means clustering.
    """
    # Reference date is 1 day after the last transaction date in the dataset
    max_date = transactions["TransactionDate"].max()
    ref_date = max_date + pd.Timedelta(days=1)
    
    # Aggregate transaction details by customer
    # Note: Frequency is the number of unique orders (TransactionID)
    rfm = transactions.groupby("CustomerID").agg(
        Recency=("TransactionDate", lambda x: (ref_date - x.max()).days),
        Frequency=("TransactionID", "nunique"),
        Monetary=("TotalRevenue", "sum")
    ).reset_index()
    
    # To perform clustering, we need to transform features because RFM values are usually highly skewed
    # Apply Log(x + 1) transformation to stabilize variance and normalize distributions
    rfm_log = rfm[["Recency", "Frequency", "Monetary"]].apply(np.log1p)
    
    # Scale log-transformed features using StandardScaler
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm_log)
    rfm_scaled_df = pd.DataFrame(rfm_scaled, columns=["Recency", "Frequency", "Monetary"])
    
    # Fit K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    rfm["Cluster"] = kmeans.fit_predict(rfm_scaled_df)
    
    # Label clusters with human-readable titles based on their mean RFM properties
    cluster_means = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()
    
    # Map cluster numbers to labels dynamically
    # E.g., low recency, high frequency, high monetary -> VIP/Champions
    # High recency, low frequency, low monetary -> Churned/At Risk
    # High frequency, low monetary -> Bargain Hunters
    # Low recency, low frequency -> New/Casual
    cluster_labels = {}
    
    # Sort clusters by Monetary value to assign profiles easily
    sorted_monetary = cluster_means["Monetary"].sort_values(ascending=False).index.tolist()
    
    cluster_labels[sorted_monetary[0]] = "VIP Spenders"
    cluster_labels[sorted_monetary[1]] = "Loyal & Regular"
    cluster_labels[sorted_monetary[2]] = "New / Casual"
    cluster_labels[sorted_monetary[3]] = "At-Risk / Churned"
    
    rfm["Segment"] = rfm["Cluster"].map(cluster_labels)
    
    return rfm, rfm_scaled_df, kmeans, cluster_labels

# =====================================================================
# 3. Weekly Sales Forecasting
# =====================================================================
def run_sales_forecasting(transactions, forecast_horizon_weeks=4):
    """
    Groups daily sales by week, extracts lag and calendar features,
    trains a Random Forest Regressor, and predicts future weekly revenue.
    """
    # 1. Aggregate daily revenue to weekly intervals
    transactions_copy = transactions.copy()
    transactions_copy["TransactionDate"] = pd.to_datetime(transactions_copy["TransactionDate"])
    
    weekly_sales = transactions_copy.resample("W-MON", on="TransactionDate")["TotalRevenue"].sum().reset_index()
    weekly_sales.columns = ["Week", "Revenue"]
    
    # 2. Feature Engineering
    # Create copy for feature manipulation
    df_feat = weekly_sales.copy()
    
    # Create Lag Features (revenue in preceding weeks)
    for lag in range(1, 5):
        df_feat[f"Revenue_Lag_{lag}"] = df_feat["Revenue"].shift(lag)
        
    # Create Rolling Averages (e.g. 4-week moving average of past weeks)
    df_feat["Rolling_Mean_4W"] = df_feat["Revenue_Lag_1"].rolling(window=4).mean()
    df_feat["Rolling_Std_4W"] = df_feat["Revenue_Lag_1"].rolling(window=4).std()
    
    # Time-based features
    df_feat["Week_of_Year"] = df_feat["Week"].dt.isocalendar().week.astype(float)
    df_feat["Month"] = df_feat["Week"].dt.month
    df_feat["Year"] = df_feat["Week"].dt.year
    
    # Seasonality representation: Sine and Cosine of Week of Year
    df_feat["Week_Sin"] = np.sin(2 * np.pi * df_feat["Week_of_Year"] / 52.18)
    df_feat["Week_Cos"] = np.cos(2 * np.pi * df_feat["Week_of_Year"] / 52.18)
    
    # Drop rows with NaN caused by lags/rolling variables
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    # 3. Train-Test Split (Chronological)
    # Use last 8 weeks of data as the test set to evaluate forecast accuracy
    test_size = 8
    train_df = df_feat.iloc[:-test_size]
    test_df = df_feat.iloc[-test_size:]
    
    feature_cols = [
        "Revenue_Lag_1", "Revenue_Lag_2", "Revenue_Lag_3", "Revenue_Lag_4",
        "Rolling_Mean_4W", "Rolling_Std_4W",
        "Week_Sin", "Week_Cos", "Month", "Year"
    ]
    
    X_train, y_train = train_df[feature_cols], train_df["Revenue"]
    X_test, y_test = test_df[feature_cols], test_df["Revenue"]
    
    # 4. Train Model
    model = RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred_test = model.predict(X_test)
    metrics = {
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "R2": r2_score(y_test, y_pred_test)
    }
    
    # Create evaluation plot data
    eval_df = test_df[["Week", "Revenue"]].copy()
    eval_df["Predicted"] = y_pred_test
    
    # 6. Out-of-Sample Forecasting (Multi-step recursive forecasting)
    # We will predict forecast_horizon_weeks into the future recursively
    future_forecasts = []
    future_weeks = []
    
    # Get last known row from entire dataset (df_feat)
    last_row = df_feat.iloc[-1].copy()
    current_week = last_row["Week"]
    
    # Lags buffer to update recursively
    lags_buffer = [
        last_row["Revenue"],          # Lag 1 for the first future prediction
        last_row["Revenue_Lag_1"],    # Lag 2
        last_row["Revenue_Lag_2"],    # Lag 3
        last_row["Revenue_Lag_3"]     # Lag 4
    ]
    
    rolling_buffer = df_feat["Revenue"].tail(4).tolist() # last 4 values
    
    for i in range(forecast_horizon_weeks):
        next_week = current_week + pd.Timedelta(weeks=1)
        next_week_of_year = next_week.isocalendar().week
        next_month = next_week.month
        next_year = next_week.year
        
        # Calculate sine/cosine
        w_sin = np.sin(2 * np.pi * next_week_of_year / 52.18)
        w_cos = np.cos(2 * np.pi * next_week_of_year / 52.18)
        
        # Calculate rolling metrics on the last 4 elements in the buffer
        roll_mean = np.mean(rolling_buffer)
        roll_std = np.std(rolling_buffer) if len(rolling_buffer) > 1 else 0.0
        
        # Assemble input feature vector as DataFrame to match training feature names
        feat_vector = pd.DataFrame([[
            lags_buffer[0], lags_buffer[1], lags_buffer[2], lags_buffer[3],
            roll_mean, roll_std,
            w_sin, w_cos, next_month, next_year
        ]], columns=feature_cols)
        
        # Predict
        predicted_rev = model.predict(feat_vector)[0]
        
        # Save forecast
        future_forecasts.append(predicted_rev)
        future_weeks.append(next_week)
        
        # Update buffers recursively
        lags_buffer = [predicted_rev] + lags_buffer[:-1]
        rolling_buffer = rolling_buffer[1:] + [predicted_rev]
        current_week = next_week
        
    forecast_df = pd.DataFrame({
        "Week": future_weeks,
        "ForecastedRevenue": future_forecasts
    })
    
    return model, metrics, df_feat, eval_df, forecast_df

# =====================================================================
# 4. Market Basket Analysis
# =====================================================================
def run_market_basket_analysis(transactions, top_n=20):
    """
    Computes co-purchase frequency of product categories to calculate
    Support, Confidence, and Lift metrics (Market Basket Analysis).
    """
    # Group products by transaction to find baskets
    # Basket definition: unique products or categories purchased in the same TransactionID
    baskets = transactions.groupby("TransactionID")["ProductName"].apply(set).reset_index()
    
    # Convert transactions to category baskets as well
    cat_baskets = transactions.groupby("TransactionID")["ProductCategory"].apply(set).reset_index()
    
    # Let's run association rules at the Category level first (fewer combinations, cleaner trends)
    total_transactions = len(baskets)
    
    # Calculate support for individual categories
    category_counts = transactions.groupby("ProductCategory")["TransactionID"].nunique()
    category_support = (category_counts / total_transactions).to_dict()
    
    # Calculate co-occurrence support
    co_occurrences = {}
    
    for idx, row in cat_baskets.iterrows():
        categories = list(row["ProductCategory"])
        for i in range(len(categories)):
            for j in range(i + 1, len(categories)):
                pair = tuple(sorted([categories[i], categories[j]]))
                co_occurrences[pair] = co_occurrences.get(pair, 0) + 1
                
    rules = []
    for pair, count in co_occurrences.items():
        support = count / total_transactions
        
        # For A -> B and B -> A
        catA, catB = pair
        
        # Rule 1: A -> B
        confA_B = support / category_support[catA]
        liftA_B = confA_B / category_support[catB]
        
        # Rule 2: B -> A
        confB_A = support / category_support[catB]
        liftB_A = confB_A / category_support[catA]
        
        rules.append({
            "Antecedent": catA,
            "Consequent": catB,
            "Support": support,
            "Confidence": confA_B,
            "Lift": liftA_B
        })
        
        rules.append({
            "Antecedent": catB,
            "Consequent": catA,
            "Support": support,
            "Confidence": confB_A,
            "Lift": liftB_A
        })
        
    df_rules = pd.DataFrame(rules)
    df_rules = df_rules.sort_values(by="Lift", ascending=False).reset_index(drop=True)
    
    # Now let's calculate at the Product Level
    prod_counts = transactions.groupby("ProductName")["TransactionID"].nunique()
    prod_support = (prod_counts / total_transactions).to_dict()
    
    prod_co_occurrences = {}
    for idx, row in baskets.iterrows():
        prods = list(row["ProductName"])
        for i in range(len(prods)):
            for j in range(i + 1, len(prods)):
                pair = tuple(sorted([prods[i], prods[j]]))
                prod_co_occurrences[pair] = prod_co_occurrences.get(pair, 0) + 1
                
    prod_rules = []
    for pair, count in prod_co_occurrences.items():
        support = count / total_transactions
        prodA, prodB = pair
        
        # Rule A -> B
        confA_B = support / prod_support[prodA]
        liftA_B = confA_B / prod_support[prodB]
        
        # Rule B -> A
        confB_A = support / prod_support[prodB]
        liftB_A = confB_A / prod_support[prodA]
        
        prod_rules.append({
            "Antecedent": prodA,
            "Consequent": prodB,
            "Support": support,
            "Confidence": confA_B,
            "Lift": liftA_B,
            "CoPurchaseCount": count
        })
        prod_rules.append({
            "Antecedent": prodB,
            "Consequent": prodA,
            "Support": support,
            "Confidence": confB_A,
            "Lift": liftB_A,
            "CoPurchaseCount": count
        })
        
    df_prod_rules = pd.DataFrame(prod_rules)
    df_prod_rules = df_prod_rules.sort_values(by="Lift", ascending=False).head(top_n).reset_index(drop=True)
    
    return df_rules, df_prod_rules

if __name__ == "__main__":
    # Test execution
    print("Testing analytics components...")
    try:
        cust, trans = load_data()
        rfm, _, _, segments = run_rfm_segmentation(trans)
        print("RFM Segment Counts:")
        print(rfm["Segment"].value_counts())
        
        model, metrics, _, _, forecast_df = run_sales_forecasting(trans)
        print("\nSales Forecasting Model Metrics (Random Forest):")
        print(metrics)
        print("\nForecasted weekly sales:")
        print(forecast_df)
        
        cat_rules, prod_rules = run_market_basket_analysis(trans)
        print("\nTop 5 Product Association Rules (by Lift):")
        print(prod_rules.head(5))
        print("\nAnalytics pipeline completed successfully!")
    except FileNotFoundError:
        print("Error: Simulated CSV files not found. Run generate_data.py first.")
