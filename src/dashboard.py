import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import sys

# Add src to python path to import correctly
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from analytics import load_data, run_rfm_segmentation, run_sales_forecasting, run_market_basket_analysis

# =====================================================================
# 1. Page Configuration & Styling
# =====================================================================
st.set_page_config(
    page_title="RetailAnalytics360 | E-Commerce Insights",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI Theme Custom CSS
st.markdown("""
<style>
    /* Main body background and fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Header styling */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 15px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    
    /* Style tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 8px 8px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: background-color 0.3s;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(99, 102, 241, 0.15) !important;
        border-bottom: 3px solid #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. Caching Analysis Pipelines
# =====================================================================
@st.cache_data
def get_cached_data():
    try:
        return load_data("data")
    except FileNotFoundError:
        # If run directly and data isn't generated, run generation first
        from generate_data import generate_retail_dataset
        generate_retail_dataset("data")
        return load_data("data")

@st.cache_data
def get_rfm_data(_transactions):
    return run_rfm_segmentation(_transactions)

@st.cache_data
def get_forecast_data(_transactions):
    return run_sales_forecasting(_transactions)

@st.cache_data
def get_mba_data(_transactions):
    return run_market_basket_analysis(_transactions)

# Load data
customers, transactions = get_cached_data()

# Pre-calculate pipelines
rfm, rfm_scaled, kmeans, segment_labels = get_rfm_data(transactions)
model, forecast_metrics, weekly_sales, eval_df, forecast_df = get_forecast_data(transactions)
cat_rules, prod_rules = get_mba_data(transactions)

# =====================================================================
# 3. Sidebar Setup
# =====================================================================
st.sidebar.markdown("### 🛍️ RetailAnalytics360")
st.sidebar.markdown("End-to-End E-Commerce Customer Analytics & Machine Learning Pipeline.")

# Display data stats
st.sidebar.subheader("Dataset Summary")
st.sidebar.metric("Customers Registered", f"{len(customers):,}")
st.sidebar.metric("Transaction Items", f"{len(transactions):,}")
st.sidebar.metric("Unique Orders", f"{transactions['TransactionID'].nunique():,}")

# Filter by Date range (Sidebar)
transactions['TransactionDate'] = pd.to_datetime(transactions['TransactionDate'])
min_date = transactions['TransactionDate'].min().date()
max_date = transactions['TransactionDate'].max().date()

st.sidebar.subheader("Global Filters")
date_range = st.sidebar.date_input(
    "Select Transaction Period",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Filter dataset based on selected date
if len(date_range) == 2:
    start_dt, end_dt = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    filtered_trans = transactions[
        (transactions['TransactionDate'] >= start_dt) & 
        (transactions['TransactionDate'] <= end_dt + pd.Timedelta(days=1))
    ]
else:
    filtered_trans = transactions

# Regenerate button
if st.sidebar.button("🔄 Regenerate Raw Data"):
    with st.spinner("Generating fresh synthetic dataset..."):
        from generate_data import generate_retail_dataset
        generate_retail_dataset("data")
        st.cache_data.clear()
        st.rerun()

# =====================================================================
# 4. Main Page Header
# =====================================================================
st.markdown('<h1 class="main-title">RetailAnalytics360 🛍️</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">An analytical engine and forecasting platform for modern retail optimization.</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Executive Overview",
    "🎯 Customer Segmentation",
    "🔮 Sales Forecasting",
    "🛒 Market Basket Insights",
    "👤 Customer Deep-Dive"
])

# =====================================================================
# Tab 1: Executive Overview
# =====================================================================
with tab1:
    # 1. KPI Metrics
    total_sales = filtered_trans["TotalRevenue"].sum()
    unique_orders = filtered_trans["TransactionID"].nunique()
    avg_order_value = total_sales / unique_orders if unique_orders > 0 else 0
    total_units = filtered_trans["Quantity"].sum()
    avg_discount = filtered_trans["Discount"].mean() * 100
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Revenue", f"${total_sales:,.2f}")
    col2.metric("Total Orders", f"{unique_orders:,}")
    col3.metric("Avg Order Value", f"${avg_order_value:,.2f}")
    col4.metric("Total Units Sold", f"{total_units:,}")
    col5.metric("Avg Discount", f"{avg_discount:.1f}%")
    
    st.markdown("---")
    
    # 2. Charts Row 1: Sales Trends & Category Share
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        st.subheader("Monthly Sales Trend")
        monthly_sales = filtered_trans.resample("ME", on="TransactionDate")["TotalRevenue"].sum().reset_index()
        fig_trend = px.line(
            monthly_sales, 
            x="TransactionDate", 
            y="TotalRevenue",
            labels={"TotalRevenue": "Revenue ($)", "TransactionDate": "Month"},
            template="plotly_dark",
            color_discrete_sequence=["#6366f1"]
        )
        fig_trend.update_traces(line=dict(width=3), mode="lines+markers")
        fig_trend.update_layout(hovermode="x unified", margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_trend, use_container_width=True)
        
    with col_right:
        st.subheader("Sales Share by Category")
        cat_sales = filtered_trans.groupby("ProductCategory")["TotalRevenue"].sum().reset_index()
        fig_cat = px.pie(
            cat_sales, 
            values="TotalRevenue", 
            names="ProductCategory",
            hole=0.4,
            color_discrete_sequence=["#6366f1", "#a855f7", "#ec4899", "#f59e0b"],
            template="plotly_dark"
        )
        fig_cat.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=True)
        st.plotly_chart(fig_cat, use_container_width=True)
        
    st.markdown("---")
    
    # 3. Charts Row 2: Customer Demographics
    col_dem1, col_dem2, col_dem3 = st.columns([1, 1, 1])
    
    with col_dem1:
        st.subheader("Customer Age Distribution")
        fig_age = px.histogram(
            customers, 
            x="Age", 
            nbins=15,
            color_discrete_sequence=["#a855f7"],
            template="plotly_dark",
            opacity=0.8
        )
        fig_age.update_layout(margin=dict(t=10, b=10, l=10, r=10), yaxis_title="Count")
        st.plotly_chart(fig_age, use_container_width=True)
        
    with col_dem2:
        st.subheader("Gender Breakdown")
        gender_counts = customers["Gender"].value_counts().reset_index()
        fig_gen = px.bar(
            gender_counts, 
            x="Gender", 
            y="count",
            color="Gender",
            color_discrete_sequence=["#6366f1", "#ec4899", "#f59e0b"],
            template="plotly_dark"
        )
        fig_gen.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, yaxis_title="Count")
        st.plotly_chart(fig_gen, use_container_width=True)
        
    with col_dem3:
        st.subheader("Regional Sales Density")
        reg_sales = filtered_trans.merge(customers, on="CustomerID").groupby("Region")["TotalRevenue"].sum().reset_index()
        fig_reg = px.bar(
            reg_sales, 
            y="Region", 
            x="TotalRevenue",
            orientation="h",
            color_discrete_sequence=["#ec4899"],
            template="plotly_dark"
        )
        fig_reg.update_layout(margin=dict(t=10, b=10, l=10, r=10), xaxis_title="Revenue ($)")
        st.plotly_chart(fig_reg, use_container_width=True)

# =====================================================================
# Tab 2: Customer Segmentation
# =====================================================================
with tab2:
    st.subheader("RFM Customer Segmentation (K-Means Clustering)")
    st.markdown("""
    Using RFM Analysis (Recency, Frequency, Monetary) to cluster customers into 4 action-oriented cohorts. 
    Metrics are transformed logarithmically to scale out skewness, followed by standard normalization before K-Means clustering.
    """)
    
    # Merge customer segment labels back into RFM
    # Segment column is added in run_rfm_segmentation
    
    col_cluster_left, col_cluster_right = st.columns([3, 2])
    
    with col_cluster_left:
        st.markdown("**Interactive 3D Customer Segmentation Space**")
        # Visualizing RFM in 3D
        fig_3d = px.scatter_3d(
            rfm, 
            x="Recency", 
            y="Frequency", 
            z="Monetary",
            color="Segment",
            log_y=True,
            log_z=True,
            color_discrete_sequence=["#ec4899", "#6366f1", "#a855f7", "#f59e0b"],
            hover_data=["CustomerID"],
            opacity=0.8,
            height=600,
            template="plotly_dark"
        )
        fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        st.plotly_chart(fig_3d, use_container_width=True)
        
    with col_cluster_right:
        st.markdown("**Cohort Profile Metrics**")
        # Display cluster averages
        profile_table = rfm.groupby("Segment").agg(
            Count=("CustomerID", "count"),
            AvgRecency=("Recency", "mean"),
            AvgFrequency=("Frequency", "mean"),
            AvgMonetary=("Monetary", "mean")
        ).reset_index()
        
        # Format columns
        profile_table["AvgRecency"] = profile_table["AvgRecency"].map(lambda x: f"{x:.1f} days")
        profile_table["AvgFrequency"] = profile_table["AvgFrequency"].map(lambda x: f"{x:.1f} purchases")
        profile_table["AvgMonetary"] = profile_table["AvgMonetary"].map(lambda x: f"${x:,.2f}")
        profile_table["Count"] = profile_table["Count"].map(lambda x: f"{x:,} ({x/len(rfm)*100:.1f}%)")
        
        st.dataframe(profile_table, hide_index=True, use_container_width=True)
        
        # Explanations of segments and marketing action plans
        st.markdown("### 🎯 Strategic Cohort Playbooks")
        
        with st.expander("👑 VIP Spenders (Highly Engaged, Top Revenue)"):
            st.success("""
            - **Profile**: Recently purchased, extremely frequent buyers, highest monetary contributions.
            - **Strategy**: Loyalty programs, early access to new product rollouts, dedicated VIP support, and personal referral rewards.
            """)
            
        with st.expander("⚡ Loyal & Regular (Stable, Reliable Engagement)"):
            st.info("""
            - **Profile**: Medium-to-low recency, buy frequently, medium spend.
            - **Strategy**: Cross-selling and up-selling products based on category rules, bundling deals, and offering subscription incentives.
            """)
            
        with st.expander("🌱 New / Casual (Low Frequency, Low Spend)"):
            st.warning("""
            - **Profile**: Low recency (recently active), but low purchase count and spend.
            - **Strategy**: Welcome discounts, onboarding tutorial guides, survey triggers, and recommendations of popular catalog items.
            """)
            
        with st.expander("⚠️ At-Risk / Churned (Inactive, Churn Risk)"):
            st.error("""
            - **Profile**: Haven't purchased in a long time (high recency), low frequency, low monetary value.
            - **Strategy**: Re-engagement email flows, win-back discount coupons (e.g., 'We miss you, here is 20% off'), and survey requests to identify pain points.
            """)

# =====================================================================
# Tab 3: Sales Forecasting
# =====================================================================
with tab3:
    st.subheader("Weekly Revenue Forecasting (Random Forest Regressor)")
    st.markdown("""
    This model forecasts future weekly demand. It is trained on historical revenue using lag variables (past 4 weeks of sales), 
    4-week rolling averages, and sine/cosine calendar functions to model seasonal patterns.
    """)
    
    # 1. Model accuracy metrics
    mcol1, mcol2, mcol3 = st.columns(3)
    mcol1.metric("Model R² Score (Fit Quality)", f"{forecast_metrics['R2']:.3f}")
    mcol2.metric("Mean Absolute Error (MAE)", f"${forecast_metrics['MAE']:,.2f}")
    mcol3.metric("Root Mean Squared Error (RMSE)", f"${forecast_metrics['RMSE']:,.2f}")
    
    st.markdown("---")
    
    # 2. Forecasting Chart
    st.subheader("Demand Forecast: Next 4 Weeks")
    
    # Construct complete visual timeline
    hist_df = weekly_sales.copy()
    hist_df["Type"] = "Historical"
    
    # For predictions
    eval_df_copy = eval_df.copy()
    eval_df_copy["Type"] = "Test Set Prediction"
    
    # For future predictions
    future_df = forecast_df.rename(columns={"ForecastedRevenue": "Revenue"})
    future_df["Type"] = "Future Forecast (4 Weeks)"
    
    # Combine
    plot_forecast_df = pd.concat([hist_df, eval_df_copy, future_df], ignore_index=True)
    plot_forecast_df = plot_forecast_df.sort_values("Week")
    
    # Build beautiful plotly chart with distinct lines
    fig_fore = go.Figure()
    
    # Add historical line
    fig_fore.add_trace(go.Scatter(
        x=hist_df["Week"],
        y=hist_df["Revenue"],
        name="Historical Sales",
        line=dict(color="#6366f1", width=2.5)
    ))
    
    # Add evaluation line
    fig_fore.add_trace(go.Scatter(
        x=eval_df["Week"],
        y=eval_df["Revenue"],
        name="Actual Sales (Test Period)",
        line=dict(color="#64748b", width=2, dash="dash")
    ))
    fig_fore.add_trace(go.Scatter(
        x=eval_df["Week"],
        y=eval_df["Predicted"],
        name="Predicted Sales (Test Period)",
        line=dict(color="#ec4899", width=2.5)
    ))
    
    # Add future forecast line
    fig_fore.add_trace(go.Scatter(
        x=future_df["Week"],
        y=future_df["Revenue"],
        name="Future Sales Forecast",
        line=dict(color="#f59e0b", width=3)
    ))
    
    fig_fore.update_layout(
        template="plotly_dark",
        xaxis_title="Timeline",
        yaxis_title="Weekly Revenue ($)",
        margin=dict(t=20, b=20, l=20, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig_fore, use_container_width=True)
    
    st.markdown("---")
    
    # 3. Model Feature Importances
    col_feat, col_feat_desc = st.columns([3, 2])
    
    with col_feat:
        st.subheader("Forecasting Feature Importance")
        # Extract features and importances
        feature_cols = [
            "Revenue_Lag_1", "Revenue_Lag_2", "Revenue_Lag_3", "Revenue_Lag_4",
            "Rolling_Mean_4W", "Rolling_Std_4W",
            "Week_Sin", "Week_Cos", "Month", "Year"
        ]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        feat_imp_df = pd.DataFrame({
            "Feature": [feature_cols[i] for i in indices],
            "Importance": [importances[i] for i in indices]
        })
        
        fig_imp = px.bar(
            feat_imp_df,
            x="Importance",
            y="Feature",
            orientation="h",
            color="Importance",
            color_continuous_scale=px.colors.sequential.Agsunset,
            template="plotly_dark"
        )
        fig_imp.update_layout(margin=dict(t=10, b=10, l=10, r=10), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)
        
    with col_feat_desc:
        st.subheader("Model Insights")
        st.markdown("""
        - **Rolling Mean 4W / Lag 1**: Strongest predictors of weekly sales. E-commerce transaction volume exhibits short-term inertia where the best predictor of next week's sales is the recent average.
        - **Seasonality (Sin/Cos of Week)**: Evaluates seasonal trends, particularly useful for capturing holiday spikes (Q4) and mid-year slowdowns.
        - **Strategic Use**: This forecast helps the logistics team scale inventory and warehousing operations up to 4 weeks in advance, minimizing stockouts.
        """)

# =====================================================================
# Tab 4: Market Basket Analysis
# =====================================================================
with tab4:
    st.subheader("Market Basket Analysis (Product Association Rules)")
    st.markdown("""
    Analyzes which products or product categories are frequently bought together. This allows for smart bundling, cross-selling, 
    and personalized product recommendations.
    """)
    
    col_mba_l, col_mba_r = st.columns([3, 2])
    
    with col_mba_l:
        st.markdown("**Category Co-purchase Heatmap (Support Matrix)**")
        # Generate cross-tab categories co-purchase count
        cat_matrix = pd.crosstab(cat_rules["Antecedent"], cat_rules["Consequent"], values=cat_rules["Support"], aggfunc="mean").fillna(0)
        
        fig_heat = px.imshow(
            cat_matrix,
            labels=dict(x="Consequent Category", y="Antecedent Category", color="Support"),
            x=cat_matrix.columns,
            y=cat_matrix.index,
            color_continuous_scale="Viridis",
            template="plotly_dark"
        )
        fig_heat.update_layout(margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig_heat, use_container_width=True)
        
    with col_mba_r:
        st.markdown("**Product Recommendation Engine Simulator**")
        all_prods_list = sorted(list(set(prod_rules["Antecedent"].unique())))
        
        selected_prod = st.selectbox(
            "Select a product currently in the customer's cart:",
            options=all_prods_list
        )
        
        # Find recommendations
        recs = prod_rules[prod_rules["Antecedent"] == selected_prod].sort_values("Lift", ascending=False)
        
        if len(recs) > 0:
            st.markdown(f"**Top Recommended Add-ons for *'{selected_prod}'*:**")
            
            for idx, rec in recs.head(3).iterrows():
                st.info(f"💡 **{rec['Consequent']}** (Lift: **{rec['Lift']:.2f}** | Confidence: **{rec['Confidence']*100:.1f}%**)")
                st.markdown(f"""
                - **Lift of {rec['Lift']:.2f}** means the customer is **{rec['Lift']:.1f}x more likely** to buy a *{rec['Consequent']}* when they have *{selected_prod}* in their cart compared to buying it normally.
                - **Confidence of {rec['Confidence']*100:.1f}%** indicates that out of all transactions containing *{selected_prod}*, {rec['Confidence']*100:.1f}% of them also contained *{rec['Consequent']}*.
                """)
        else:
            st.write("No strong product association rules found for this product.")

# =====================================================================
# Tab 5: Customer Deep-Dive
# =====================================================================
with tab5:
    st.subheader("Customer Profile Explorer & Churn Analysis")
    st.markdown("Look up individual customer profiles, review their order history, and assess their churn risk indicators.")
    
    # Customer select box
    cust_list = sorted(rfm["CustomerID"].unique().tolist())
    selected_cust_id = st.selectbox("Search Customer ID:", options=cust_list)
    
    # Get Customer demographic details
    cust_profile = customers[customers["CustomerID"] == selected_cust_id].iloc[0]
    cust_rfm = rfm[rfm["CustomerID"] == selected_cust_id].iloc[0]
    cust_orders = filtered_trans[filtered_trans["CustomerID"] == selected_cust_id]
    
    col_c1, col_c2 = st.columns([1, 2])
    
    with col_c1:
        st.markdown("### Profile Card")
        # Format profile card
        st.markdown(f"""
        - **Customer ID**: `{cust_profile['CustomerID']}`
        - **Age**: {cust_profile['Age']} years
        - **Gender**: {cust_profile['Gender']}
        - **Region**: {cust_profile['Region']}
        - **Original Registration Profile**: {cust_profile['Profile']}
        - **Signup Date**: {cust_profile['JoinDate'].strftime('%Y-%m-%d')}
        """)
        
        # RFM metrics
        st.markdown("### RFM Scoreboard")
        st.metric("Recency", f"{cust_rfm['Recency']} days ago")
        st.metric("Frequency", f"{cust_rfm['Frequency']} orders")
        st.metric("Monetary (Lifetime Spend)", f"${cust_rfm['Monetary']:,.2f}")
        
        # Cluster categorization
        st.markdown("### Cohort Assignment")
        if cust_rfm['Segment'] == "VIP Spenders":
            st.success(f"👑 {cust_rfm['Segment']}")
        elif cust_rfm['Segment'] == "Loyal & Regular":
            st.info(f"⚡ {cust_rfm['Segment']}")
        elif cust_rfm['Segment'] == "New / Casual":
            st.warning(f"🌱 {cust_rfm['Segment']}")
        else:
            st.error(f"⚠️ {cust_rfm['Segment']}")
            
    with col_c2:
        st.markdown("### Transaction Order History")
        if len(cust_orders) > 0:
            # Aggregate order lines to order sums
            order_summary = cust_orders.groupby(["TransactionID", "TransactionDate", "PaymentMethod"]).agg(
                TotalQuantity=("Quantity", "sum"),
                RevenueSpent=("TotalRevenue", "sum"),
                AvgDiscount=("Discount", "mean"),
                UniqueProducts=("ProductName", "nunique")
            ).reset_index().sort_values("TransactionDate", ascending=False)
            
            # Format display
            order_summary["RevenueSpent"] = order_summary["RevenueSpent"].map(lambda x: f"${x:,.2f}")
            order_summary["AvgDiscount"] = order_summary["AvgDiscount"].map(lambda x: f"{x*100:.1f}%")
            
            st.dataframe(order_summary, use_container_width=True, hide_index=True)
            
            # Detailed products purchased
            st.markdown("### Detailed Items Purchased")
            item_display = cust_orders[["TransactionDate", "ProductName", "ProductCategory", "Quantity", "UnitPrice", "TotalRevenue"]].sort_values("TransactionDate", ascending=False)
            item_display["UnitPrice"] = item_display["UnitPrice"].map(lambda x: f"${x:.2f}")
            item_display["TotalRevenue"] = item_display["TotalRevenue"].map(lambda x: f"${x:.2f}")
            st.dataframe(item_display, use_container_width=True, hide_index=True)
            
        else:
            st.write("No transactions registered in this selected timeframe.")
            
        # Churn Risk Prediction (Custom rule-based logic from segment & recency)
        st.markdown("### 🔍 Churn Risk Assessment")
        
        recency = cust_rfm['Recency']
        if cust_rfm['Segment'] == "At-Risk / Churned" or recency > 180:
            st.error(f"🔴 **HIGH CHURN RISK** (Inactive for {recency} days). Customer has likely left.")
            st.markdown("""
            **Recommended Action Plan**:
            - Trigger win-back campaign with standard discount coupon.
            - Send exit feedback survey to investigate why they became inactive.
            """)
        elif recency > 60:
            st.warning(f"🟡 **MEDIUM CHURN RISK** (Inactive for {recency} days). Customer is slipping.")
            st.markdown("""
            **Recommended Action Plan**:
            - Deliver automated re-engagement email with customized suggestions based on co-purchase rules.
            - Provide a small milestone loyalty reward.
            """)
        else:
            st.success(f"🟢 **LOW CHURN RISK** (Active recently, last purchase {recency} days ago).")
            st.markdown("""
            **Recommended Action Plan**:
            - Maintain regular communication, present premium offers, and promote matching add-on accessories.
            """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; font-size: 0.85rem;'>RetailAnalytics360 Pipeline. Created by Antigravity Coding Assistant.</div>", unsafe_allow_html=True)
