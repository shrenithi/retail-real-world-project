import os
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_retail_dataset(output_dir="data", num_customers=1000, start_year=2024, end_year=2025):
    """
    Generates a realistic synthetic e-commerce sales dataset with structural patterns.
    - Customer Segments: VIP, Regular, Occasional, Inactive
    - Seasonality: Q4 Holiday surge, weekend spikes
    - Product Associations: Co-purchases (e.g., Laptop & Mouse)
    - Pricing: Consistent product pricing
    """
    print("Initializing dataset generation...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    random.seed(42)
    
    # ----------------------------------------------------
    # 1. Product Definitions
    # ----------------------------------------------------
    # Product Catalog grouped by categories with base prices
    products_catalog = {
        "Electronics": {
            "Laptop": 999.99,
            "Mouse": 25.50,
            "Keyboard": 49.99,
            "Headphones": 89.99,
            "Monitor": 199.99
        },
        "Home & Kitchen": {
            "Coffee Maker": 79.99,
            "Blender": 39.99,
            "Toaster": 29.99,
            "Vacuum Cleaner": 149.99,
            "Air Fryer": 119.99
        },
        "Apparel": {
            "T-Shirt": 19.99,
            "Hoodie": 45.00,
            "Jeans": 59.99,
            "Sneakers": 85.00,
            "Socks": 9.99
        },
        "Books": {
            "Fiction Novel": 14.99,
            "Sci-Fi Trilogy": 34.99,
            "Self-Help Book": 18.99,
            "Biography": 22.50,
            "Cookbook": 27.99
        }
    }
    
    # Flat lists of products for easy lookup
    all_products = []
    product_category_map = {}
    product_price_map = {}
    
    for category, items in products_catalog.items():
        for prod_name, price in items.items():
            all_products.append(prod_name)
            product_category_map[prod_name] = category
            product_price_map[prod_name] = price
            
    # Co-purchase probability definitions (for MBA)
    # If key is bought, high probability of buying value in same transaction
    co_purchase_rules = {
        "Laptop": ["Mouse", "Keyboard"],
        "Coffee Maker": ["Toaster"],
        "T-Shirt": ["Hoodie", "Socks"],
        "Fiction Novel": ["Sci-Fi Trilogy"]
    }
    
    # ----------------------------------------------------
    # 2. Customer Table Generation
    # ----------------------------------------------------
    print(f"Generating {num_customers} customers...")
    
    genders = ["Male", "Female", "Other"]
    gender_probs = [0.48, 0.48, 0.04]
    
    regions = ["North America", "Europe", "Asia-Pacific", "South America"]
    region_probs = [0.45, 0.30, 0.15, 0.10]
    
    customer_profiles = ["VIP", "Regular", "Occasional", "Inactive"]
    profile_probs = [0.10, 0.45, 0.30, 0.15]
    
    customers_data = []
    
    start_date = datetime(2023, 6, 1) # Customers sign up starting mid-2023
    
    for i in range(num_customers):
        cust_id = f"C{10000 + i}"
        
        # Age distribution: mean=36, std=12, bounded [18, 80]
        age = int(np.clip(np.random.normal(36, 12), 18, 80))
        gender = np.random.choice(genders, p=gender_probs)
        region = np.random.choice(regions, p=region_probs)
        profile = np.random.choice(customer_profiles, p=profile_probs)
        
        # Sign-up date
        days_offset = random.randint(0, 500)
        join_date = start_date + timedelta(days=days_offset)
        
        customers_data.append({
            "CustomerID": cust_id,
            "Age": age,
            "Gender": gender,
            "Region": region,
            "Profile": profile,
            "JoinDate": join_date.strftime("%Y-%m-%d")
        })
        
    df_customers = pd.DataFrame(customers_data)
    
    # ----------------------------------------------------
    # 3. Transactions Table Generation
    # ----------------------------------------------------
    print("Generating transaction records...")
    
    transactions_data = []
    trans_id_counter = 100001
    
    start_sim_date = datetime(start_year, 1, 1)
    end_sim_date = datetime(end_year, 12, 31)
    total_sim_days = (end_sim_date - start_sim_date).days
    
    payment_methods = ["Credit Card", "PayPal", "Debit Card", "Bank Transfer"]
    payment_probs = [0.55, 0.25, 0.15, 0.05]
    
    # Generate purchases customer by customer according to their profile
    for _, customer in df_customers.iterrows():
        cust_id = customer["CustomerID"]
        profile = customer["Profile"]
        join_dt = datetime.strptime(customer["JoinDate"], "%Y-%m-%d")
        
        # Determine purchase frequency and activity parameters based on profile
        if profile == "VIP":
            # VIPs buy frequently, spend more, and stay active
            num_purchases = random.randint(25, 55)
            churned = False
            avg_items = 3
        elif profile == "Regular":
            # Regulars buy moderately
            num_purchases = random.randint(10, 25)
            churned = random.random() < 0.15 # 15% chance to churn early
            avg_items = 2
        elif profile == "Occasional":
            # Occasionals buy rarely
            num_purchases = random.randint(3, 10)
            churned = random.random() < 0.40 # 40% chance to churn
            avg_items = 1
        else:  # Inactive
            # Inactive customers buy 1-2 times near their sign-up date and stop
            num_purchases = random.randint(1, 2)
            churned = True
            avg_items = 1
            
        # If churned, they stop buying after some percentage of the timeline
        active_days_ratio = random.uniform(0.1, 0.6) if churned else 1.0
        allowed_sim_days = int(total_sim_days * active_days_ratio)
        
        purchase_dates = []
        for _ in range(num_purchases):
            # Purchase must be after sign-up date
            days_after_join = random.randint(0, max(1, allowed_sim_days))
            purchase_date = join_dt + timedelta(days=days_after_join)
            
            # Bound date within simulation period
            if start_sim_date <= purchase_date <= end_sim_date:
                purchase_dates.append(purchase_date)
                
        # Sort purchases chronologically
        purchase_dates.sort()
        
        # Create transactions
        for p_date in purchase_dates:
            # Inject seasonal multipliers to transaction counts (surges Q4, weekends)
            # We do this by dropping some transactions or creating new ones.
            # Alternatively, we adjust the transaction details (e.g., volume or revenue).
            # Let's adjust volume.
            
            # Weekend multiplier
            is_weekend = p_date.weekday() >= 5
            weekend_mult = 1.3 if is_weekend else 1.0
            
            # Q4 Holiday multiplier (Nov-Dec)
            is_q4 = p_date.month in [11, 12]
            q4_mult = 1.8 if is_q4 else 1.0
            
            # Combine multipliers to determine number of items in this order
            volume_scale = weekend_mult * q4_mult
            num_items = int(np.clip(np.random.poisson(avg_items * volume_scale), 1, 6))
            
            payment = np.random.choice(payment_methods, p=payment_probs)
            
            # Populate order items
            order_items = []
            
            # Choose a base product
            first_product = random.choice(all_products)
            order_items.append(first_product)
            
            # Try to add other products, incorporating co-purchase rules
            for _ in range(num_items - 1):
                last_item = order_items[-1]
                if last_item in co_purchase_rules and random.random() < 0.60:
                    # Co-purchase triggered!
                    co_item = random.choice(co_purchase_rules[last_item])
                    if co_item not in order_items:
                        order_items.append(co_item)
                    else:
                        order_items.append(random.choice(all_products))
                else:
                    order_items.append(random.choice(all_products))
            
            # Save items as transaction rows
            order_id = f"O{trans_id_counter}"
            trans_id_counter += 1
            
            for prod in order_items:
                qty = int(np.random.choice([1, 2, 3, 4], p=[0.75, 0.15, 0.08, 0.02]))
                price = product_price_map[prod]
                
                # Apply discount (VIPs always get some discount; others occasionally)
                if profile == "VIP":
                    discount = round(random.choice([0.05, 0.10, 0.15, 0.20]), 2)
                else:
                    discount = round(random.choice([0.0, 0.0, 0.0, 0.05, 0.10]), 2)
                    
                total_rev = round(qty * price * (1.0 - discount), 2)
                
                transactions_data.append({
                    "TransactionID": order_id,
                    "CustomerID": cust_id,
                    "TransactionDate": p_date.strftime("%Y-%m-%d %H:%M:%S"),
                    "ProductID": f"P{100 + all_products.index(prod)}",
                    "ProductName": prod,
                    "ProductCategory": product_category_map[prod],
                    "Quantity": qty,
                    "UnitPrice": price,
                    "Discount": discount,
                    "TotalRevenue": total_rev,
                    "PaymentMethod": payment
                })
                
    df_transactions = pd.DataFrame(transactions_data)
    
    # Sort transactions by date
    df_transactions["TransactionDate"] = pd.to_datetime(df_transactions["TransactionDate"])
    df_transactions = df_transactions.sort_values("TransactionDate").reset_index(drop=True)
    df_transactions["TransactionDate"] = df_transactions["TransactionDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
    
    # ----------------------------------------------------
    # 4. Save to CSV files
    # ----------------------------------------------------
    customers_file = os.path.join(output_dir, "customers.csv")
    transactions_file = os.path.join(output_dir, "transactions.csv")
    
    df_customers.to_csv(customers_file, index=False)
    df_transactions.to_csv(transactions_file, index=False)
    
    print(f"Data generation complete!")
    print(f"Saved: {customers_file} ({df_customers.shape[0]} customers)")
    print(f"Saved: {transactions_file} ({df_transactions.shape[0]} rows of transaction items)")
    
    # Print some interesting high-level summaries
    total_sales = df_transactions["TotalRevenue"].sum()
    print(f"Total simulated sales revenue: ${total_sales:,.2f}")

if __name__ == "__main__":
    generate_retail_dataset()
