import pandas as pd

transactions = [
    {"date": "2026-05-01", "description": "Grocery Outlet Store",     "amount": 84.50,  "category": "Food"},
    {"date": "2026-05-01", "description": "Shell Gas Station",         "amount": 45.00,  "category": "Transportation"},
    {"date": "2026-05-02", "description": "Netflix Subscription",      "amount": 15.49,  "category": "Entertainment"},
    {"date": "2026-05-03", "description": "Organic Farmers Market",    "amount": 32.25,  "category": "Food"},
    {"date": "2026-05-05", "description": "Metropolitan Transit Pass", "amount": 75.00,  "category": "Transportation"},
    {"date": "2026-05-06", "description": "Local Coffee Roasters",     "amount": 6.75,   "category": "Food"},
    {"date": "2026-05-08", "description": "AMC Movie Theater tickets", "amount": 28.00,  "category": "Entertainment"},
    {"date": "2026-05-10", "description": "Downtown Auto Repair",      "amount": 240.00, "category": "Transportation"},
    {"date": "2026-05-11", "description": "Corner Bakery Cafe",        "amount": 18.90,  "category": "Food"},
    {"date": "2026-05-12", "description": "Spotify Premium",           "amount": 11.99,  "category": "Entertainment"},
    {"date": "2026-05-14", "description": "Whole Foods Market",        "amount": 112.30, "category": "Food"},
    {"date": "2026-05-15", "description": "Chevron Fuel",              "amount": 52.50,  "category": "Transportation"},
    {"date": "2026-05-17", "description": "City Concert Tickets",      "amount": 135.00, "category": "Entertainment"},
    {"date": "2026-05-18", "description": "Taco Truck Lunch",          "amount": 14.50,  "category": "Food"},
    {"date": "2026-05-20", "description": "Uber Ride Share",           "amount": 24.25,  "category": "Transportation"},
    {"date": "2026-05-22", "description": "Trader Joe's Groceries",    "amount": 95.10,  "category": "Food"},
    {"date": "2026-05-23", "description": "Steam Games Sale",          "amount": 42.50,  "category": "Entertainment"},
    {"date": "2026-05-25", "description": "Target Superstore",         "amount": 63.80,  "category": "Shopping"},
    {"date": "2026-05-26", "description": "Patreon Creators",          "amount": 10.00,  "category": "Entertainment"},
    {"date": "2026-05-27", "description": "Subway Sandwich",           "amount": 11.20,  "category": "Food"},
    {"date": "2026-05-29", "description": "Public Parking Garage",     "amount": 15.00,  "category": "Transportation"},
    {"date": "2026-05-30", "description": "Amazon Prime Order",        "amount": 34.99,  "category": "Shopping"},
]

df = pd.DataFrame(transactions)

# --- Q a: Food transactions ---
print("--- Food Transactions ---")
food_df = df[df["category"] == "Food"]
print(food_df)

# --- Q b: Transactions over $100 ---
print("\n--- Transactions Over $100 ---")
high_spend_df = df[df["amount"] > 100]
print(high_spend_df)

# --- Q c: Total spent per category ---
print("\n--- Spending Summary by Category ---")
summary = df.groupby("category").agg(
    total_spent       = ("amount", "sum"),
    average_spent     = ("amount", "mean"),
    transaction_count = ("amount", "count")
).reset_index()
print(summary)

# --- Q d: Most expensive transaction ---
print("\n--- Most Expensive Transaction ---")
top1 = df.sort_values("amount", ascending=False).head(1)
print(top1)

# --- Q e: is_large computed column ---
df["is_large"] = df["amount"] > 200
print("\n--- Full DataFrame with is_large column ---")
print(df)

# --- is_large filter (idiomatic boolean mask) ---
print("\n--- Large Transactions (is_large == True) ---")
print(df[df["is_large"]])   # No == True needed

# --- Schema and statistics ---
print("\n--- dtypes ---")
print(df.dtypes)
print("\n--- describe() ---")
print(df.describe())