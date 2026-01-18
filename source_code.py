# ===============================
#  Install required libraries
# ===============================
!pip install gradio seaborn scikit-learn

# ===============================
# STEP 1: Import libraries
# ===============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import gradio as gr

plt.style.use("seaborn-v0_8")

# ===============================
# STEP 2: Upload CSV files
# ===============================
from google.colab import files
uploaded = files.upload()  # Upload: olist_orders_dataset.csv, olist_order_items_dataset.csv, olist_customers_dataset.csv

# ===============================
# STEP 3: Read CSV files
# ===============================
orders = pd.read_csv("olist_orders_dataset.csv")
items = pd.read_csv("olist_order_items_dataset.csv")
customers = pd.read_csv("olist_customers_dataset.csv")

# ===============================
# STEP 4: Data Cleaning & Preparation
# ===============================
# Convert purchase timestamp to datetime
orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])

# Keep only delivered orders
orders = orders[orders['order_status'] == 'delivered']

# Merge datasets
sales_data = orders.merge(items, on='order_id', how='inner')
sales_data = sales_data.merge(customers, on='customer_id', how='inner')

# Total sales per order
sales_data['total_sales'] = sales_data['price'] + sales_data['freight_value']

# Extract year-month for aggregation
sales_data['order_month'] = sales_data['order_purchase_timestamp'].dt.to_period('M')
monthly_sales = sales_data.groupby('order_month')['total_sales'].sum().reset_index()
monthly_sales['order_month'] = monthly_sales['order_month'].astype(str)

# Add month index for ML
monthly_sales['month_index'] = np.arange(len(monthly_sales))

# ===============================
# STEP 5: Data Visualization
# ===============================
# 1. Monthly sales trend
plt.figure(figsize=(12,6))
plt.plot(monthly_sales['order_month'], monthly_sales['total_sales'], marker='o')
plt.xticks(rotation=45)
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.show()

# 2. Heatmap: Year vs Month
monthly_sales['year'] = monthly_sales['order_month'].str[:4]
monthly_sales['month'] = monthly_sales['order_month'].str[5:7].astype(int)
pivot = monthly_sales.pivot("month", "year", "total_sales")
plt.figure(figsize=(10,6))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu")
plt.title("Monthly Sales Heatmap (Year vs Month)")
plt.ylabel("Month")
plt.show()

# 3. Rolling average trend
monthly_sales['total_sales_MA3'] = monthly_sales['total_sales'].rolling(3).mean()
plt.figure(figsize=(12,6))
plt.plot(monthly_sales['order_month'], monthly_sales['total_sales'], label="Original")
plt.plot(monthly_sales['order_month'], monthly_sales['total_sales_MA3'], label="3-Month MA", color="red")
plt.xticks(rotation=45)
plt.title("Monthly Sales with 3-Month Rolling Average")
plt.legend()
plt.show()

# 4. Top 5 states by total sales
top_states = sales_data.groupby('customer_state')['total_sales'].sum().sort_values(ascending=False).head(5)
plt.figure(figsize=(8,5))
sns.barplot(x=top_states.index, y=top_states.values, palette="rocket")
plt.title("Top 5 States by Total Sales")
plt.ylabel("Total Sales")
plt.show()

# ===============================
# STEP 6: Machine Learning - Forecasting
# ===============================
X = monthly_sales[['month_index']]
y = monthly_sales['total_sales']

# Train-Test split (time series, no shuffle)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

# Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred, model_name):
    print(f"--- {model_name} Evaluation ---")
    print("MAE:", round(mean_absolute_error(y_true, y_pred),2))
    print("RMSE:", round(np.sqrt(mean_squared_error(y_true, y_pred)),2))
    print("R2 Score:", round(r2_score(y_true, y_pred),2))
    print("\n")

evaluate_model(y_test, y_pred_lr, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest Regressor")

# Actual vs Predicted Graph
plt.figure(figsize=(12,6))
plt.plot(monthly_sales['order_month'][-len(y_test):], y_test, label="Actual Sales", marker='o')
plt.plot(monthly_sales['order_month'][-len(y_test):], y_pred_lr, label="LR Prediction", linestyle='--')
plt.plot(monthly_sales['order_month'][-len(y_test):], y_pred_rf, label="RF Prediction", linestyle='-.')
plt.xticks(rotation=45)
plt.title("Actual vs Predicted Sales (Test Set)")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.legend()
plt.show()

# ===============================
# STEP 7: Future Forecast (Next 12 Months)
# ===============================
last_index = monthly_sales['month_index'].max()
future_index = np.arange(last_index + 1, last_index + 13).reshape(-1,1)

future_pred_lr = lr_model.predict(future_index)
future_pred_rf = rf_model.predict(future_index)

# Future months labels
last_month_str = monthly_sales['order_month'].iloc[-1]
last_month = pd.to_datetime(last_month_str + "-01")
future_months = pd.date_range(last_month + pd.offsets.MonthBegin(1), periods=12, freq='MS').strftime("%Y-%m").tolist()

# Forecast graph
plt.figure(figsize=(14,6))
plt.plot(monthly_sales['order_month'], monthly_sales['total_sales'], label="Actual Sales", marker='o')
plt.plot(future_months, future_pred_lr, label="LR Forecast", linestyle='--', marker='x')
plt.plot(future_months, future_pred_rf, label="RF Forecast", linestyle='-.', marker='s')
plt.xticks(rotation=45)
plt.title("Sales Forecast for Next 12 Months")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.legend()
plt.show()

# ===============================
# STEP 8: Gradio UI - Local
# ===============================
def predict_sales_ui(month_index):
    month_index = np.array([[monthly_sales['month_index'].max() + month_index]])
    lr_pred = lr_model.predict(month_index)[0]
    rf_pred = rf_model.predict(month_index)[0]
    return f"Linear Regression Forecast: ₹{round(lr_pred,2)}\nRandom Forest Forecast: ₹{round(rf_pred,2)}"

inputs = gr.Slider(minimum=1, maximum=12, step=1, label="Month Index (1 = Next Month)")
outputs = gr.Textbox(label="Predicted Sales", lines=2)

interface = gr.Interface(
    fn=predict_sales_ui,
    inputs=inputs,
    outputs=outputs,
    title="🛒 OLIST Sales Forecast Predictor",
    description="Enter future month index to get sales prediction (Local session only)."
)

interface.launch(share=False)
