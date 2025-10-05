# 🚚 Amazon Delivery Time Prediction (Streamlit App)

A web application that predicts **Amazon-like delivery time (in minutes)** based on factors like weather, traffic, agent rating, distance, and more.  
Built with **Streamlit** and powered by an **XGBoost Regression Model** trained on real-world delivery data.

---

## 🌐 Live Demo
👉 [View on Streamlit Cloud](https://amazon-delivery-time-estimator.streamlit.app/)

---

## 💡 Features
- Interactive web app built using Streamlit  
- Predicts delivery time using 16 key features  
- Includes intelligent feature engineering:
  - `traffic_weather`, `vehicle_traffic`, `vehicle_category`
- Uses trained **XGBoost pipeline** (`R² ~ 0.82`, `MAE = 17.5`, `RMSE = 22.47`)  
- Simple, fast, and easy to deploy  

---

## 🧠 Model Details
- **Algorithm:** XGBoost Regressor  
- **Best Parameters:**
  - `learning_rate=0.01`
  - `n_estimators=400`
  - `max_depth=10`
  - `subsample=0.6`
  - `reg_lambda=1`, `reg_alpha=0.1`
- **Performance (Test Set):**
  - **R²:** 0.811  
  - **MAE:** 17.50  
  - **RMSE:** 22.47  

---

## ⚙️ Tech Stack
- Python 3.12  
- Streamlit  
- Scikit-learn  
- XGBoost  
- Pandas, NumPy  
- Joblib  

---


---

## 🧩 Features Used in Model
| Feature | Description |
|----------|-------------|
| `Agent_Age` | Delivery agent’s age |
| `Agent_Rating` | Agent’s average customer rating |
| `Weather` | Weather conditions |
| `Traffic` | Current traffic level |
| `Vehicle` | Type of delivery vehicle |
| `Area` | Delivery area type |
| `Category` | Product category |
| `distance_km` | Delivery distance |
| `order_dayofweek` | Day of the week (0=Mon, 6=Sun) |
| `order_month` | Month of the order |
| `is_weekend` | 1 if weekend else 0 |
| `order_hour` | Hour of order placement |
| `is_rush_hour` | 1 if rush hour (8-10 or 18-21) |
| `traffic_weather`, `vehicle_traffic`, `vehicle_category` | Engineered features |

---

## 🖥️ How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/Amazon-Delivery-Time-Prediction.git
cd Amazon-Delivery-Time-Prediction

# 2. Create and activate virtual environment (optional)
python -m venv venv
venv\Scripts\activate  # on Windows
# source venv/bin/activate  # on Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py

