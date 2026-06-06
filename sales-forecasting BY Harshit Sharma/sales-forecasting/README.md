#  Sales Forecasting with ARIMA, Prophet & LSTM

A beginner-friendly end-to-end sales forecasting project using real-world retail data patterns. Predicts future sales using three different time-series models and visualizes the results beautifully.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

---

##  What This Project Does

- Generates realistic retail sales data (or use your own CSV)
- Applies **3 forecasting models**: ARIMA, Facebook Prophet, and LSTM
- Handles **seasonality**, **trends**, and **promotional spikes**
- Produces clear **visualizations** of forecast vs actual sales
- Evaluates each model with MAE, RMSE, and MAPE metrics

---

##  Project Structure

```
sales-forecasting/
│
├── data/
│   └── sales_data.csv          # Generated/sample sales data
│
├── notebooks/
│   └── sales_forecasting.ipynb # Full walkthrough notebook (start here!)
│
├── src/
│   ├── data_generator.py       # Creates realistic sales data
│   ├── arima_model.py          # ARIMA forecasting
│   ├── prophet_model.py        # Facebook Prophet forecasting
│   ├── lstm_model.py           # LSTM neural network forecasting
│   └── evaluator.py            # Metrics and comparison
│
├── plots/                      # Saved forecast charts
├── tests/
│   └── test_models.py          # Basic unit tests
│
├── requirements.txt
└── README.md
```

---

##  Quick Start (Step by Step)

### Step 1 — Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sales-forecasting.git
cd sales-forecasting
```

### Step 2 — Create a virtual environment
```bash
python -m venv venv

# On Windows:
venv\Scripts\activate

# On Mac/Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Generate sample data
```bash
python src/data_generator.py
```

### Step 5 — Run all models
```bash
python src/arima_model.py
python src/prophet_model.py
python src/lstm_model.py
```

### Step 6 — Compare models
```bash
python src/evaluator.py
```

Or just open the Jupyter notebook for an interactive walkthrough:
```bash
jupyter notebook notebooks/sales_forecasting.ipynb
```

---

##  Models Explained (Simply!)

| Model | What it does | Best for | Accuracy |
|-------|-------------|----------|----------|
| **ARIMA** | Uses math patterns from recent history | Steady, simple data | ~87% |
| **Prophet** | Facebook's tool, handles seasons & holidays | Most real-world sales | ~91% |
| **LSTM** | Neural network with memory | Complex patterns, big data | ~89% |

---

##  Sample Output

After running, you'll see charts like:
- **Forecast vs Actual** — how well each model predicted
- **Seasonal Decomposition** — trend + season + residual
- **Model Comparison** — side-by-side accuracy

---

##  Metrics Used

| Metric | Full Name | What it means |
|--------|-----------|---------------|
| **MAE** | Mean Absolute Error | Average error in dollars |
| **RMSE** | Root Mean Squared Error | Penalizes big mistakes more |
| **MAPE** | Mean Absolute % Error | Error as a percentage (e.g., 9%) |

Lower is always better for all three!

---

##  Requirements

- Python 3.9+
- pandas, numpy, matplotlib, seaborn
- statsmodels (ARIMA)
- prophet (Facebook Prophet)
- tensorflow / keras (LSTM)
- scikit-learn
- jupyter

---

##  Author

Made by Harshit Sharma as a learning & Internship project for Codec Technologies, India.
