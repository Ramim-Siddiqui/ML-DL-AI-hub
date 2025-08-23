# 🔮 Dynamic Predictor

A versatile **machine learning prediction tool** that dynamically trains multiple models (regression, ensemble, and deep learning) on a given dataset, evaluates them, and selects the **best-performing model** automatically.  
Supports **daily & monthly predictions** with customizable inputs, and outputs results to CSV files for easy analysis.  

---

## 🚀 Features
- 📂 Supports **CSV & XLSX** datasets.  
- ✅ Automatic data validation (max 15 columns).  
- 🔧 Handles **date columns**, extracts year/month/day features.  
- 🔠 Automatic **label encoding** for categorical/string features.  
- 📉 Drops invalid rows (negative/NaN values).  
- 🧠 Trains and evaluates multiple models:  
  - **Classical ML**: Linear Regression, Ridge, Lasso, ElasticNet  
  - **Tree-based**: Decision Trees, Random Forests, Extra Trees, Gradient Boosting, XGBoost  
  - **Probabilistic**: Bayesian Ridge  
  - **Polynomial Regression**  
  - **Time Series**: ARIMA, Exponential Smoothing (ETS)  
  - **Deep Learning**: LSTM (Keras)  
- 📊 Calculates **R² scores** for all models and saves them (`r2scores.csv`).  
- 🏆 Automatically selects the **best model**.  
- 🔮 Supports **Daily & Monthly Predictions** with output CSVs:  
  - `predictions.csv` (daily / combinations)  
  - `monthly_predictions.csv` (aggregated by month)  

---

## 🛠 Tech Stack
- **Python**
- **Pandas, NumPy** – Data preprocessing  
- **Scikit-learn** – Machine learning models  
- **XGBoost** – Gradient boosting  
- **Statsmodels** – ARIMA, ETS models  
- **Keras / TensorFlow** – LSTM deep learning model  

---

## 📌 How It Works
1. Load dataset (CSV/XLSX).  
2. Choose the **target column**.  
3. (Optional) Specify **date column** → extracts `year`, `month`, `day`.  
4. (Optional) Drop irrelevant columns.  
5. Automatic **label encoding** for categorical data.  
6. Train multiple models → Compare using **R² score**.  
7. Select **best model** and perform predictions.  
8. Save results to CSV.  

---

## ▶️ Usage
Clone the repository and run:

```bash
python predictor.py
Follow the interactive prompts:

Enter dataset path (.csv or .xlsx).

Select target column.

(Optional) Specify date column.

Choose prediction type: daily or monthly.

Get predictions saved to CSV files.
```
### 📂 Example Output
```bash
R² Scores → r2scores.csv

Daily Predictions → predictions.csv

Monthly Predictions (averaged by month) → monthly_predictions.csv
```
### 📊 Sample Workflow
```bash
Please enter the path for the dataset (CSV or XLSX format): dataset.csv
Please enter the column name you want to predict: sales
Please enter the column name that contains dates: date
Do you want to drop any more columns? (yes/no): no
How would you like to do predictions? (daily/monthly): monthly
```
### 👨‍💻 Author
Developed by Ramim Siddiqui
BSCS (AI) Graduate | Passionate about AI & Web Development

### 📜 License
This project is open-source under the MIT License.
