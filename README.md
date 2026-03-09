# 🌍 Air Quality Index (AQI) Prediction  

**Repository:** [https://github.com/Anurag0115/aqi-prediction](https://github.com/Anurag0115/aqi-prediction)  
**Live App:** [Deploy on Streamlit Community Cloud](https://share.streamlit.io)

## 📌 Project Overview  
Air pollution is a critical environmental issue, and predicting air quality is essential for public health and policymaking.  
This project leverages **Machine Learning (ML) techniques** to forecast **Air Quality Index (AQI)** based on key environmental pollutants.  
The models implemented include **Linear Regression, Random Forest, and XGBoost**, with **XGBoost** emerging as the best-performing model.  

---

## 📂 Project Resources  
🔹 **Dataset (CSV File)**: [Download city_day.csv](https://github.com/SunnyRao07/Air-Quality-Index-AQI-Prediction/blob/main/city_day.csv)  
🔹 **Dataset (Kaggle Link)**: [View on Kaggle](https://www.kaggle.com/code/anjusunilkumar/air-quality-index-prediction?select=city_day.csv)  
🔹 **Project Code (.ipynb)**: [View Jupyter Notebook](https://github.com/SunnyRao07/Air-Quality-Index-AQI-Prediction/blob/main/Code.ipynb)  
🔹 **Presentation (PPTX File)**: [Download Project Report](https://github.com/SunnyRao07/Air-Quality-Index-AQI-Prediction/blob/main/Report.pptx)  

---

## 📊 Dataset Overview  
The dataset used for this project contains **29,531 records** and **16 features**, including:  
- **Pollutants**: PM2.5, PM10, NO₂, SO₂, CO, O₃, Benzene, etc.  
- **Date & City**: Identifying the location and time of recording.  
- **AQI (Target Variable)**: Measures pollution severity and categorizes it into buckets (Good, Moderate, Poor, etc.).  

---

## 🛠 Data Preprocessing  
✔️ **Handling Missing Values**:  
- Removed rows with missing AQI values.  
- Imputed missing pollutant values using **mean imputation**.  

✔️ **Feature Scaling**:  
- Used **StandardScaler** to standardize numerical features (zero mean, unit variance).  

✔️ **Feature Selection**:  
- Analyzed **feature correlation** using a **heatmap** to determine the most influential pollutants.  
- **PM2.5** (0.65) and **CO** (0.68) showed the highest correlation with AQI.  

---

## 🤖 Models Used  
The project explores different regression models to predict AQI:  

### 1️⃣ **Linear Regression**  
🔹 Simple and interpretable but struggles with complex patterns.  

### 2️⃣ **Random Forest Regressor**  
🔹 Uses multiple decision trees for better accuracy and handles non-linearity.  

### 3️⃣ **XGBoost Regressor** (🏆 Best Model)  
✅ More accurate, faster, and better at handling missing data than Random Forest.  
✅ Captures complex relationships effectively with boosting techniques.  
✅ Reduces overfitting and performs well on large datasets.  

---

## 📈 Model Evaluation  
To measure performance, we used the following metrics:  

📌 **Mean Absolute Error (MAE)** – Lower values indicate better accuracy.  
📌 **Root Mean Squared Error (RMSE)** – Penalizes large prediction errors.  
📌 **R² Score (Coefficient of Determination)** – Measures variance explained by the model.  

📌 **Model Comparison**:  
| Model | MAE | RMSE | R² Score |
|--------|------|------|---------|
| **Linear Regression** | High | High | Low |
| **Random Forest** | Moderate | Moderate | Moderate |
| **XGBoost** 🏆 | Low | Low | High |

---

## 🔧 Hyperparameter Tuning  
To improve the **XGBoost** model, we optimized the following parameters using **GridSearchCV**:  
- `n_estimators`: [50, 100, 200]  
- `learning_rate`: [0.01, 0.1, 0.2]  
- `max_depth`: [3, 5, 7]  

✅ **Best Configuration:**  
`learning_rate = 0.2, max_depth = 5, n_estimators = 100`  

The final model was saved using `joblib` for future predictions.  

---

## 📌 Results & Predictions  
- The trained **XGBoost Regressor** effectively predicts AQI values for unseen data.  
- A **line plot** was used to visualize **Actual vs. Predicted AQI**, confirming the model's accuracy.  

---

## 🚀 Future Scope  
🔹 Improve accuracy using **Deep Learning (LSTMs, CNNs)**.  
🔹 Integrate real-time AQI data & meteorological features.  
🔹 Develop **AI-powered dashboards & mobile apps** for real-time AQI tracking.  

---

## 🏆 Individual Contribution  
This was a group project, but I **contributed to the entire project**, including:  
✔️ Data Preprocessing & Cleaning  
✔️ Model Implementation & Evaluation  
✔️ Hyperparameter Tuning & Optimization  
✔️ Model Interpretation & Documentation  

---
