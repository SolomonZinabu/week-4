# Rossmann Pharmaceuticals: Sales Forecasting and Data Exploration

### Author: Solomon Zinabu

---

## **Introduction**

Rossmann Pharmaceuticals is a large retail chain with multiple stores across various cities. Managers in individual stores have traditionally relied on personal experience to forecast sales. However, the finance team has identified the need for a more data-driven approach to predict sales six weeks ahead of time.

The objective of this analysis is to explore customer purchasing behavior and forecast sales using a variety of factors such as promotions, competition, holidays, and seasonality. This report covers the exploratory data analysis (EDA) phase of the project, which will form the foundation for machine learning and deep learning models.

---

## **Project Structure**

This repository contains the following files:

- **notebooks/**: Contains the Jupyter notebooks for EDA, feature engineering, and visualizations.
- **scripts/**: Contains modular Python scripts for data processing and visualizations.
  - `data_processing.py`: Handles data cleaning and feature engineering.
  - `visualizations.py`: Contains functions to generate visualizations such as sales distribution, promo effect, and competition analysis.
- **data/**: Contains the dataset files used for the analysis.
  - `train.csv`: Historical sales data including features like store, date, customers, open status, etc.
  - `store.csv`: Additional store information including store type, assortment level, and competition details.
  - `test.csv`: Test set without the sales column (for prediction purposes).

---

## **Business Need**

Rossmann Pharmaceuticals' finance team seeks to forecast sales across various stores six weeks ahead, allowing the company to plan resources and marketing strategies more efficiently. The primary objective is to:
- Explore customer purchasing behavior and identify key trends.
- Predict daily sales across stores based on various features like promotions, holidays, and competition.
- Serve these predictions in a real-time dashboard for the finance team.

---

## **Data Overview**

The dataset consists of sales data from 2013 to 2015 and includes the following files:

- **train.csv**: Contains daily sales figures for each store along with features like store type, promotion status, state holidays, etc.
- **store.csv**: Contains details about each store, including distance to competitors, assortment type, and promotional information.

### Key Columns:
- **Store**: Unique identifier for each store.
- **Sales**: Daily turnover for a given store.
- **Customers**: Number of customers for that day.
- **Promo**: Whether the store was running a promotion that day.
- **StateHoliday**: Whether the day was a state holiday.
- **CompetitionDistance**: Distance to the nearest competitor store.
- **Promo2**: Whether the store is part of a continuous promotion campaign.

---

## **Exploratory Data Analysis (EDA)**

The EDA focused on understanding the relationships between the key features and how they impact sales across different stores.

### **1. Sales Distribution**
We analyzed the distribution of sales across all stores. The data showed that the majority of stores have sales concentrated in the range of €2,000 to €10,000 per day, with some outliers.

- **Key Finding**: Most stores have consistent sales, but there are a few high-performing stores with significantly higher sales.

![Sales Distribution](sales_distribution_plot.png)

### **2. Promotion Impact on Sales**
We examined how promotions affect sales. Promotions play a crucial role in attracting more customers and boosting sales, as evident from the data.

- **Key Finding**: Stores running promotions see a significant increase in sales compared to non-promo days.

![Promo Effect on Sales](promo_effect_plot.png)

### **3. Store Type and Sales**
Stores are categorized into different types (`a`, `b`, `c`, `d`), each representing a specific business model. The analysis showed that certain store types consistently outperform others.

- **Key Finding**: Store types `a` and `b` have the highest sales, indicating that they are possibly located in more lucrative areas or cater to higher customer demand.

![Store Type vs Sales](store_type_vs_sales_plot.png)

### **4. Competition Distance and Sales**
The distance to the nearest competitor store is an important factor in determining sales. We analyzed how this distance affects sales figures.

- **Key Finding**: Stores located farther from competitors generally have higher sales, although the effect is not always linear.

![Competition Effect on Sales](competition_effect_plot.png)

### **5. Seasonal and Holiday Trends**
We also explored how sales fluctuate during public holidays and specific seasons (e.g., Christmas, Easter). Holidays generally cause temporary dips in sales, but stores tend to recover quickly with higher-than-usual sales afterward.

---

## **Data Cleaning and Feature Engineering**

To prepare the data for machine learning, several data cleaning and feature engineering steps were performed:

### **Data Cleaning**
- Missing values in `CompetitionDistance`, `Promo2SinceYear`, and similar columns were handled by filling them with median values or zero.
- Outliers in sales data were identified but not removed, as they represent high-performing stores.

### **Feature Engineering**
- **CompetitionOpen**: We calculated how many months the nearest competitor has been open.
- **Promo2Open**: We calculated how many weeks each store has participated in continuous promotions.
- **Holiday Features**: Extracted whether a particular day was before or after a holiday.

---

## **Key Insights**

- **Promotions Matter**: Stores running promotions see a clear boost in sales, especially during holiday seasons.
- **Store Type Performance**: Certain store types (especially type `a`) outperform others in terms of sales.
- **Competition Influence**: Distance to competitors plays a role, but it's more pronounced for some stores than others.
- **Seasonality**: Sales spike around public holidays, especially Christmas and Easter, suggesting a strong seasonal trend.

---

## **Recommendations**

1. **Maximize Promo Effectiveness**:
   - Focus promotional campaigns on high-performing store types (`a` and `b`) and during key holidays like Christmas and Easter.
   - Leverage promo data to refine targeted marketing strategies for stores located in high-competition areas.

2. **Store-Specific Strategies**:
   - For stores located near competitors, consider increasing promotional frequency or offering loyalty programs to retain customers.
   - Invest in stores located in areas without nearby competitors, as they tend to have higher sales.

3. **Holiday Planning**:
   - Allocate more resources and promotional budget to stores in the lead-up to major holidays, as this is when customer activity peaks.
   - Consider adjusting store hours or offerings during post-holiday periods to maximize the sales spike.

---
## Task 2: Prediction of Store Sales (Random Forest)

In this task, we use a Random Forest model to predict store sales based on historical data. The model considers factors such as promotions, holidays, and competition, which are crucial to predicting sales.

### Steps
1. **Data Preprocessing**:
    - Convert non-numeric features to numeric.
    - Handle missing values.
    - Scale the data using `StandardScaler`.

2. **Feature Engineering**: Extract additional features such as the number of days to holidays, day of the month, and other temporal features.

3. **Model Training**:
    - Split the data into training and testing sets.
    - Train a Random Forest regressor using the training set.
    - Evaluate model performance using RMSE (Root Mean Square Error).

```python
# Random Forest Model Training
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluation
rmse_rf = mean_squared_error(y_test, y_pred_rf, squared=False)
print(f"Random Forest RMSE: {rmse_rf}")


# LSTM Model Training
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Build LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Predictions
y_pred_lstm = model.predict(X_test)

# Evaluation
rmse_lstm = mean_squared_error(y_test, y_pred_lstm, squared=False)
print(f"LSTM RMSE: {rmse_lstm}")

## **Next Steps: Prediction and Deployment**

With the EDA completed, the next steps involve building machine learning and deep learning models to forecast sales. This will include:
- Preprocessing data for model input.
- Building tree-based regression models (e.g., Random Forest) and Recurrent Neural Networks (LSTM) for time series forecasting.
- Deploying the trained models through a REST API for real-time sales predictions.


## **Contact**

For any questions or further discussion, feel free to reach out:
- **Email**: solomonzinabu1988@gmail.com

