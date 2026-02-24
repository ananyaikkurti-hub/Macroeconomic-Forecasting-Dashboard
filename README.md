# Macroeconomic-Forecasting-Dashboard
Macroeconomic forecasting app built with Streamlit, implementing lag feature engineering, Ridge regression, and TimeSeriesSplit validation to predict future GDP growth.


[![Macroeconomic Forecasting Dashboard App](https://img.shields.io/badge/Streamlit-Live%20App-red)](https://macroeconomic-forecasting-dashboard-ixzunoaxp5shisevxb3txq.streamlit.app/)


-🚀 Project Overview
-
This project builds a predictive model to estimate future GDP growth based on macroeconomic indicators such as inflation, interest rates, and unemployment.

The app allows users to:

-Select a country

-Choose a historical or future year

-View predicted GDP growth

-Visualize historical trends

-See model performance metrics

-🧠 Machine Learning Approach
-
-Feature Engineering

-Computed GDP growth using percentage change

-Created lag feature (GDP_growth_lag1)

-Removed infinite and missing values

-Clipped extreme growth values to stabilize training

-Model
-

-Ridge Regression

-Standardized inputs using StandardScaler

-Recursive forecasting for multi-year future predictions

-Validation Strategy

-Time-Series Cross Validation (TimeSeriesSplit)

-Avoided data leakage

-Evaluated using:
-
-R²

-MAE

-RMSE
