import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("clean_macro_data.csv")

# -----------------------------
# Feature Engineering
# -----------------------------
df["GDP_growth"] = df.groupby("Country Name")["GDP"].pct_change()
df["GDP_growth_lag1"] = df.groupby("Country Name")["GDP_growth"].shift(1)

df.replace([np.inf, -np.inf], np.nan, inplace=True)
df["GDP_growth"] = df["GDP_growth"].clip(-0.3, 0.3)

df = df.dropna()

# -----------------------------
# Model Training
# -----------------------------
features = ["Inflation", "Interest_value", "Unemployment", "GDP_growth_lag1"]

X = df[features]
y = df["GDP_growth"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = Ridge(alpha=1.0)
model.fit(X_scaled, y)

# -----------------------------
# Time Series Cross Validation
# -----------------------------
tscv = TimeSeriesSplit(n_splits=5)

r2_scores = []
mae_scores = []
rmse_scores = []

for train_index, test_index in tscv.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    temp_model = Ridge(alpha=1.0)
    temp_model.fit(X_train, y_train)

    y_pred = temp_model.predict(X_test)

    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))

r2 = np.mean(r2_scores)
mae = np.mean(mae_scores)
rmse = np.mean(rmse_scores)


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 Mini Macroeconomic Forecasting Dashboard")

country = st.selectbox("Select Country", df["Country Name"].unique())
max_data_year = int(df["Year"].max())

year = st.slider(
    "Select Year",
    int(df["Year"].min()),
    max_data_year + 5,
    max_data_year
)

country_data = df[df["Country Name"] == country].sort_values("Year")

if country_data.empty:
    st.warning("No data available for selected country.")

else:

    if year <= max_data_year:
        selected = country_data[country_data["Year"] == year]

        if not selected.empty:
            X_input = selected[features]
            X_input_scaled = scaler.transform(X_input)
            prediction = model.predict(X_input_scaled)[0]
        else:
            st.warning("No data available for selected year.")
            st.stop()

    else:
        # Recursive Forecast
        last_row = country_data.iloc[-1]
        prediction = last_row["GDP_growth"]

        for future_year in range(max_data_year + 1, year + 1):
            input_data = [[
                last_row["Inflation"],
                last_row["Interest_value"],
                last_row["Unemployment"],
                prediction
            ]]
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)[0]

    # -----------------------------
    # Display Prediction
    # -----------------------------
    st.subheader(f"Predicted GDP Growth for {country} ({year})")

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prediction * 100,
        title={'text': "GDP Growth (%)"},
        gauge={'axis': {'range': [-10, 10]}}
    ))

    st.plotly_chart(fig)

    st.subheader("Historical GDP Growth Trend")
    st.line_chart(country_data.set_index("Year")["GDP_growth"] * 100)


# -----------------------------
# Display Model Performance
# -----------------------------
st.subheader("📈 Time-Series Model Performance")

st.write(f"Average R²: {r2:.4f}")
st.write(f"Average MAE: {mae:.4f}")
st.write(f"Average RMSE: {rmse:.4f}")