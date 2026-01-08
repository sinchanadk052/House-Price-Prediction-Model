import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics

st.title("🏠 House Price Prediction")
st.write("House Price Prediction using Machine Learning")

house_price_data = pd.read_csv("housing.csv")

st.subheader("Dataset Preview")
st.dataframe(house_price_data.head())

correlation = house_price_data.corr()

st.subheader("Correlation Heatmap")
fig1, ax1 = plt.subplots(figsize=(5,5))
sns.heatmap(correlation, cbar=True, fmt=".1f", annot=True, annot_kws={"size":8}, cmap="Blues", ax=ax1)
st.pyplot(fig1)

X = house_price_data.drop(["MEDV"], axis=1)
Y = house_price_data["MEDV"]

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2, shuffle=True
)

model = XGBRegressor()
model.fit(X_train, Y_train)

training_prediction = model.predict(X_train)
test_prediction = model.predict(X_test)

train_score = metrics.r2_score(Y_train, training_prediction)
test_score = metrics.r2_score(Y_test, test_prediction)

st.subheader("Model Accuracy")
st.write(f"Training R² Score: {train_score:.2f}")
st.write(f"Testing R² Score: {test_score:.2f}")

fig2, ax2 = plt.subplots()
ax2.scatter(Y_train, training_prediction)
ax2.set_xlabel("Actual Prices")
ax2.set_ylabel("Predicted Prices")
st.pyplot(fig2)

st.subheader("Predict House Price")

rm = st.number_input("Average number of rooms (RM)", value=6.0)
lstat = st.number_input("Lower status population (%) (LSTAT)", value=5.0)
ptratio = st.number_input("Pupil-teacher ratio (PTRATIO)", value=15.0)

if st.button("Predict Price"):
    input_data = np.array([rm, lstat, ptratio]).reshape(1, -1)
    prediction = model.predict(input_data)
    st.success(f"Predicted House Price: ${prediction[0]:.2f}")
