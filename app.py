import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer

# Load Dataset
df = pd.read_csv("dataset.csv")

# Select relevant columns
features = ['Experience Level', 'Industry', 'Location']
target = 'Salary'

# Drop rows with missing salary values
df = df.dropna(subset=[target])

# Encode categorical variables
column_transformer = ColumnTransformer([
    ('encoder', OneHotEncoder(handle_unknown='ignore'), features)
], remainder='passthrough')

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Transform categorical features
X_train_transformed = column_transformer.fit_transform(X_train)
X_test_transformed = column_transformer.transform(X_test)

# Train the model
model = LinearRegression()
model.fit(X_train_transformed, y_train)

# Streamlit App
st.title("💼 Salary Prediction App")
st.write("Enter job details to predict salary")

# User inputs
experience_level = st.selectbox("Experience Level", df['Experience Level'].unique())
industry = st.selectbox("Industry", df['Industry'].unique())
location = st.selectbox("Location", df['Location'].unique())

# Predict salary
if st.button("Predict Salary"):
    user_input = pd.DataFrame([[experience_level, industry, location]], columns=features)
    user_input_transformed = column_transformer.transform(user_input)
    predicted_salary = model.predict(user_input_transformed)[0]
    st.success(f"💰 Estimated Salary: ${predicted_salary:,.2f}")
