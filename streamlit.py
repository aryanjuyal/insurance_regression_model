import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()

with open('regressor.pkl','rb') as f:
    model_regressor=pickle.load(f)



st.title("insurance regression model")


   

st.header("Enter your details")

age = st.number_input("Age", min_value=18, max_value=100, value=30)
bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0)
children = st.number_input("Number of Children", min_value=0, max_value=5, value=0)
sex = st.selectbox("Sex", ["male", "female"])
smoker = st.selectbox("Smoker", ["yes", "no"])
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    
sex = 1 if sex == "male" else 0
smoker = 1 if smoker == "yes" else 0

region_northeast = 1 if region == "northeast" else 0
region_northwest = 1 if region == "northwest" else 0
region_southeast = 1 if region == "southeast" else 0
region_southwest = 1 if region == "southwest" else 0




input_data = np.array([[age, bmi, children, sex, smoker,region_northeast, region_northwest, region_southeast, region_southwest ]])
input_data = sc.transform(input_data)
if st.button("predict"):
   prediction = model_regressor.predict(input_data)
   st.success(f"prediction:{prediction}")