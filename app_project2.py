import streamlit as st 
import sklearn
import joblib
import numpy as np 
model=joblib.load("bank_churn_model.pkl")
scaler=joblib.load("scaler.pkl")

st.title("bank customer chunk predictor")
st.write("enter the customer details to check whether they are likely to leave the bunk")
col1,col2=st.columns(2)
with col1:
    credit_score=st.number_input("credit score",min_value=300,max_value=850,value=600)
    country=st.selectbox("country",["france","germany","spain"])
    gender=st.selectbox('gender',['Male','Female'])
    age=st.slider('age',18,100,42)
    tenure=st.slider("tenure(years)",0,10,3)
with col2:
    balance=st.number_input("enter the balance in account",min_value=0.0,value=60000.0) 
    num_products=st.selectbox("products number",[1,2,3,4],index=0)
    credit_card=st.selectbox("has credit card",[0,1]) 
    active_mem=st.selectbox("is active member",[0,1])
    salary=st.number_input("your salary",min_value=0.0,value=50000.0)
gender_val=1 if gender=="Male"  else 0
country_map={"germany":1,"france":0,"spain":2}
country_value=country_map[country]  
if st.button("predict churn") :
   input_data = [[credit_score, country_value, gender_val, age, 
               tenure, balance, num_products, 
               credit_card, active_mem, salary]]
    
   input_scaled=scaler.transform(input_data)   
   prob = model.predict_proba(input_scaled)[:, 1][0] 
   st.subheader(f"Churn Probability: {prob:.2%}")
   if prob >= 0.38:
        st.error("⚠️ High Risk! Customer is likely to leave.")
   else:
        st.success("✅ Low Risk. Customer is likely to stay.")
        
   



                  
 