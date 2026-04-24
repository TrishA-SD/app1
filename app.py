import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import MinMaxScaler
model=pickle.load(open('gb_model.pkl','rb'))
scaler=MinMaxScaler()
st.title("Insurance Cost Prediction")
age=st.number_input('Age',min_value=1,max_value=100,value=20)
gender=st.selectbox('Gender',('Male','Female'))
bmi=st.number_input('BMI',min_value=20.0,max_value=100.0,value=30.0)
smoker=st.selectbox('Smoker',('Yes','No'))
children=st.number_input('No of Children',min_value=0,max_value=10,value=0)
region=st.selectbox('Region',('northwest','northeast','southwest','southeast'))
Smoker=1 if smoker=='Yes' else 0
region_dict={'southwest':0,'northwest':1,'northeast':2,'southeast':3}
Region=region_dict[region]
sex_male=1 if gender=='male' else 0 
sex_female=1 if gender=='female' else 0
input_features=pd.DataFrame(
    {'age':[age],
     'bmi':[bmi],
     'children':[children],	
     'Smoker':[Smoker],
     'sex_female':[sex_female],	
     'sex_male':[sex_male],
     'Region':[Region]}
)
input_features[['age','bmi']]=scaler.fit_transform(input_features[['age','bmi']])
features=np.array(input_features)
if st.button('Predict'):
    prediction = model.predict(features)
    output = round(prediction[0], 2)
    st.success(f'Predicted Charge: ${output}')