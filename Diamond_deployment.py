#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pickle
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import streamlit as st


# In[9]:


loaded_model = pickle.load(open('Diamond_model.sav','rb'))


# In[10]:


loaded_model = pickle.load(open('Diamond_model.sav','rb'))
def ctr_function(input_data):
    input_data_as_numpy_array=np.array(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    return prediction
    
    
def main():
    st.title("Diamond price Prediction Model")
    carat=st.text_input("Enter the carat :")
    cut=st.text_input("Enter the cut  :")
    color=st.text_input("Enter the color :")
    clarity=st.text_input("Enter the clarity : ")
    depth=st.text_input("Enter depth : ")
    table=st.text_input("Enter the table : ")
    x=st.text_input("Enter the x : ")
    y=st.text_input("Enter the y :")
    z=st.text_input("Enter the z : ")
    
    diagnosis=""
    if st.button("Price for given Data is : "):
        diagnosis= ctr_function([carat,cut,color,clarity,depth,table,x,y,z])
    st.success(diagnosis)
    
    
if __name__=="__main__":
    main()
    


# In[ ]:




