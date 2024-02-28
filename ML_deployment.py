#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:





# In[3]:


import streamlit as st


# In[10]:


loaded_model = pickle.load(open('trained_model11.sav','rb'))


# In[11]:


def ctr_function(input_data):
    input_data_as_numpy_array=np.array(input_data)
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)
    prediction=loaded_model.predict(input_data_reshape)
    return prediction
    
    
def main():
    st.title("Blue Stone's CTR prediction for next Campaign")
    clicks=st.text_input("Enter the Clicks :")
    impressions=st.text_input("Enter the impressions  :")
    campaign_budget_usd=st.text_input("Enter the campaign budget in usd :")
    no_of_days=st.text_input("Enter the no of days : ")
    ext_service_id=st.text_input("Enter the service id : ")
    media_cost_usd=st.text_input("Enter the media cost usd : ")
    advertiser_id=st.text_input("Enter the advertiser id : ")
    network_id=st.text_input("Enter the network id :")
    approved_budget=st.text_input("Enter the approved budget : ")
    channel_id=st.text_input("Enter the channel id :")
    
    diagnosis=""
    if st.button("CTR for given Data is : "):
        diagnosis= ctr_function([clicks ,impressions,campaign_budget_usd,no_of_days,ext_service_id,
                                        media_cost_usd,advertiser_id,network_id,approved_budget,channel_id])
    st.success(diagnosis)
    
    
if __name__=="__main__":
    main()
    


# In[ ]:




