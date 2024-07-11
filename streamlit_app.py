#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install streamlit


# In[ ]:


import pandas as pd
import streamlit as st
from gmail import EmailSearch  # Adjust the import path based on your file structure

@st.cache_resource
def init_emailsearch():
    connection_string = (
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=localhost\\SQLEXPRESS;"
        "Database=ENRON_Emails;"
        "UID='Your Uid';"
        "PWD='Password';"
    )

    COHERE_API_KEY = 'Your_cohere_apikey'

    emailsearch = EmailSearch(connection_string=connection_string, COHERE_API_KEY=COHERE_API_KEY, thr=0.65)
    emailsearch.get_engine()
    emailsearch.Create_database(csv_file_path='small_enron.csv')
    
    return emailsearch

emailsearch = init_emailsearch()

st.title("Email Search App")

query = st.text_input("Enter your query:")

if query:
    result_ids = emailsearch.search(query)
    
    if result_ids and len(result_ids):
        result_df = emailsearch.get_mails(result_ids)
        result_df = result_df.drop(columns=['EmailId'])
        st.write("Search Results:")
        st.dataframe(result_df.head())
    else:
        st.write("No results found for the given query.")


# In[ ]:


#streamlit run Mail.py


# In[ ]:





# In[ ]:




