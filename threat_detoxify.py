#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install detoxify


# In[23]:


# Import the Detoxify library
import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
from detoxify import Detoxify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

#get data from ctvnews
url = "https://www.ctvnews.ca/"
response = requests.get(url)
bs = BeautifulSoup(response.content, "html.parser")



# init dict
scores = []


def load_detoxify():
    # Create an instance of the Detoxify model
    return Detoxify('original')
model = load_detoxify()


def get_data():
    #get data from ctvnews
    url = "https://www.ctvnews.ca/"
    response = requests.get(url)
    bs = BeautifulSoup(response.content, "html.parser")
    web_text = []
    for article in bs.find_all('article', class_='c-stack b-simple-list-custom__item'):
        heading = article.find('h3', class_='c-heading')
        if heading:
            web_text.append({'text':heading.get_text()})
    return web_text[:9]
    

def analyze(text):
    results = model.predict(text)
    scores.append(results)
    return results

def map_sum_threat(value):
    if value < 0.01:
        return "None detected"
    elif value >= 0.01 and value < 0.4:
        return "Low"
    elif value >= 0.4 and value < 0.6:
        return "Medium"
    else:
        return "High"
    


# In[24]:

# # load the data. use this when no Internet data
# data = { 'text': [ "You are a terrible person and I hate you.", "Have a great day!", "I can't believe you did that, you idiot." ] }
# df = pd.DataFrame(data)
# 

# In[28]:

web_text = get_data()

st.title("Threat and Sentiment Detection")
new_text = st.text_area("Enter your text for analysis", value="I am going to get a gun and shoot people at all schools.")
df_new_text = pd.DataFrame({"text":[new_text]})
st.write("This demo retrieves the latest news articles from CTV News and analyses the content. Your text will be added to the dataset for analysis. This concept can be used to monitor threats for specific user account using an API.")
df = pd.DataFrame(web_text)

# create a st button
if st.button("Analyze"):
    df = pd.concat([df, df_new_text], ignore_index=True) #add new text for analysis
    df["toxicity"] = df["text"].apply(analyze)
    df_scores = pd.DataFrame(scores)
    # VaderSentiment Analysis
    array_dicts = []
    array_pred = []
    
    for row_index, row in df.iterrows():
        analyzer = SentimentIntensityAnalyzer()
        vs = analyzer.polarity_scores(row["text"])
        array_dicts.append(vs)
        if vs["compound"] < 0:
            array_pred.append("Negative")
        elif vs["compound"] > 0:
            array_pred.append("Positive")
        else:
            array_pred.append("Neutral")
    df_vs = pd.DataFrame(array_dicts)
    df_vs["vs_score"] = array_pred
    
    df_scores["sum_threat"] = df_scores[["toxicity", "severe_toxicity", "threat"]].sum(axis=1)
    df_scores["Detected Threat"] = df_scores["sum_threat"].apply(map_sum_threat)
    
    df = pd.concat([df, df_scores, df_vs], axis=1)
    df = df[["text", "vs_score", "Detected Threat", "sum_threat"]]
    df = df.sort_values(by="sum_threat", ascending=False)
    st.table(df)



# In[30]:





# In[ ]:




