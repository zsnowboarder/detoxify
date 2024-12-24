#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#pip install detoxify


# In[9]:


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

# Create an instance of the Detoxify model
model = Detoxify('original')

# init dict
scores = []

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
    


# In[10]:


# get the data from internet
texts = bs.find_all('div', class_='c-list__item__text')
web_text = []

for text in texts:
    sy = text.find('p', class_='c-list__item__description')
    if sy:
        web_text.append({'text':sy.get_text()})


# # load the data. use this when no Internet data
# data = { 'text': [ "You are a terrible person and I hate you.", "Have a great day!", "I can't believe you did that, you idiot." ] }
# df = pd.DataFrame(data)
# 

# In[13]:

st.title("Threat and Sentiment Detection")
new_text = st.text_area("Enter your text for analysis", value="I am going to get a gun and shoot people at all schools.")
df_new_text = pd.DataFrame({"text":[new_text]})
st.write("This demo retrieves news articles from CTV News and analyses the content. Your text will be added to the dataset for analysis.")
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

# In[ ]:




