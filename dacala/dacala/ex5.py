def ex5():
    print('''
5 - NLP for patients insights from a medical texts
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import nltk
import re
import spacy
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize

from transformers import pipeline
from wordcloud import WordCloud

nltk.download('stopwords') # Add this line to download stopwords

stop_words=set(stopwords.words('english'))

texts = [
    "Patient has a history of hypertension and diabetes . Prescribed medication X .",
    "Asthma diagnosis confirmed . Patient advised to use inhaler daily .",
    "Hypertension patient . Needs regular monitoring of blood pressure .",
    "Diabetes patient . Recommended diet and exercise .",
    "Patient diagnosed with hypertension . Medication Y prescribed ."
]

df = pd.DataFrame({"Medical_Texts": texts})
df.head()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\W+", " ", text)
    text_list = [words for words in text.split()
                 if words not in stopwords.words("english")
                 and words not in string.punctuation]
    return " ".join(text_list)

df['Medical_Texts'] = df['Medical_Texts'].apply(preprocess)
df.head()

NER = spacy.load("en_core_web_sm")

def extract_entities(text):
    doc = NER(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities if entities else []

df['Named_Entities'] = df['Medical_Texts'].apply(extract_entities)
df.head()

sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def sentiment(x):
    return sentiment_pipeline(x)[0]

df['Sentiment'] = df['Medical_Texts'].apply(sentiment)
df['Sentiment'].head()

all_texts = " ".join(df['Medical_Texts'])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_texts)

plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')''')

ex5()