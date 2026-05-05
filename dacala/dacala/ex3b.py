### Fixing 3b

def ex3b():
    print('''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
import re
import string
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

    # (Run once if not downloaded)
    nltk.download('stopwords')

    # Create unstructured dataset
texts = [
        "Patient has a history of hypertension and diabetes . Prescribed medication X .",
        "Asthma diagnosis confirmed . Patient advised to use inhaler daily .",
        "Hypertension patient . Needs regular monitoring of blood pressure .",
        "Diabetes patient . Recommended diet and exercise .",
        "Patient diagnosed with hypertension . Medication Y prescribed ."
    ]

df = pd.DataFrame({"Medical_Texts": texts})

    # Text preprocessing function
def preprocess(text):
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\W+", " ", text)
    text_list = [
            word for word in text.split()
            if word not in stopwords.words("english")
            and word not in string.punctuation
        ]
    return " ".join(text_list)

df['Medical_Texts'] = df['Medical_Texts'].apply(preprocess)

    # TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=10)
x_tfidf = vectorizer.fit_transform(df['Medical_Texts']).toarray()

print("TF-IDF Matrix:\n", x_tfidf)

    # KMeans Clustering
kmeans = KMeans(n_clusters=2)
predictions = kmeans.fit_predict(x_tfidf)

print("Cluster Predictions:\n", predictions)

    # Generate WordCloud
all_texts = " ".join(df['Medical_Texts'])
wordcloud = WordCloud(width=800, height=400, background_color='black').generate(all_texts)

    # Display WordCloud
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()''')

ex3b()