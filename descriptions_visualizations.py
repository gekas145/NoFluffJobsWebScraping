import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import itertools

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


df = pd.read_csv('job_postings.csv')
corpus = df['description_preprocessed']
corpus = [descr.split(' ') for descr in corpus]

descrs_length = [len(descr) for descr in corpus]

plt.hist(descrs_length, bins=50)
plt.title('Distribution of descriptions length')
plt.xlabel('Number of non stopwords')
plt.ylabel('Number of occurences')
plt.show()



words = [' '.join(descr) for descr in corpus]

words = " ".join(words) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()








