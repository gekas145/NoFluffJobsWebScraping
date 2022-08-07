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
from sklearn.manifold import TSNE
import itertools

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


df = pd.read_csv('job_postings.csv')
df = df[df.description.notna()]
corpus = df['description'].to_list()


def preprocess_corpus(text_corpus):
    sw = stopwords.words("english")
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    content_preprocessed = []
    # descrs_length = []
    for descr in text_corpus:
        word_list = tokenizer.tokenize(descr)
        # descrs_length.append(len(word_list))
        descr_preprocessed = [w.lower() for w in word_list if w.lower() not in sw]
        descr_preprocessed = [lemmatizer.lemmatize(w) for w in descr_preprocessed]
        content_preprocessed.append(descr_preprocessed)
    return content_preprocessed


# df = pd.DataFrame(words_lemmatized, columns=['words'])
# df = df.groupby('words').agg({'words': 'count'})
# df.index = df.index.set_names(['w'])
# df.reset_index(inplace=True)
# df.sort_values(['words'], inplace=True, ascending=False)
# print(df.head(50))

# plt.hist(descrs_length, bins=30)
# plt.show()

preprocessed_text = preprocess_corpus(corpus)
preprocessed_text = list(itertools.chain.from_iterable(preprocessed_text))

words = " ".join(preprocessed_text) + " "

wordcloud = WordCloud(width=800, height=800,
                      background_color='white',
                      min_font_size=10).generate(words)

plt.figure(figsize=(8, 8), facecolor=None)
plt.imshow(wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()

# plt.hist(descrs_length, bins=30)
# plt.show()
