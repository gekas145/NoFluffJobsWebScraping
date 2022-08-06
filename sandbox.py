import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
import matplotlib.pyplot as plt

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

df = pd.read_csv('job_postings.csv')
# descrs = df['description'].to_list()

df = df[df.description.notna()]
print(df.shape)

corpus = df['description']
# vectorizer = CountVectorizer(stop_words='english')
# X = vectorizer.fit_transform(corpus)
# print(vectorizer.get_feature_names_out())
# print(df['description'][0])
# print(vectorizer.get_stop_words())


tokenizer = RegexpTokenizer(r'\w+')
lemmatizer = WordNetLemmatizer()

words_lemmatized = []
descrs_length = []
for descr in corpus:
    word_list = tokenizer.tokenize(descr)
    descrs_length.append(len(word_list))
    descr_lemmatized = [lemmatizer.lemmatize(w).lower() for w in word_list]
    words_lemmatized += descr_lemmatized

df = pd.DataFrame(words_lemmatized, columns=['words'])
df = df.groupby('words').agg({'words': 'count'})
df.index = df.index.set_names(['w'])
df.reset_index(inplace=True)
df.sort_values(['words'], inplace=True, ascending=False)
print(df.head(50))

plt.hist(descrs_length, bins=30)
plt.show()