import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords

# uncomment if those are not downloaded
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('stopwords')


df = pd.read_csv('../job_postings.csv')
df = df[df.description.notna()]
corpus = df['description'].to_list()


def preprocess_corpus(text_corpus):
    sw = stopwords.words("english")
    tokenizer = RegexpTokenizer(r'\w+')
    lemmatizer = WordNetLemmatizer()

    content_preprocessed = []
    for descr in text_corpus:
        word_list = tokenizer.tokenize(descr)
        descr_preprocessed = [w.lower() for w in word_list if w.lower() not in sw]
        descr_preprocessed = [lemmatizer.lemmatize(w) for w in descr_preprocessed]
        content_preprocessed.append(' '.join(descr_preprocessed))
    return content_preprocessed


preprocessed_text = preprocess_corpus(corpus)
df['description_preprocessed'] = preprocessed_text
df.drop(['posting_href'], axis=1, inplace=True)
df.to_csv('job_postings.csv', index=False, encoding='utf-8-sig')
