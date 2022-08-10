# https://stackoverflow.com/questions/52759118/how-to-return-a-dataframe-from-sklearn-tfidf-vectorizer-within-pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


class DenseTfidfVectorizer(TfidfVectorizer):

    def transform(self, raw_documents):
        X = super().transform(raw_documents)
        return X.toarray()

    def fit_transform(self, raw_documents, y=None):
        X = super().fit_transform(raw_documents, y=y)
        return X.toarray()
