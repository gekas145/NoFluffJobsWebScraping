import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from dense_tfidf_vectorizer import DenseTfidfVectorizer
import joblib

df = pd.read_csv('job_postings.csv')


def get_interest(x):
    if x in ['Business Analysis', 'Big Data', 'Business Intelligence', 'AI']:
        return 1
    return 0


df['interest'] = df.apply(lambda x: get_interest(x['main_category']), axis=1)

# corpus_interesting = df[df.interest == 'interesting']['description_preprocessed']
# corpus_not_interesting = df[df.interest != 'interesting']['description_preprocessed']
corpus = df['description'].to_list()

# descrs_length = [len(descr) for descr in corpus]
#
# plt.hist(descrs_length, bins=50)
# plt.title('Distribution of descriptions length')
# plt.xlabel('Number of non stopwords')
# plt.ylabel('Number of occurences')
# plt.show()


# words = [' '.join(descr) for descr in corpus]
#
# words = " ".join(words) + " "

# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white',
#                       min_font_size=10).generate(' '.join(corpus_interesting))
#
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()

# wordcloud = WordCloud(width=800, height=800,
#                       background_color='white',
#                       min_font_size=10).generate(' '.join(corpus_not_interesting))
#
# plt.figure(figsize=(8, 8), facecolor=None)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.tight_layout(pad=0)
# plt.show()


# vectorizer = DenseTfidfVectorizer()
# vectorizer = CountVectorizer()
# X = vectorizer.fit_transform(corpus)

# pca = PCA(n_components=20)

model = Pipeline([('vectorize', DenseTfidfVectorizer()),
                  ('pca', PCA(n_components=20, random_state=123)),
                  ('clf', KNeighborsClassifier())])

y = df['interest'].to_list()
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=42, stratify=y)

# model.fit(X_train, y_train)

params = {
    'clf__n_neighbors': [3, 4, 5],
    'clf__weights': ['uniform'],
    'clf__metric': ['minkowski'],
    'clf__p': [1, 2]
}

# params = {
#     'clf__n_estimators': [50, 100, 200, 300],
#     'clf__max_features': ['sqrt', 10],
#     'clf__bootstrap': [True, False],
#     'clf__class_weight': ['balanced', 'balanced_subsample', {0: 1, 1: 3}],
#     'clf__min_samples_split': [2, 5, 10, 20]
# }

cross_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
search = RandomizedSearchCV(model, params,
                            scoring='f1',
                            cv=cross_val,
                            random_state=123, verbose=5)
search.fit(X_train, y_train)
model = search.best_estimator_

# sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['interest'])
# plt.show()



print('Train')
print(f'Accuracy: {accuracy_score(model.predict(X_train), y_train)}, f1: {f1_score(model.predict(X_train), y_train)}')
print('Test')
print(f'Accuracy: {accuracy_score(model.predict(X_test), y_test)}, f1: {f1_score(model.predict(X_test), y_test)}')

print(classification_report(model.predict(X_test), y_test,
                            target_names=['not interested', 'interested']))

# print(search.best_params_)
# model.fit(corpus, y)
# joblib.dump(model, 'kneighbors.pkl')
# model = joblib.load('kneighbors.pkl')
# print(classification_report(model.predict(X_test), y_test,
#                             target_names=['not interested', 'interested']))

