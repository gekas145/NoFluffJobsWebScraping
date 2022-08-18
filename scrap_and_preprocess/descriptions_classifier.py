import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


df = pd.read_csv('../job_postings.csv')


def get_interest(x):
    if x in ['Business Analysis', 'Big Data', 'Business Intelligence', 'AI']:
        return 1
    return 0


df['interest'] = df.apply(lambda x: get_interest(x['main_category']), axis=1)

corpus = df['description_preprocessed'].to_list()

model = Pipeline([('vectorize', TfidfVectorizer()),
                  ('svd', TruncatedSVD(random_state=123)),
                  ('clf', KNeighborsClassifier())])

y = df['interest'].to_list()
X_train, X_test, y_train, y_test = train_test_split(corpus, y, test_size=0.2, random_state=123, stratify=y)

params = {
    'clf__n_neighbors': [10, 15, 20],
    'clf__p': [1, 2],
    'svd__n_components': [20, 30, 40]
}

cross_val = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
search = GridSearchCV(model, params,
                      scoring='f1',
                      cv=cross_val,
                      verbose=5)
search.fit(X_train, y_train)
model = search.best_estimator_


print('Train')
print(f'Accuracy: {accuracy_score(model.predict(X_train), y_train)}, f1: {f1_score(model.predict(X_train), y_train)}')
print('Test')
print(f'Accuracy: {accuracy_score(model.predict(X_test), y_test)}, f1: {f1_score(model.predict(X_test), y_test)}')

print(classification_report(model.predict(X_test), y_test,
                            target_names=['not interested', 'interested']))


model.fit(corpus, y)
print(classification_report(model.predict(corpus), y,
                            target_names=['not interested', 'interested']))

print(search.best_params_)

joblib.dump(model, '../kneighbors.pkl')

