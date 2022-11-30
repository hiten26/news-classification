import numpy as np  # linear algebra
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

print("Reading data from files: ")
train_data = pd.read_csv("./dbpedia_csv/train.csv", header=None, names=['class', 'name', 'description'])
test_data = pd.read_csv("./dbpedia_csv/test.csv", header=None, names=['class', 'name', 'description'])

class_dict = {
    1: 'Company',
    2: 'EducationalInstitution',
    3: 'Artist',
    4: 'Athlete',
    5: 'OfficeHolder',
    6: 'MeanOfTransportation',
    7: 'Building',
    8: 'NaturalPlace',
    9: 'Village',
    10: 'Animal',
    11: 'Plant',
    12: 'Album',
    13: 'Film',
    14: 'WrittenWork'
}

print("mapping class to dictionary")
train_data['class'] = train_data['class'].map(class_dict)
test_data['class'] = test_data['class'].map(class_dict)


def multiclass_logloss(actual, predicted, eps=1e-15):
    """Multi class version of Logarithmic Loss metric.
    :param actual: Array containing the actual target classes
    :param predicted: Matrix with class predictions, one probability per class
    """
    # Convert 'actual' to a binary array if it's not already:
    if len(actual.shape) == 1:
        actual2 = np.zeros((actual.shape[0], predicted.shape[1]))
        for i, val in enumerate(actual):
            actual2[i, val] = 1
        actual = actual2

    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota


#stop_words = stopwords.words('english')

print("encoding labels")
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(train_data["class"])

print("dividing data to train test")
xtrain, xvalid, ytrain, yvalid = train_test_split(train_data["description"], y,
                                                  stratify=y,
                                                  random_state=42,
                                                  test_size=0.1, shuffle=True)

# Always start with these features. They work (almost) everytime!
print("Creating tf-idf vectorizer")
tfv = TfidfVectorizer(min_df=3, max_features=None,
                      strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                      ngram_range=(1, 3), use_idf=1, smooth_idf=1, sublinear_tf=1,
                      stop_words='english')
print("Transforming to tfidf: train and test both")
# Fitting TF-IDF to both training and test sets (semi-supervised learning)
tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv = tfv.transform(xtrain)
xvalid_tfv = tfv.transform(xvalid)

print("Fitting LR model")
# Fitting a simple Logistic Regression on TFIDF
clf = LogisticRegression(C=1.0)
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict_proba(xvalid_tfv)

print("logloss: %0.3f " % multiclass_logloss(yvalid, predictions))

article = "World Cup Brazil Breaks Through With a Flourish; Ronaldo Scores in Fifth World Cup"
article_tfv = tfv.transform([article])
k = clf.predict(article_tfv)[0]
class_dict[k]

print("Saving models and transformer")
pickle.dump(tfv, open('transform.pkl', 'wb'))
pickle.dump(clf, open('logistic_regression_model.pkl', 'wb'))
