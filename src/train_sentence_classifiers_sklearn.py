"""
Trains different classifiers available in sklearn for sentence classification
"""

import pandas as pd
import numpy as np
import pickle

import itertools
import xgboost as xgb
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score

import seaborn as sns
import matplotlib.pyplot as plt

EMOTION_DICT = {'ang': 0, 'hap': 1, 'sad': 2, 'fea': 3, 'sur': 4, 'neu': 5}
EMO_KEYS = list(['ang', 'hap', 'sad', 'fea', 'sur', 'neu'])


def train_tfidf_vectors(df):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2',
                            encoding='latin-1', ngram_range=(1, 2),
                            stop_words='english')
    return tfidf.fit_transform(df.transcription).toarray()


def create_train_test_split(features, labels, test_size=0.2):
    return train_test_split(features, labels, test_size=0.20)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    # plt.figure(figsize=(8,8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def one_hot_encoder(true_labels, num_records, num_classes):
    temp = np.array(true_labels[:num_records])
    true_labels = np.zeros((num_records, num_classes))
    true_labels[np.arange(num_records), temp] = 1
    return true_labels


def display_results(y_test, pred_probs, cm=True):
    pred = np.argmax(pred_probs, axis=-1)
    one_hot_true = one_hot_encoder(y_test, len(pred), len(EMOTION_DICT))
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    if cm:
        plot_confusion_matrix(confusion_matrix(y_test, pred), classes=EMO_KEYS)


def model_random_forest_classifier(x_train, y_train, x_test, y_test):
    rf_classifier = RandomForestClassifier(n_estimators=6000,
                                           min_samples_split=25)
    rf_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = rf_classifier.predict_proba(x_test)

    # Results
    display_results(y_test, pred_probs)

    with open('pred_probas/text_rf_classifier.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)


def model_xgb_classifier(x_train, y_train, x_test, y_test):
    xgb_classifier = xgb.XGBClassifier(max_depth=7, learning_rate=0.008,
                                       objective='multi:softprob',
                                       n_estimators=600, sub_sample=0.8,
                                       num_class=len(EMOTION_DICT),
                                       booster='gbtree', n_jobs=4)
    xgb_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = xgb_classifier.predict_proba(x_test)

    # Results
    display_results(y_test, pred_probs)

    with open('pred_probas/text_xgb_classifier.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)


def model_svc_classifier(x_train, y_train, x_test, y_test):
    svc_classifier = LinearSVC()

    svc_classifier.fit(x_train, y_train)

    # Predict
    pred = svc_classifier.predict(x_test)

    # Results
    one_hot_true = one_hot_encoder(y_test, len(pred), len(EMOTION_DICT))
    print('Test Set Accuracy =  {0:.3f}'.format(accuracy_score(y_test, pred)))
    print('Test Set F-score =  {0:.3f}'.format(f1_score(y_test, pred, average='macro')))
    print('Test Set Precision =  {0:.3f}'.format(precision_score(y_test, pred, average='macro')))
    print('Test Set Recall =  {0:.3f}'.format(recall_score(y_test, pred, average='macro')))
    plot_confusion_matrix(confusion_matrix(y_test, pred), classes=EMOTION_DICT.keys())

    with open('pred_probas/text_svc_classifier_model.pkl', 'wb') as f:
        pickle.dump(svc_classifier, f)


def model_multinomial_naive_bayes_classifier(x_train, y_train, x_test, y_test):
    mnb_classifier = MultinomialNB()

    mnb_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = mnb_classifier.predict_proba(x_test)

    # Results
    display_results(y_test, pred_probs)

    with open('pred_probas/text_mnb_classifier.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)


def model_mlp_classifier(x_train, y_train, x_test, y_test):
    mlp_classifier = MLPClassifier(hidden_layer_sizes=(500, ),
                                   activation='relu', solver='adam',
                                   alpha=0.0001, batch_size='auto',
                                   learning_rate='adaptive',
                                   learning_rate_init=0.01, power_t=0.5,
                                   max_iter=1000, shuffle=True,
                                   random_state=None, tol=0.0001,
                                   verbose=False, warm_start=True,
                                   momentum=0.8, nesterovs_momentum=True,
                                   early_stopping=False,
                                   validation_fraction=0.1,
                                   beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    mlp_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = mlp_classifier.predict_proba(x_test)

    # Results
    display_results(y_test, pred_probs)

    with open('pred_probas/text_mlp_classifier.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)


def model_lr_classifier(x_train, y_train, x_test, y_test):
    lr_classifier = LogisticRegression(solver='lbfgs',
                                       multi_class='multinomial',
                                       max_iter=1000)

    lr_classifier.fit(x_train, y_train)

    # Predict
    pred_probs = lr_classifier.predict_proba(x_test)

    # Results
    display_results(y_test, pred_probs)

    with open('pred_probas/text_lr_classifier.pkl', 'wb') as f:
        pickle.dump(pred_probs, f)


def model_ensemble_of_classifiers(y_test):
    # Load predicted probabilities
    with open('pred_probas/text_rf_classifier.pkl', 'rb') as f:
        rf_pred_probs = pickle.load(f)

    with open('pred_probas/text_xgb_classifier.pkl', 'rb') as f:
        xgb_pred_probs = pickle.load(f)

    with open('pred_probas/text_svc_classifier_model.pkl', 'rb') as f:
        svc_preds = pickle.load(f)

    with open('pred_probas/text_mnb_classifier.pkl', 'rb') as f:
        mnb_pred_probs = pickle.load(f)

    with open('pred_probas/text_mlp_classifier.pkl', 'rb') as f:
        mlp_pred_probs = pickle.load(f)

    with open('pred_probas/text_lr_classifier.pkl', 'rb') as f:
        lr_pred_probs = pickle.load(f)

    # Average of the predicted probabilites
    ensemble_pred_probs = (xgb_pred_probs +
                           mlp_pred_probs +
                           rf_pred_probs +
                           mnb_pred_probs +
                           lr_pred_probs)/5.0

    # Show metrics
    display_results(y_test, ensemble_pred_probs)


def load_data():
    df = pd.read_csv('data/t2e/text_train.csv')
    df = df.append(pd.read_csv('data/t2e/text_test.csv'))
    features = train_tfidf_vectors(df)
    labels = df.label
    return features, labels


def main():
    x_train, x_test, y_train, y_test = create_train_test_split(load_data())
    model_random_forest_classifier(x_train, y_train, x_test, y_test)
    model_xgb_classifier(x_train, y_train, x_test, y_test)
    model_svc_classifier(x_train, y_train, x_test, y_test)
    model_multinomial_naive_bayes_classifier(x_train, y_train, x_test, y_test)
    model_ensemble_of_classifiers(y_test)


if __name__ == '__main__':
    main()
