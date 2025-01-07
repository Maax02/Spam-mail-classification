import numpy as np
import pandas as pd
import chardet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn import svm

from sklearn.metrics import accuracy_score


# data loading
data = pd.read_csv('/content/spam.csv')
print(data.head())
print(data.shape)

Y = data['Category'] # spam or ham
# Y.tail()
X = data['Message'] # message
# X.head()

# splitting data into test and train
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=3)
print(X_train.shape)
print(X_test.shape)

# changing labels (spam/ham) to (1/0)
# spam = 1, ham = 0
def change_label(y, classes):
    result = []
    for sh in y:
      result.append(classes[sh])
    return result

classes = {'spam': 1, 'ham': 0}
classes_inv = {1: 'spam', 0: 'ham'}
Y_train = change_label(Y_train, classes)
Y_test = change_label(Y_test, classes)
# print(y_train)
# print(y_test)

#lower case message
X_train = X_train.str.lower()
X_test = X_test.str.lower()

# remove stop words (words without "value")
def remove_stop_words(data, stop_words):
    new_data = []
    sw = set(stop_words)
    for line in data:
      removed = " ".join([word for word in line.split() if word not in sw])
      new_data.append(removed)
    return new_data


stop_words = np.loadtxt('/content/EN-Stopwords.txt', delimiter=',', dtype='str')
# X_train = remove_stop_words(X_train, stop_words)
# X_test = remove_stop_words(X_test, stop_words)
# print(X_train)

# transforming messages into vectors
# bag of words method
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train)
X_test_bow = bow.transform(X_test)

# TfidfVectorizer method
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

#print(X_train_bow)
# print(X_train_tfidf)

# training models

# logistic regression
# bow vectors
log_reg_bow = LogisticRegression()
log_reg_bow.fit(X_train_bow, Y_train)

# training data
prediction_train_bow = log_reg_bow.predict(X_train_bow)
accuracy_train_bow = accuracy_score(Y_train, prediction_train_bow)
print("Training accuracy for logistic regression bow: {}".format(accuracy_train_bow))

# test data
prediction_test_bow = log_reg_bow.predict(X_test_bow)
accuracy_test_bow = accuracy_score(Y_test, prediction_test_bow)
print("Test accuracy for logistic regression bow: {}".format(accuracy_test_bow))

# tfidf
log_reg_tfidf = LogisticRegression()
log_reg_tfidf.fit(X_train_tfidf, Y_train)

# training data
prediction_train_tfidf = log_reg_tfidf.predict(X_train_tfidf)
accuracy_train_tfidf = accuracy_score(Y_train, prediction_train_tfidf)
print("Training accuracy for logistic regression tfidf: {}".format(accuracy_train_tfidf))

# test data
prediction_test_tfidf = log_reg_bow.predict(X_test_tfidf)
accuracy_test_tfidf = accuracy_score(Y_test, prediction_test_tfidf)
print("Test accuracy for logistic regression tfidf: {}".format(accuracy_test_tfidf))

# svm model
# bow
svm_model_bow = svm.SVC()
svm_model_bow.fit(X_train_bow, Y_train)

print("Traning accuracy for SVM bow: {}".format(svm_model_bow.score(X_train_bow, Y_train)))
print("Test accuracy for SVM bow: {}".format(svm_model_bow.score(X_test_bow, Y_test)))

# tfidf
svm_model_tfidf = svm.SVC()
svm_model_tfidf.fit(X_train_tfidf, Y_train)

print("Traning accuracy for SVM tfidf: {}".format(svm_model_tfidf.score(X_train_tfidf, Y_train)))
print("Test accuracy for SVM tfidf: {}".format(svm_model_tfidf.score(X_test_tfidf, Y_test)))

# randomly hand picked mails from spam.csv
# spam (1)
mail = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]
mail = remove_stop_words(mail, stop_words)
mail_bow = bow.transform(mail)
mail_tfidf = tfidf.transform(mail)

print(log_reg_bow.predict(mail_bow))
print(log_reg_tfidf.predict(mail_tfidf))
print(svm_model_bow.predict(mail_bow))
print(svm_model_tfidf.predict(mail_tfidf))

# ham (0)
mail = ["Pls go ahead with watts. I just wanted to be sure. Do have a great weekend. Abiola"]
mail = remove_stop_words(mail, stop_words)
mail_bow = bow.transform(mail)
mail_tfidf = tfidf.transform(mail)

print(log_reg_bow.predict(mail_bow))
print(log_reg_tfidf.predict(mail_tfidf))
print(svm_model_bow.predict(mail_bow))
print(svm_model_tfidf.predict(mail_tfidf))