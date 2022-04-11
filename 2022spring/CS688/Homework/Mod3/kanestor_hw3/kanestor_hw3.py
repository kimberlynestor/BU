"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 04/07/22
Homework Problem: Hw 3
Description of Problem: This program
Q3 dataset:
https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news?select=fake_or_real_news.csv
"""

import sys
from os.path import join as opj

import numpy as np
import pandas as pd
# from scipy.stats import boxcox
from scipy import stats

import sklearn.metrics as sk
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.datasets import fetch_20newsgroups
from sklearn.linear_model import SGDClassifier

import matplotlib.pyplot as plt
import seaborn as sns


def get_dum_dict(class_lst):
    """This function takes as input a unique list of class names and returns a
    dict of dummy encoded values"""
    dum_dict = {}
    for i in range(len(class_lst)):
        # set last val to all zeros
        if i == len(class_lst)-1:
            dum = np.zeros(len(class_lst)-1)
            dum_dict[class_lst[i]] = dum
        # otherwise one hot encode
        else:
            dum = np.zeros(len(class_lst) - 1)
            dum[i] = 1
            dum_dict[class_lst[i]] = dum

    return(dum_dict)

# function takes list of class vals and returns dummy vals
real_dummy = lambda i : ir_cls_dict['setosa'] if i==0 \
    else (ir_cls_dict['versicolor'] if i==1 else ir_cls_dict['virginica'])


# load rivers data
curr_dir = sys.path[0]
riv_data = np.array(list(map(int, open(opj(curr_dir, 'rivers_data'), "r"). \
                        read().split(', '))))

# plot with kde
# plt.figure(figsize = (8, 8))
# sns.distplot(riv_data)
# # sns.distplot(np.log2(riv_data))
# # sns.distplot(stats.boxcox(riv_data)[0])
# plt.show()
# sys.exit()

#### Q1 - data transformation
# set fig info
fig, ax = plt.subplots(nrows=1, ncols=3)
fig.suptitle('Histogram of rivers data', fontname="serif", fontsize=13, fontweight='bold')

# plot original data histogram
ax[0].hist(riv_data)
ax[0].set_title('Original', fontname="serif")

# plot log transform histogram
ax[1].hist(np.log2(riv_data))
# ax[1] = sns.distplot(np.log2(riv_data))
ax[1].set_title('Log transform', fontname="serif")

# plot box cox transform histogram
ax[2].hist(stats.boxcox(riv_data)[0])
# ax[2] = sns.distplot(stats.boxcox(riv_data)[0])
ax[2].set_title('Box Cox transform', fontname="serif")

# whole figure axis labels
fig.add_subplot(1, 1, 1, frame_on=False).\
    tick_params(labelcolor="none", bottom=False, left=False)
plt.xlabel('Rivers data', fontname="serif", fontsize=10, fontweight='bold')
plt.ylabel("Frequency", fontname="serif", fontsize=10, fontweight='bold')
plt.savefig('hist_rivs_all.png', dpi=300)
plt.show()



#### Q2 - dummy encoding on iris data: [0, 1, 2] == ['setosa' 'versicolor' 'virginica']
# load iris data
ir_cls = datasets.load_iris().target_names
ir_cls_data = datasets.load_iris().target

# muse udf to make list of dummy vals
ir_cls_dict = get_dum_dict(ir_cls)
ir_tar_dum = [real_dummy(i) for i in ir_cls_data]

print("Iris data original class list:\n", ir_cls_data, "\n")
print("Iris data dummy encoded class list:\n", [list(map(int,i)) for i in ir_tar_dum])


#### Q3 - NLP on fake and real news dataset: label = fake,real = 0,1
# https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
df_news = pd.read_csv(opj(curr_dir, 'fake_or_real_news.csv'))
df_news = df_news.rename(columns={df_news.columns[0]: 'id'})
df_news['class'] = df_news['label'].apply(lambda i: 0 if i == 'FAKE' else 1)

# split into train and test
# df_news_train, df_news_test = train_test_split(df_news, random_state=0)

# convert data to list, train test split
news_data = df_news[['id', 'title', 'text', 'class']].values
news_train, news_test = train_test_split(news_data, random_state=0)

news_train_data = list(map(lambda x:x[2], news_train))
news_train_class = list(map(lambda x:x[-1], news_train))

news_test_data = list(map(lambda x:x[2], news_test))
news_test_class = list(map(lambda x:x[-1], news_test))

# vectorizer = CountVectorizer(stop_words='english', min_df=0.25)
# xx = vectorizer.fit_transform(news_train_data[0])
# print( news_train_data[0])
# print(vectorizer.get_feature_names()) #new= get_feature_names_out()
# sys.exit()


# CountVectorizer includes preprocessing: lowercase, tokenize, lemmatizing, stemming, vectorization
# removes: stop words, punctuation, space, low words
text_clf = Pipeline([ ('vect', CountVectorizer(stop_words='english')), \
                      ('tfidf', TfidfTransformer()), \
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', \
                                            alpha=.00001, random_state=0, \
                                            max_iter=1000, tol=0.001)) ])
# train classifier
text_clf.fit(news_train_data, news_train_class)

# test classifier
predicted = text_clf.predict(news_test_data)
# acc = np.mean(predicted == news_test_class)

# performance
rec = round(sk.recall_score(news_test_class, predicted), 4)
pre = round(sk.precision_score(news_test_class, predicted), 4)
f1 = round(sk.f1_score(news_test_class, predicted), 4)
acc = round(sk.accuracy_score(news_test_class, predicted), 4)

print('\nSVM Classifier')
print('  Recall: {}'.format(rec))
print('  Precision: {}'.format(pre))
print('  F1 score: {}'.format(f1))
print('  Accuracy: {}\n'.format(acc))

df_news_test = pd.DataFrame(news_test, columns=['id', 'title', 'text', 'class'])
df_news_test.insert(len(df_news_test.columns), column='Pred', value=predicted)

print(df_news_test.head(10))
print(df_news_test.tail(10))










#### test section on 20newsgroups data - IGNORE
"""
# text_clf = Pipeline([ ('vect', CountVectorizer(stop_words='english')), \
#                       ('tfidf', TfidfTransformer()), \
#                       ('clf', MultinomialNB()) ])

text_clf = Pipeline([ ('vect', CountVectorizer()), \
                      ('tfidf', TfidfTransformer()), \
                      ('clf', SGDClassifier(loss='hinge', penalty='l2', \
                                            alpha=1e-3, random_state=42, \
                                            max_iter=5, tol=None)) ])

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
# train data
twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
docs_train = twenty_train.data
train_class = twenty_train.target

# test data
twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
docs_test = twenty_test.data
test_class = twenty_test.target

text_clf.fit(docs_train, train_class)
predicted = text_clf.predict(docs_test)
acc = np.mean(predicted == test_class)
print(acc)
sys.exit()
"""