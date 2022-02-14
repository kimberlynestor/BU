"""
Kimberly Nestor
CS777 Big Data Analytics
10/2021
Homework 5
Description: implement logistic regression and SVM using spark libraries
"""

import re
import sys
import time

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark .ml. classification import LogisticRegression
from pyspark .ml. classification import LinearSVC
from pyspark.sql.functions import *
from pyspark.sql.types import *

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors, VectorUDT

from sklearn.metrics import f1_score

import numpy as np


## IMPORT DATA
# small train
train_data = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module5/SmallTrainingData.txt'
# train_data = 'gs://metcs777/SmallTrainingData.txt'

# large train
# train_data = 'gs://metcs777/TrainingData.txt'

# test data
test_data = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module5/TestingData.txt'
# test_data = 'gs://metcs777/TestingData.txt'


## INIT
sc = SparkContext()
sp = SparkSession.builder.getOrCreate()

n_word = 20000


def build_tf_vec(nest_rdd):
    # tf_vec = np.zeros(n_word)
    tf_vec = [0.0] * n_word
    for i in nest_rdd:
        pos = int(i[0])
        tf = float(i[1])
        tf_vec[pos] = tf
    # return list(tf_vec)
    # print(tf_vec)
    return tf_vec

#df column version
build_tf_vec_col = udf(lambda x: build_tf_vec(x), ArrayType(DoubleType()))

#udfs to handle looking for position of word in top 20k dict
top_check = lambda xx: dict_top_20k_places[xx] if xx in dict_top_20k.keys() else -1 #else 0
map_top_check = lambda inlist: list(map(top_check, inlist))
top_check_col = udf(lambda x: top_check(x), IntegerType())

#udfs to determine true binary labels, of article type
aus_check = lambda xx: 1 if xx.startswith('AU') else 0 # 1 = AUS article
aus_check_col = udf(lambda x: aus_check(x), IntegerType())

#udf to sum colum of arraytype
summ_arr = udf(lambda x: float(np.array(x, dtype=float).sum()), DoubleType())

#convert col arraytype to vector
to_vec = udf(lambda vs: Vectors.dense(vs), VectorUDT())


def get_data(data_path):
    global dict_top_20k
    global dict_top_20k_places

    # import and clean data
    d_corpus = sc.textFile(data_path)
    d_keyAndText = d_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], \
                   x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')

    # doc id and list of words in doc, ('id', [list of words])
    d_keyAndListOfWords = d_keyAndText.map(lambda x: (str(x[0]), \
                            regex.sub(' ', x[1]).lower().split()))


    # organize data into ("word", 123) form, where idx 1 is num of times word is used
    num_words = d_keyAndListOfWords.flatMap(lambda x: x[1]). \
        groupBy(lambda x: x).mapValues(lambda x: len(x)). \
        sortBy(lambda x: x[1], ascending=False)

    # get top 20k used words as array and dictionary
    top_20k_words = num_words.take(n_word)  # word and num times used
    dict_top_20k = dict(top_20k_words)

    top_20k_places = [(i, j) for i, j in zip(dict_top_20k.keys(), list(
        range(n_word)))]  # word and rank in freq use
    dict_top_20k_places = dict(top_20k_places)


    # creat df with doc and words, explode words, sum num words in doc
    df_doc = sp.createDataFrame(d_keyAndListOfWords, ['docID', 'word'])

    # create docs with info of tot word count and true article type labels
    df_doc_tot_wrd = df_doc.select(df_doc.docID.alias('ID'),\
                                size('word').alias('doc_tot_wrd'))
    df_doc_tot_wrd = df_doc_tot_wrd.withColumn('label', aus_check_col(df_doc_tot_wrd.ID))

    # go back to exploding the df
    df_doc = df_doc.select(df_doc.docID, explode(df_doc.word).alias('word'), ). \
                withColumn('count', lit(1))  # lit, map to all rows

    df_doc = df_doc.groupBy('docID', 'word').agg(sum('count').alias('count'))

    # calculate term frequency, position of words, filter ones not in top20k
    df_doc = df_doc.join(df_doc_tot_wrd, df_doc.docID == df_doc_tot_wrd.ID). \
                withColumn('tf', col('count') / col('doc_tot_wrd'))

    df_doc_info = df_doc.select(['docID', 'label', 'doc_tot_wrd', 'word', 'count', 'tf'])
    df_doc_info = df_doc_info.withColumn('top_pos', top_check_col(df_doc_info.word))

    df_info_fil = df_doc_info.filter(df_doc_info.top_pos != -1)

    # assemble 20k vector in dataframe for each doc, with relevant tf inserted in word position
    assembler = VectorAssembler(inputCols=['top_pos', 'tf'], outputCol='pos_tf')

    df_info_vec = df_info_fil.select('*')
    df_info_vec = assembler.transform(df_info_vec)  # output
    df_info_vec = df_info_vec.withColumn('pos_tf', vector_to_array('pos_tf')). \
        groupBy('docID').agg(collect_list('pos_tf').alias('pos_tf_all'))

    df_info_vec = df_info_vec.withColumn('tf_vec',
                                         build_tf_vec_col(df_info_vec.pos_tf_all))
    # df_info_vec.show()#truncate=False)

    # normalize x data
    df_info_vec = df_info_vec.withColumn('tf_sum', summ_arr('tf_vec'))

    df_info_vec = df_info_vec.withColumn('features', \
            expr("""transform(tf_vec, x -> x / tf_sum )""")) #tf_vec_norm

    # make df for training
    df_info_vec = df_info_vec.join(df_doc_tot_wrd, df_info_vec.docID == df_doc_tot_wrd.ID)
    df_output = df_info_vec.select(['docID', 'label', 'features'])
    df_output = df_output.withColumn('features', to_vec('features'))

    # df_train.cache()
    # df_train.show()
    return df_output


#### TRAIN DATA
read_train_start = time.time()

df_train = get_data(train_data)
# df_train.show()

read_train_end = time.time()
read_train_time = read_train_end - read_train_start ####

#### TEST DATA
read_test_start = time.time()

df_test = get_data(test_data)
# df_test.show()

read_test_end = time.time()
read_test_time = read_test_end - read_test_start ####


##### TASK 1 - LOGISTIC REGRESSION
# fit train data
lr_train_start = time.time()

lr = LogisticRegression(maxIter=100, regParam=0.3, elasticNetParam=0.8)
lrModel = lr.fit(df_train)

lr_train_end = time.time()
lr_train_time = lr_train_end - lr_train_start ####

# predict using test data
lr_test_start = time.time()

lr_pred = lrModel.transform(df_test)
# lr_pred.show(100)

lr_test_end = time.time()
lr_test_time = lr_test_end - lr_test_start ####

lr_tot_time = np.sum([read_train_time, read_test_time, lr_train_time, lr_test_time])

#calculate f1 score
y_true = lr_pred.select('label').rdd.map(lambda row : row[0]).collect()
y_pred = lr_pred.select('prediction').rdd.map(lambda row : row[0]).collect()

f1score = f1_score(y_true, y_pred, average='weighted')
print("\nF1 score for Logistic Regression: ", f1score)
print(f"Time to read train data ({read_train_time}), read test data ({read_test_time}), ", \
      f"train model ({lr_train_time}), test model ({lr_test_time}) = {lr_tot_time}\n")


##### TASK 2 - SVM
# fit train data
svm_train_start = time.time()

svm = LinearSVC(maxIter=100, regParam=1)
svmModel = svm.fit(df_train)

svm_train_end = time.time()
svm_train_time = svm_train_end - svm_train_start ####

# predict using test data
svm_test_start = time.time()

svm_pred = svmModel.transform(df_test)
# svm_pred.show(100)

svm_test_end = time.time()
svm_test_time = svm_test_end - svm_test_start ####

svm_tot_time = np.sum([read_train_time, read_test_time, svm_train_time, svm_test_time])

#calculate f1 score
y_true = svm_pred.select('label').rdd.map(lambda row : row[0]).collect()
y_pred = svm_pred.select('prediction').rdd.map(lambda row : row[0]).collect()

f1score = f1_score(y_true, y_pred, average='weighted')
print("\nF1 score for SVM: ", f1score)
print(f"Time to read train data ({read_train_time}), read test data ({read_test_time}), ", \
      f"train model ({svm_train_time}), test model ({svm_test_time}) = {svm_tot_time}\n")

