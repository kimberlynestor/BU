"""
Kimberly Nestor
CS777 Big Data Analytics
10/10/2021
Homework 4
Description: test logistic regression algorithm
"""


import re
import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

from sklearn.metrics import f1_score

import numpy as np

np.set_printoptions(threshold=sys.maxsize)


## IMPORT DATA
# test data
# test_data = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/TestingData.txt'
test_data = 'gs://metcs777/TestingData.txt'

#output regression coefficient weights
# outdir = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/lr_putput'
# outdir = 'gs://bigdatamodfour/lr_output'

# weight file
# weight_path = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/output_small_noreg/lr_weights_noreg/lr_weights_part-00000'
weight_path = 'gs://bigdatamodfour/lr_weights_part-00000'


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

#udf for dot product
dot = lambda arr1, arr2: arr1.dot(arr2)
dot_col = udf(dot, DoubleType())


## INIT
sc = SparkContext(appName="LogisticRegression")
sp = SparkSession.builder.getOrCreate()

n_word = 20000


####
# import weights
lr_weights_file = sc.textFile(weight_path)
lr_weights = np.array(lr_weights_file.map(lambda x: float(x)).collect())
####

# lr_weights_file = sc.textFile(weight_path)[0]
# lr_weights = np.array(lr_weights_file.map(lambda x: float(x)).collect())


# import and clean data
d_corpus = sc.textFile(test_data)
d_keyAndText = d_corpus.map(lambda x: (x[x.index('id="') + 4: x.index('" url=')], \
               x[x.index('">') + 2:][:-6]))
regex = re.compile('[^a-zA-Z]')

# doc id and list of words in doc, ('id', [list of words])
d_keyAndListOfWords = d_keyAndText.map(lambda x: (str(x[0]), regex.sub(' ', \
                                                        x[1]).lower().split()))

# organize data into ("word", 123) form, where idx 1 is num of times word is used
num_words = d_keyAndListOfWords.flatMap(lambda x: x[1]).groupBy(lambda x: x).\
            mapValues(lambda x: len(x)).sortBy(lambda x: x[1], ascending=False)

# get top 20k used words as array and dictionary
top_20k_words = num_words.take(n_word)  # word and num times used
dict_top_20k = dict(top_20k_words)

top_20k_places = [(i, j) for i, j in zip(dict_top_20k.keys(), \
                    list(range(n_word)))]  # word and rank in freq use
dict_top_20k_places = dict(top_20k_places)

# creat df with doc and words, explode words, sum num words in doc
df_doc = sp.createDataFrame(d_keyAndListOfWords, ['docID', 'word'])

# create docs with info of tot word count and true article type labels
df_doc_tot_wrd = df_doc.select(df_doc.docID.alias('ID'),
                               size('word').alias('doc_tot_wrd'))
df_doc_tot_wrd = df_doc_tot_wrd.withColumn('y_true',
                                           aus_check_col(df_doc_tot_wrd.ID))
# df_doc_labels = df_doc.select(df_doc.docID).withColumn('y_true', aus_check_col(df_doc.docID))

# go back to exploding the df
df_doc = df_doc.select(df_doc.docID, explode(df_doc.word).alias('word'), ). \
                withColumn('count', lit(1))  # lit, map to all rows

df_doc = df_doc.groupBy('docID', 'word').agg(sum('count').alias('count'))

# calculate term frequency, position of words, filter ones not in top20k
df_doc = df_doc.join(df_doc_tot_wrd, df_doc.docID == df_doc_tot_wrd.ID). \
                withColumn('tf', col('count') / col('doc_tot_wrd'))

df_doc_info = df_doc.select(['docID', 'y_true', 'doc_tot_wrd', 'word', 'count', 'tf'])
df_doc_info = df_doc_info.withColumn('top_pos', top_check_col(df_doc_info.word))
y_true_rdd = df_doc.select(['docID', 'y_true']).rdd
# print(y_true_rdd.collect())
# sys.exit()

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


# normalize x_data
df_info_vec = df_info_vec.withColumn('tf_min', array_min(df_info_vec.tf_vec))
df_info_vec = df_info_vec.withColumn('tf_max', array_max(df_info_vec.tf_vec))

df_info_vec = df_info_vec.withColumn('tf_vec_norm', \
            expr("""transform(tf_vec, x -> (x - tf_min)/ (tf_max - tf_min) )"""))
# df_info_vec.show()
# df_info_vec = df_info_vec.withColumn('tf_vec_norm', col('tf_vec') )


####  TASK 3 - TEST
# determined predicted classification of wiki pages
y_true_df = sp.createDataFrame(y_true_rdd, ['ID', 'y_true'])

df_test = df_info_vec.select(['docID', 'tf_vec_norm'])
df_test = df_test.join(y_true_df, df_test.docID == y_true_df.ID)

df_test = df_test.withColumn('weights', array([lit(i) for i in list( lr_weights )]) )
df_test = df_test.withColumn('theta', dot_col(col('weights'), col('tf_vec_norm')))
df_test = df_test.withColumn('y_pred', 1/ (1 + exp(-1*col('theta'))) )

df_test_fin = df_test.select(['docID', 'y_true', 'y_pred'])
# df_test_fin.show(500)

#calculate f1 score
y_true = np.array(y_true_rdd.map(lambda row : row[0]).collect())
y_pred = np.array(df_test_fin.select('y_pred').rdd.map(lambda row : row[0]).collect())

f1score = f1_score(y_true, y_pred)

print("\nF1 score for Task 2: ", f1score, "\n")


"""False positives from this model occured when non-australian court case wiki 
pages contained words with alot of legal jargon. In these instances it would be 
difficult for the model to determine the correct classification since it is the 
words in the document and specifially how many times words occurred to determine 
the classification."""