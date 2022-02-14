"""
Kimberly Nestor
CS777 Big Data Analytics
10/10/2021
Homework 4
Description: implement logistic regression using gradient descent from scratch
"""

from __future__ import print_function

import re
import sys

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.mllib.feature import HashingTF, IDF
from pyspark.mllib.linalg import DenseVector
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array

from collections import Counter
from operator import add
import heapq

import numpy as np
from sklearn import preprocessing

np.set_printoptions(threshold=sys.maxsize)
# sp.conf.set('spark.sql.pivotMaxValues', u'20000')


## IMPORT DATA
# small train
# train_data = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/SmallTrainingData.txt'
train_data = 'gs://metcs777/SmallTrainingData.txt'

# large train
# train_data = 'gs://metcs777/TrainingData.txt'

# test data
test_data = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/TestingData.txt'
# test_data = 'gs://metcs777/TestingData.txt'

#output regression coefficient weights
# outdir = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/lr_putput'
outdir = 'gs://bigdatamodfour/lr_output'


## INIT
sc = SparkContext(appName="LogisticRegression")
sp = SparkSession.builder.getOrCreate()

hashingTF = HashingTF()

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

if __name__ == "__main__":

    # import and clean data
    d_corpus = sc.textFile(train_data)
    d_keyAndText = d_corpus.map(lambda x : (x[x.index('id="') + 4 : x.index('" url=')], \
                                            x[x.index('">') + 2:][:-6]))
    regex = re.compile('[^a-zA-Z]')

    # doc id and list of words in doc, ('id', [list of words])
    d_keyAndListOfWords = d_keyAndText.map(lambda x : (str(x[0]), regex.sub(' ', x[1]).lower().split()))


    ####  TASK 1
    # organize data into ("word", 123) form, where idx 1 is num of times word is used
    num_words = d_keyAndListOfWords.flatMap(lambda x: x[1]). \
                    groupBy(lambda x: x).mapValues(lambda x: len(x)). \
                    sortBy(lambda x: x[1], ascending=False)

    # get top 20k used words as array and dictionary
    top_20k_words = num_words.take(n_word) #word and num times used
    dict_top_20k = dict(top_20k_words)

    top_20k_places = [(i,j) for i, j in zip(dict_top_20k.keys(), list(range(n_word)))] #word and rank in freq use
    dict_top_20k_places = dict(top_20k_places)


    # check position of list of words for task1
    check_wrds = ['applicant', 'and', 'attack', 'protein', 'car']
    check_wrds_pos = map_top_check(check_wrds)
    check_wrds_places = [(i,j) for i, j in zip(check_wrds, check_wrds_pos)] #word and rank in freq use

    print("\nFrequency position of check words for Task 1:\n", check_wrds_places, "\n")


    ####  TASK 2
    # doc_vec = d_keyAndListOfWords.map(lambda x: x[1])
    # doc_vec_pairs = map(lambda listt: tuple(map(lambda word: (word, 1), listt)), doc_vec.collect())
    # doc_pairs = sc.parallelize(doc_vec_pairs)

    #creat df with doc and words, explode words, sum num words in doc
    df_doc = sp.createDataFrame(d_keyAndListOfWords, ['docID', 'word'])

    #create docs with info of tot word count and true article type labels
    df_doc_tot_wrd = df_doc.select(df_doc.docID.alias('ID'), size('word').alias('doc_tot_wrd'))
    df_doc_tot_wrd = df_doc_tot_wrd.withColumn('y_true', aus_check_col(df_doc_tot_wrd.ID))
    # df_doc_labels = df_doc.select(df_doc.docID).withColumn('y_true', aus_check_col(df_doc.docID))

    #go back to exploding the df
    df_doc = df_doc.select(df_doc.docID, explode(df_doc.word).alias('word'), ). \
                        withColumn('count', lit(1)) #lit, map to all rows

    df_doc = df_doc.groupBy('docID', 'word').agg(sum('count').alias('count'))


    # calculate term frequency, position of words, filter ones not in top20k
    df_doc = df_doc.join(df_doc_tot_wrd, df_doc.docID == df_doc_tot_wrd.ID). \
        withColumn('tf', col('count') / col('doc_tot_wrd') )

    df_doc_info = df_doc.select(['docID', 'y_true', 'doc_tot_wrd', 'word', 'count', 'tf'])
    df_doc_info = df_doc_info.withColumn('top_pos', top_check_col(df_doc_info.word))

    df_info_fil = df_doc_info.filter(df_doc_info.top_pos != -1)



    """
    #re-combine words and count of words with docID, implode
    df_info_rdd = df_info_fil.select('docID', struct(['word', 'tf', 'top_pos']). \
                                    alias('wrd_tf_pos')) #struct combine two cols
    df_info_rdd = df_info_rdd.groupBy('docID').agg(collect_list('wrd_tf_pos'). \
                                                           alias('doc_wrd_tf_pos'))
    doc_info_rdd = df_info_rdd.rdd
    y_true = df_info_fil.select('y_true').rdd #no longer in right order
    # df_doc.where(col('docID') == 'AU35').sort('word').show(100)
    # print(doc_info_rdd.top(1))

    #get rdd just pos and tf, make tf sparse vector of top words, and get y_true
    all_pos_tf = doc_info_rdd.map(lambda x: list(map(lambda xx: (xx[2], xx[1]), x[1]))  )

    # reduce according to position in top 20k, make tf vector
    pos_tf_red = all_pos_tf.flatMap(lambda x: x).reduceByKey(add).sortBy(lambda x: x[0])
    tf_vec_red = build_tf_vec(pos_tf_red.collect())
    # df_pos_tf_red = pos_tf_red.toDF(['pos', 'red_tf'])
    # df_pos_tf_red.show()
    # print(df_pos_tf_red.count())
    # sys.exit()

    # 20k vector, ordered by tf word position
    # all_tf_vec = all_pos_tf.map(lambda x: build_tf_vec(x))
    """

    #assemble 20k vector in dataframe for each doc, with relevant tf inserted in word position
    assembler = VectorAssembler(inputCols=['top_pos', 'tf'], outputCol='pos_tf')

    df_info_vec = df_info_fil.select('*')
    df_info_vec = assembler.transform(df_info_vec)  # output
    df_info_vec = df_info_vec.withColumn('pos_tf', vector_to_array('pos_tf')). \
                    groupBy('docID').agg(collect_list('pos_tf').alias('pos_tf_all'))

    df_info_vec = df_info_vec.withColumn('tf_vec', build_tf_vec_col(df_info_vec.pos_tf_all))
    # df_info_vec.show()#truncate=False)

    #normalize x data
    df_info_vec = df_info_vec.withColumn('tf_min', array_min(df_info_vec.tf_vec))
    df_info_vec = df_info_vec.withColumn('tf_max', array_max(df_info_vec.tf_vec))

    df_info_vec = df_info_vec.withColumn('tf_vec_norm', \
                expr("""transform(tf_vec, x -> (x - tf_min)/ (tf_max - tf_min) )"""))
    # df_info_vec.show()
    # sys.exit()


    #### TRAIN - gradient descent
    l_rate = 0.1  # 0.0000001  0.0001  0.01
    epoch = 400

    m_curr = 0.1
    m_curr_vec = np.array(([0.1] * n_word))
    b_curr = 0.1
    prev_loss = 0

    lam = 1

    output_lst = []


    """
    # get initialized dataframe as input to GD algorithm
    df_gd = df_info_fil.withColumn('m_curr', lit(np.mean(m_curr_vec)))
    df_gd = df_gd.withColumn('theta', col('m_curr') *  col('tf') ). \
                withColumn('y_theta', col('theta') *  col('y_true') )

    #reduce to all docs by word position
    # df_gd_red = df_gd.groupBy('top_pos').sum('theta')#.sum('y_theta')
    df_gd_red = df_gd.groupBy('top_pos').agg(sum('theta').alias('theta'), \
                            sum('y_theta').alias('y_theta'), avg('y_true').alias('y_true')). \
                            sort('top_pos')
    df_gd_red.show(1000)
    # df_gd_red.cache()

    #convert 20k reduce vector to array
    theta_vec = np.array(df_gd_red.select(df_gd_red.theta).collect()) #think dict if doesn't keep order
    y_theta_vec = np.array(df_gd_red.select(df_gd_red.y_theta).collect())
    y_vec = 0

    print(theta_vec)
    print(np.exp(theta_vec))
    """

    #make df for gradient descent
    df_info_vec = df_info_vec.join(df_doc_tot_wrd, df_info_vec.docID == df_doc_tot_wrd.ID)
    df_gd = df_info_vec.select(['docID', 'y_true', 'tf_vec_norm'])
    # df_gd.cache()
    # df_gd.show()


    # Loop to implement gradient descent algorithm
    for i in range(epoch):
        # break

        #calculate theta
        # df_gd.cache()
        df_gd = df_gd.withColumn("m_curr", lit(np.mean(m_curr_vec)))
        df_gd = df_gd.withColumn('pos', array([lit(i) for i in list(range(n_word))]))

        df_gd = df_gd.withColumn('theta', expr("""transform(tf_vec_norm, x -> x * m_curr )"""))
        df_gd = df_gd.withColumn('theta', summ_arr('theta'))

        #calculate exponent theta
        df_gd = df_gd.withColumn('exp_theta', exp(col('theta')) )

        #calculate y theta term
        df_gd = df_gd.withColumn('y_theta', -1*(col('y_true') * col('theta')) )

        #calculate partial derivative
        # df_gd = df_gd.withColumn("parder1",  lit(-1))
        df_gd = df_gd.withColumn('parder1', expr("""transform(tf_vec_norm, x -> -1*(x * y_true) )"""))
        df_gd = df_gd.withColumn('parder2', col('exp_theta')/ (1+ col('exp_theta'))  )
        df_gd = df_gd.withColumn('parder2', expr("""transform(tf_vec_norm, x -> x * parder2 )"""))


        # reduce according to word position in top 20k dict
        df_parder = df_gd.select(['docID', 'parder1', 'parder2', 'pos'])
        df_parder = df_parder.withColumn('zip', arrays_zip('parder1', 'parder2', 'pos')). \
                            withColumn('zip', explode('zip')).select("docID", \
                            col("zip.parder1"), col("zip.parder2"), col("zip.pos"))
        # df_parder.cache()


        df_parder = df_parder.groupBy('pos').agg(sum('parder1').alias('parder1'), \
                                sum('parder2').alias('parder2')).sort('pos')

        df_parder = df_parder.withColumn('term1', col('parder1') + col('parder1') )


        # get vector of regression coefficients - partial derivative
        r_weight = np.array(df_parder.select('term1').rdd.map(lambda row : row[0]).\
                            collect())
        term2 = np.array(((2. * lam) * m_curr_vec))
        r_weight_vec = np.add(r_weight, term2)
        # print(r_weight_vec)

        #calculate intercept
        df_gd = df_gd.withColumn('b_inter', summ_arr('tf_vec_norm'))
        b_inter = (-1.0 / df_gd.count()) * df_gd.withColumn('b_inter', (col('y_true') - \
                    (np.mean(m_curr_vec) * col('b_inter')))).groupBy().sum().collect()[0][-1]
        # print(b_inter)


        # loss - no regularization
        curr_loss = df_gd.withColumn('loss', col('y_theta') +  log((1+ col('exp_theta') ))  ). \
                            groupBy().sum().collect()[0][-1]

        # # #loss - regularization
        # curr_loss = df_gd.withColumn('loss', col('y_theta') +  log((1+ col('exp_theta'))) + \
        #                 (lam * (np.mean(m_curr_vec)**2)) ).groupBy().sum().collect()[0][-1]


        # print and save parameters and cost
        print(i, "b =", b_curr, " loss =", curr_loss)
        # out_lst.append([i, m_curr_vec, b_curr, curr_cost])
        # break

        # check is loss changes sig, save lr weights and b_inter
        # m1x1 + m2x2 + m3x3 + m4x4 + b = 0
        if prev_loss > curr_loss and (prev_loss - curr_loss) < 0.01:
            print(r_weight_vec)
            print(b_inter)

            output_lst.append(r_weight_vec)
            output_lst.append(b_inter)

            break

        else:
            # update the weights - Regression Coefficients
            m_curr_vec = np.subtract(m_curr_vec, np.multiply(l_rate, r_weight_vec))
            b_curr = b_curr - l_rate * b_inter
            prev_loss = curr_loss


    #save weights to list, then file, assign to var
    sc.parallelize(output_lst).coalesce(1).saveAsTextFile(outdir)
    lr_weights = output_lst[0] #lst
    b_inter = output_lst[1][0] #float

    """
    ##########
    weight_path = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/output_small_noreg/lr_weights_noreg/lr_weights_part-00000'
    b_inter_path = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module4/output_small_noreg/lr_weights_noreg/b_inter_noreg'

    lr_weights_file = open(weight_path, "r")
    b_inter_file = open(b_inter_path, "r")

    lr_weights = np.array(  list(map(float, lr_weights_file.readlines()))  )
    b_inter = float(b_inter_file.readlines()[0])
    ###########
    """

    #find five words with largest regression coefficients - most linked to AUS case
    lr_weights_enum = list(enumerate(lr_weights))
    lr_weights_sort = sorted(lr_weights_enum, key=lambda x: x[1], reverse=True)
    lr_weights_largest = lr_weights_sort[:5]

    largest_idx = [i[0] for i in lr_weights_largest]
    lr_top_words = list(map(lambda x: list(dict_top_20k_places.keys())[x], \
                                        largest_idx))

    print("Words with largest regression coefficients for Task 2:\n", \
          [(i,j) for i, j in zip(lr_top_words, largest_idx)], "\n")




# Frequency position of check words for Task 1:
# [('applicant', 346), ('and', 2), ('attack', 502), ('protein', 3014), ('car', 608)]

# job ended after 2hr5min