"""
Kimberly Nestor
CS777 Big Data Analytics
09/27/2021
Homework 3 Question3
Description:
"""

import sys
import numpy as np

from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors   #Vectors.dense
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql.types import StructType, StructField, FloatType, IntegerType


# Set file paths
taxi_file = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/taxi-data-sorted-small.csv'

# small
# taxi_file = 'gs://metcs777/taxi-data-sorted-small.csv.bz2'

# large
# taxi_file = 'gs://metcs777/taxi-data-sorted-large.csv.bz2'

outdir_cost_day = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/cost_day'
# outdir_cost_day = 'gs://bigdatamodthree/cost_day'
outdir_pred_day = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/pred_day'
# outdir_pred_day = 'gs://bigdatamodthree/pred_day'


outdir_cost_night = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/cost_night'
# outdir_cost_night = 'gs://bigdatamodthree/cost_night'
outdir_pred_night = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/pred_night'
# outdir_pred_night = 'gs://bigdatamodthree/pred_night'


# Exception Handling and removing wrong data lines
def isfloat(value):
    try :
        float(value)
        return True
    except :
        return False

# Remove lines if they don’t have 16 values and ...
def correctRows (p) :
    if(len(p) == 17):
        if(isfloat(p[5]) and isfloat(p[11])):
            if(float(p[5])!=0 and float(p[11])!=0):
                return p


# init
sc = SparkContext()
sp = SparkSession.builder.getOrCreate()

# obtain data from inputs
taxilines = sc.textFile(taxi_file).map(lambda x: x.split(","))
# cleaning up data
texilinesCorrected = taxilines.filter(correctRows)


# make dataframe ;  x_axis = trip_dist ;  y_axis = fare_amt
df_taxi = sp.createDataFrame(texilinesCorrected).select(col('_2').alias('driver_id'), \
        col('_5').alias('trip_time').cast('float'), col('_6').alias('dist').cast('float'), \
                    col('_16').alias('toll').cast('int'), col('_3').alias('pickup_time'), \
                                                        col('_17').alias('total_paid'))

# select relevant columns, split date and time, set dtypes
split_col = split(df_taxi['pickup_time'], ' ')
df_taxi = df_taxi.withColumn('trip_time_hr', col('trip_time')/120). \
                withColumn('p_date', split_col.getItem(0)). \
                withColumn('p_time', split_col.getItem(1)). \
                drop('pickup_time').drop('trip_time')

df_taxi = df_taxi.withColumn('p_date', to_date(df_taxi.p_date)). \
                    withColumn('p_timestamp', to_timestamp(df_taxi.p_time))

# n_rows = df_taxi.count()
# df_taxi.show()

# separate df into day and night dfs - night = 1am-6pm rides
df_taxi_day = df_taxi.filter((df_taxi["p_timestamp"] > lit('2021-09-30 06:00:00')) | \
                             (df_taxi["p_timestamp"] < lit('2021-09-30 01:00:00')))
df_taxi_night = df_taxi.filter(df_taxi["p_timestamp"] >= lit('2021-09-30 01:00:00')). \
                    filter(df_taxi["p_timestamp"] <= lit('2021-09-30 06:00:00'))


# get total feature numbers per day/night, grouped by driver and day
df_taxi_day = df_taxi_day.groupBy('driver_id', 'p_date').agg(sum("dist").alias('dist_tot'), \
                        sum("total_paid").alias('pay_tot'), sum("toll").alias('toll_tot'), \
                        sum("trip_time_hr").alias('trip_time_tot'), collect_list("p_timestamp").\
                                                       alias('p_timestamp'))
df_taxi_day = df_taxi_day.select('*', size('p_timestamp').alias('num_rides')).\
                                    drop('p_timestamp')

df_taxi_night = df_taxi_night.groupBy('driver_id', 'p_date').agg(sum("dist").alias('dist_tot'), \
                        sum("total_paid").alias('pay_tot'), sum("toll").alias('toll_tot'), \
                        sum("trip_time_hr").alias('trip_time_tot'), collect_list("p_timestamp").\
                                                       alias('p_timestamp'))
df_taxi_night = df_taxi_night.select('*', size('p_timestamp').alias('num_rides')).\
                                    drop('p_timestamp')
# df_taxi_day.show()
# df_taxi_night.show()


# INIT ALL
# learningRate = 0.01
# learningRate = 0.0000001
# learningRate = 0.00001 #left
learningRate = 0.0001 #right

num_iteration = 400 # max 400
m_curr = 0.1
m_curr_vec = [0.1] *4
b_curr = 0.1
prev_cost = 0


#### DAY MULTIPLE LINEAR REGRESSION - pay_tot prediction
day_out_lst = []
n = df_taxi_day.count()

## M section
# assemble df with weights to use in multiple linear regression
df_m_weights = df_taxi_day.select(['pay_tot', 'dist_tot', 'toll_tot', 'trip_time_tot', 'num_rides'])

#convert columns to single feature vector col
assembler = VectorAssembler(inputCols=['dist_tot', 'toll_tot', 'trip_time_tot', \
                                       'num_rides'], outputCol='weights')
df_m_weights = assembler.transform(df_m_weights) #output

#convert from vector to array type, add m_curr
df_m_weights = df_m_weights.withColumn('weights', vector_to_array('weights'))

# m_curr_vec = sc.parallelize([4.84458548, 4.81703908, 2.68240772, 5.04026948])
#make dataframe with lr coefficients, multiple to num of rows in dataset
# df_m_vec = m_curr_vec.map(lambda x: (x, )).toDF().groupBy().agg(collect_list("_1"). \
#                                                                 alias('lr_coef'))
# df_m_vec = df_m_vec.withColumn('lr_coef', explode(array_repeat(df_m_vec.lr_coef, n)) )
# m_curr_vec_dup = np.tile(m_curr_vec, (5, 1))


# Loop to implement gradient descent algorithm
for i in range(num_iteration):
    ## fimish M section
    df_m_weights = df_m_weights.withColumn("m_curr", lit(np.mean(m_curr_vec)))
    df_m_weights = df_m_weights.withColumn('m_weights', \
                expr("""transform(weights,x -> x * (pay_tot - (m_curr * x)) )"""))
    # print(df_m_weights.dtypes)
    # df_m_weights.show(truncate=False)

    # sigmoid sum, finish math operation, m vector weights
    m_weights_vec = df_m_weights.select("m_weights", df_m_weights.m_weights[0], \
                        df_m_weights.m_weights[1], df_m_weights.m_weights[2], \
                        df_m_weights.m_weights[3]).groupBy().sum().collect()
    m_weights_vec = np.squeeze((-1.0 / n) * np.array(m_weights_vec))
    # m_curr = m_weights_vec

    ##### compute costs
    curr_cost = df_taxi_day.withColumn('cost', (col('pay_tot') - (m_curr * \
                        col('dist_tot'))) ** 2).groupBy().sum().collect()[0][-1]

    # calculate partial derivatives of slope and intercept
    m_gradient = (-1.0 / n) * df_taxi_day.withColumn('m_grad', (col('dist_tot') * \
                        (col('pay_tot') - (m_curr * col('dist_tot'))))).\
                            groupBy().sum().collect()[0][-1]

    b_inter = (-1.0 / n) * df_taxi_day.withColumn('b_inter', (col('pay_tot') - \
                        (m_curr * col('dist_tot'))))

    #print and save parameters and cost
    print(i, "m=", m_curr_vec, " b=", b_curr, " Cost=", curr_cost)
    day_out_lst.append([i, m_curr_vec, b_curr, curr_cost])


    # check if cost is changing significantly, calculate and save y prediction
    # m1x1 + m2x2 + m3x3 + m4x4 + b = 0
    if prev_cost > curr_cost and (prev_cost - curr_cost) < 0.01:
        mx_vec = np.sum(np.squeeze(np.multiply(m_curr_vec, np.array(df_m_weights. \
                        select('weights').rdd.collect()))), axis=1)
        y_pred = sc.parallelize(mx_vec + b_curr).coalesce(1)
        y_pred.saveAsTextFile(outdir_pred_day)


        # df_m_weights.show()
        break

    #no bold driver
    # else:
    #     # update the weights - Regression Coefficients
    #     m_curr = m_curr - learningRate * m_gradient
    #     m_curr_vec = np.subtract(m_curr, np.multiply(learningRate, m_weights_vec))
    #
    #     b_curr = b_curr - learningRate * b_inter
    #
    #     prev_cost = curr_cost

    ### BOLD DRIVER - If cost decreases, increase learning rate
    elif prev_cost > curr_cost:
        learningRate = 1.05 * learningRate
        # update the weights - Regression Coefficients
        m_curr = m_curr - learningRate * m_gradient
        m_curr_vec = np.subtract(m_curr, np.multiply(learningRate, m_weights_vec))

        b_curr = b_curr - learningRate * b_inter

        prev_cost = curr_cost

    ### BOLD DRIVER - If cost increases, decrease rate
    elif prev_cost < curr_cost:
        learningRate = 0.5 * learningRate
        # update the weights - Regression Coefficients
        m_curr = m_curr - learningRate * m_gradient
        m_curr_vec = np.subtract(m_curr, np.multiply(learningRate, m_weights_vec))

        b_curr = b_curr - learningRate * b_inter

        prev_cost = curr_cost

# save cost and parameters to file
sc.parallelize(day_out_lst).coalesce(1).saveAsTextFile(outdir_cost_day)


# time for large dataset = 9hr 30min
# Stopped it because cost was not decreasing significantly and it ran for a while
