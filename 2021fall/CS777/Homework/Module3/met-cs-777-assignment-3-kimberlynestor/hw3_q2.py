"""
Kimberly Nestor
CS777 Big Data Analytics
09/27/2021
Homework 3 Question2
Description:
"""

import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from operator import add


# Set file paths
taxi_file = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/taxi-data-sorted-small.csv'

# small
# taxi_file = 'gs://metcs777/taxi-data-sorted-small.csv.bz2'

# large
# taxi_file = 'gs://metcs777/taxi-data-sorted-large.csv.bz2'

outdir_cost = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/cost'
outdir_pred = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/pred'

# outdir_cost = 'gs://bigdatamodthree/cost'
# outdir_pred = 'gs://bigdatamodthree/pred'


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
df_taxi = sp.createDataFrame(texilinesCorrected). \
    select(col('_6').alias('trip_dist').cast('float'), \
           col('_12').alias('fare_amt').cast('float'))
df_taxi = df_taxi.filter(( col('fare_amt') < 600 ) & (col('fare_amt') > 1 ) )

n_rows = df_taxi.count()


#### gradient descent - can use minibatch, sample every iteration
learningRate = 0.01 # 0.0000001
num_iteration = 400 # max 400
m_current = 0.1
b_current = 0.1

n = n_rows
prev_cost = 0

out_lst = []

# Loop to implement gradient descent algorithm
for i in range(num_iteration):
    # compute costs
    curr_cost = df_taxi.withColumn('cost', (col('fare_amt') - \
                            (m_current * col('trip_dist')))**2 ). \
                            groupBy().sum().collect()[0][2]

    # calculate partial derivatives of slope and intercept
    m_gradient = (-1.0 / n) * df_taxi.withColumn('cost', (col('trip_dist') * \
                            (col('fare_amt') - (m_current * col('trip_dist'))) )). \
                            groupBy().sum().collect()[0][2]

    b_inter = (-1.0 / n) * df_taxi.withColumn('cost', (col('fare_amt') - \
                            (m_current * col('trip_dist'))) ).groupBy(). \
                                sum().collect()[0][2]

    #print and save parameters and cost
    print(i, "m=", m_current, " b=", b_current, " Cost=", curr_cost)
    out_lst.append([i, m_current, b_current, curr_cost])

    #check if cost is changing significantly, calculate and save y prediction
    if prev_cost > curr_cost and (prev_cost - curr_cost) < 0.01:
        df_pred = df_taxi.withColumn('y_pred', ((m_current * col('trip_dist')) + b_current))
        # df_pred.show(100)
        df_pred.show()

        y_pred = df_pred.select('*').rdd.coalesce(1) #y_pred
        y_pred.saveAsTextFile(outdir_pred)
        break
    else:
        # update the weights - Regression Coefficients
        m_current = m_current - learningRate * m_gradient
        b_current = b_current - learningRate * b_inter

        prev_cost = curr_cost

# save cost and parameters to file
sc.parallelize(out_lst).coalesce(1).saveAsTextFile(outdir_cost)
# print(out_lst)

# small data set: epoch, m, cost = [53, 3.2674000939731513, 36119185.81536668]


# time for large dataset = 5hr 50min
# Stopped it because cost was not decreasing significantly and it ran for a while

