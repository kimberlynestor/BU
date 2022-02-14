"""
Kimberly Nestor
CS777 Big Data Analytics
09/27/2021
Homework 3 Question1
Description:
"""

import sys
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from operator import add


# Set file paths
# taxi_file = '/Users/kimberlynestor/Desktop/BU/2021fall/CS777/Homework/Module3/taxi-data-sorted-small.csv'

# small
# taxi_file = 'gs://metcs777/taxi-data-sorted-small.csv.bz2'

# large
taxi_file = 'gs://metcs777/taxi-data-sorted-large.csv.bz2'


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
# df_taxi.show()
# df_taxi.show(500, truncate=False)
# print(df_taxi.dtypes)


# find slope of the line, m
n_rows = df_taxi.count()

m_num = (n_rows * df_taxi.withColumn('m_slope', col('trip_dist') * \
            col('fare_amt')  ).groupBy().sum().collect()[0][2]) - \
          df_taxi.select(col('trip_dist')).groupBy().sum().collect()[0][0] * \
          df_taxi.select(col('fare_amt')).groupBy().sum().collect()[0][0]

m_denom = ( n_rows * df_taxi.withColumn('m_slope', col('trip_dist')**2 ). \
          groupBy().sum().collect()[0][2] ) - df_taxi.select(col('trip_dist')). \
            groupBy().sum().collect()[0][0] **2

m_slope = m_num / m_denom
print("\nSlope (m) of the line is: ", m_slope)


# find y intercept of the line, b
b_num = ( df_taxi.withColumn('b_inter', col('trip_dist')**2 ).groupBy().sum().collect()[0][2] \
          * df_taxi.select(col('fare_amt')).groupBy().sum().collect()[0][0] ) - \
        ( df_taxi.select(col('trip_dist')).groupBy().sum().collect()[0][0] * \
          df_taxi.withColumn('b_inter', col('trip_dist') * col('fare_amt')  ).\
          groupBy().sum().collect()[0][2] )

b_denom = ( n_rows * df_taxi.withColumn('b_inter', col('trip_dist')**2 ). \
          groupBy().sum().collect()[0][2] ) - df_taxi.select(col('trip_dist')). \
            groupBy().sum().collect()[0][0] **2

y_inter = b_num / b_denom
print("\ny-intercept (b) of the line is: ", y_inter, "\n")


# time to completion for large dataset, 2 worker nodes = 3.1 hrs
# Slope (m) of the line is:  1.5209074890043822e-06
# y-intercept (b) of the line is:  12.28871667778135