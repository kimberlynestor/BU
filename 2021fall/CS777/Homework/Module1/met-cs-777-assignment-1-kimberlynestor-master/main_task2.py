"""Kimberly Nestor
CS777 Big Data Analytics
09/13/2021
Module1 homework Q2
Description: This program takes as input a csv file of taxi driver information
and the routes they drove passengers. The program uses Apache Spark to determine
which drivers were the top ten earners in hourly pay on average.
"""

from __future__ import print_function

import sys
from operator import add

from pyspark import SparkContext

# conda create -n pyspark_env
# conda activate pyspark_env
# conda install pyspark

import sys
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import mean, desc
from pyspark.mllib.stat import Statistics
import numpy as np

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: wordcount <file> <output> ", file=sys.stderr)
        exit(-1)

    # init
    sc = SparkContext()
    sp = SparkSession.builder.getOrCreate()

    # obtain data from inputs
    lines = sc.textFile(sys.argv[1], 1)
    taxilines = lines.map(lambda x: x.split(","))


    # Exception Handling and removing wrong data lines
    def isfloat(value):
        try:
            float(value)
            return True
        except:
            return False

    # For example, remove lines if they don’t have 16 values and ...
    def correctRows(p):
        if(len(p) == 17):
            if (isfloat(p[5]) and isfloat(p[11])):
                if (float(p[5]) != 0 and float(p[11]) != 0):
                    return p

    # cleaning up data, get important columns
    taxilinesCorrected = taxilines.filter(correctRows)

    taxi_time_money = sc.parallelize(taxilinesCorrected.map(lambda i: (i[1], i[4], i[16])).collect())
    taxi_time = sc.parallelize(taxilinesCorrected.map(lambda i: (i[1], i[4])).collect())
    taxi_money = sc.parallelize(taxilinesCorrected.map(lambda i: (i[1], i[16])).collect())

    # get only driver, time and money data
    df_taxi = sp.createDataFrame(taxilinesCorrected)
    columns = ["_2", "_5", "_17"]
    df_money = df_taxi.select(*columns)

    # find cost of rides per minute
    df_money = df_money.withColumn('per_min',
                                   df_money["_17"] / (df_money["_5"] / 60))
    df_money = df_money.select(["_2", "per_min"])
    # df_money.show()

    top_money_drivers = df_money.groupBy("_2").agg(mean("per_min").alias("avg_hr_pay"))\
        .orderBy(desc("avg_hr_pay")).take(10)
    top_money_drivers = sc.parallelize(top_money_drivers).coalesce(1)
    # print(top_money_drivers)

    top_money_drivers.saveAsTextFile(sys.argv[2])
    sc.stop()




    """
    # money_min = sc.parallelize(taxi_time_money.map(lambda i: (i[0], int(i[2]) /(int(i[1])/60) ) if i[2] != 0 else int(i[1])/60).collect())
    driver_time_money = (taxi_time_money.map(lambda x: (x, 0)).reduceByKey(add)).sortBy(lambda x: x[0]).collect()
    driver_time = (taxi_time.map(lambda x: (x, 1)).reduceByKey(add)).sortBy(lambda x: x[0]).collect()
    driver_money = (taxi_money.map(lambda x: (x, 1)).reduceByKey(add)).sortBy(lambda x: x[0]).collect()
    # print(driver_time_money[:5])
    # print(driver_time[:5])
    # print(driver_money[:5])
    """

##### LINKS
# script= gs://bigdatamodone/main_task2.py
# output folder= gs://bigdatamodone/output_q2_small
# output folder= gs://bigdatamodone/output_q2_large

# small= gs://metcs777/taxi-data-sorted-small.csv.bz2
# large= gs://metcs777/taxi-data-sorted-large.csv.bz2

