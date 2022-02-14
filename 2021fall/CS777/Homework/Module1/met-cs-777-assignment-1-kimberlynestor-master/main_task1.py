"""Kimberly Nestor
CS777 Big Data Analytics
09/13/2021
Module1 homework Q1
Description: This program takes as input a csv file of taxi driver information
and the routes they drove passengers. The program uses Apache Spark to determine
which cars were the top ten having used by the number of drivers.
"""

from __future__ import print_function

import sys
from operator import add
from pyspark import SparkContext
from pyspark.sql import SparkSession


if __name__ == "__main__":
    # input data as input and ouput dir info
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

    # cleaning up data
    taxilinesCorrected = taxilines.filter(correctRows)

    #get only taxi and driver info
    df_taxi = sp.createDataFrame(taxilinesCorrected)
    df_taxi = df_taxi.select(df_taxi.columns[:2])
    #df_taxi.show()

    rdd_taxi = df_taxi.rdd
    #print(rdd.collect())

    #group by taxi medallion with list of drivers for that taxi
    rdd_taxi = sc.parallelize(sorted(rdd_taxi.groupByKey().mapValues(list).collect())) #list to rdd
    #print(rdd_taxi)

    rdd_drive_num = sc.parallelize(sorted( rdd_taxi.mapValues(lambda x: \
                        len(x)).collect(), key=lambda x: x[1], reverse=True ))

    #get taxis with the top ten most drivers. force output to one file
    taxi_top_ten = sc.parallelize(rdd_drive_num.top(10, key=lambda x: x[1])).coalesce(1)
    #taxi_top_ten = sc.parallelize(rdd_drive_num.take(10))

    taxi_top_ten.saveAsTextFile(sys.argv[2])
    #output = taxi_top_ten.collect()
    sc.stop()


##### LINKS
# script= gs://bigdatamodone/main_task1.py
# output folder= gs://bigdatamodone/output_q1_small
# output folder= gs://bigdatamodone/output_q1_large

# small= gs://metcs777/taxi-data-sorted-small.csv.bz2
# large= gs://metcs777/taxi-data-sorted-large.csv.bz2

