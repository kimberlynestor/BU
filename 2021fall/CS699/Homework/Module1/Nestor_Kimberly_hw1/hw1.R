#Name: Kimberly Nestor 
#Class: CS 699
#Date: 11/09/2021
# Module1 Homework


#install.packages("psych")
#install.packages("fastDummies")

library(lsa)
library(psych)
library(dummies)
library(fastDummies)
library(purrr)

data.path = '/Users/kimberlynestor/Desktop/BU/2021fall/CS699/Homework/Module1/Nestor_Kimberly_hw1/'

#### Q 3 
df = read.csv(file.path(data.path, 'a1.csv'))
head(df)
tail(df)

## 3.1 and 3.2 - for attribute A5, calculate 
# mean, median, std
five.num = fivenum( df$A5 )
mean = mean( df$A5 )
median = five.num[3]
std = sd( df$A5 )

# Q1, Q2, Q3
q1 = five.num[2]
q2 = median
q3 = five.num[4]

summary( df$A5 )
std

## 3.3 - find outliers in A5
IQR = q3 - q1

#outliers limits
low.lim = q1 - 1.5 * IQR
up.lim = q3 + 1.5 * IQR

#outliers list
outliers = sort(df[df$A5 < low.lim | df$A5 > up.lim, 'A5'])
outliers

# 3.4 - boxplot of A5 with outliers shown
boxplot(df$A5, pch=16, main="Boxplot of A5") 


#### Q 4 - plot scatter plot of attribute pairs
pairs.panels(df, method = "pearson", # correlation method
                 hist.col = "#00AFBB",
                 density = TRUE,  # show density plots
                 ellipses = TRUE, # show correlation ellipses 
                 lm=TRUE , main="Pairwise Scatterplot of Attributes")

# attributes with the strongest correlation is A2 and A3 with a positive regression 
# slope. this qualitative observation is confirmed by the correlation coefficient 
# of 0.93, which is highly positive and correlative.


#### Q 5 - calculate distance between, d(P9, P8) and d(P9, P10)
df.ppl = read.csv(file.path(data.path, 'q5_data'))
df.ppl

# binarize data
df.ppl.bin = dummy_cols(df.ppl, select_columns = 
                c('ID','job','marital','education','default','housing','loan',
                  'contact'))[,9:37]
# make matrix
dist.mtx = as.matrix(dist(df.ppl.bin, method = 'binary'))
dist.mtx

# distance for P9 and P8
dist.p9.p8 = dist.mtx[9,8]
dist.p9.p8
# distance for P9 and P10
dist.p9.p10 = dist.mtx[9,10]
dist.p9.p10


#### Q 6 - calculate cosine similarity between, cos(D2, D1) and cos(D2, D3)
df.fruit = read.csv(file.path(data.path, 'q6_data'))
df.fruit
# convert row to vector, drop nan
d1 = discard( as.numeric(as.vector(df.fruit[1,])) , is.na) 
d2 = discard( as.numeric(as.vector(df.fruit[2,])) , is.na) 
d3 = discard( as.numeric(as.vector(df.fruit[3,])) , is.na) 

# find cosine similarity
cosine.d2.d1 = cosine(d1, d2)
cosine.d2.d3 = cosine(d1, d3)

cosine.d2.d1
cosine.d2.d3

# D2 is closer to D1 than it is to D3, because the cosine.d2.d1 is 0.67 compared 
# to the lower value for cosine.d2.d3 = 0.63. 
# Conclusion: d2.d1 are more similar.


#


## QUIZ
v1 = c(35, 27, 14, 76, 84, 95)
mean(v1)
sd(v1)


# find distance between two values
O1 = c('high', 'low', 0, 'yes', 'P', 'N', 'old')
O2 = c('high', 'high', 2, 'yes', 'N', 'N', 'young')

df.q = data.frame(O1, O2)

df.q.bin = dummy_cols(df.q, select_columns = c('O1','O2','marital'))[,3:14]
df.q.bin

# make matrix
dist.mtx.q = as.matrix(dist(df.q.bin, method = 'binary'))
dist.mtx.q


# find cosine similarity between two values
O1 = c(3, 2, 0, 2)
O2 = c(1, 4, 2, 0)

cosine.q = cosine(O1, O2)
cosine.q


#find outliers
IQR = 165 - 125

#outliers limits
low.lim = 125 - 1.5 * IQR
up.lim = 165 + 1.5 * IQR
low.lim
up.lim


