#Name: Kimberly Nestor 
#Class: CS 544
#Date: 01/25/2021
# Module1 Homework


#### Part 1
#sample data showing scores of students of students in an exam
scores = c(59, 46, 76, 60, 49, 65, 82, 68, 99, 52)
scores

#### Question 1A
#How many students took the exam?
length(scores)

#Using indexing, show the expression for accessing the first two items.
scores[1:2]

#Using indexing, show the expression for accessing the first and last items.
scores[c(1,length(scores))]

#Using indexing, show the expression for accessing the middle two items.
scores[c(length(scores)/2,(length(scores)/2)+1)]


#### Question 1B
#Use median(scores) to show the median of the data.
median(scores)

#Using comparison operators, scores less than or equal to median of the data.
scores <= median(scores)

#Using comparison operators, scores greater than  median of the data.
scores > median(scores)

#Using the sum function, number of scores less than or equal to the median of data.
sum( scores <= median(scores) )

#Using the sum function, greater than the median of the data.
sum( scores > median(scores) )


#### Question 1C
#Using logical indexing, all the scores less than or equal to the median value of data
scores[ scores <= median(scores) ]

#Using logical indexing, all the scores greater than or equal to the median.
scores[ scores >= median(scores) ]


#### Question 1D
#Using logical indexing with TRUE and FALSE values, odd indexed values from scores
scores[c(TRUE, FALSE)]

#Using logical indexing with TRUE and FALSE values, even indexed values from scores
scores[c(FALSE, TRUE)]


#### Question 1E
#Using numeric indexing, odd indexed values from the scores
scores[ seq(1, length(scores), by = 2) ]

#Using numeric indexing, even indexed values from the scores
scores[ seq(2, length(scores), by = 2) ]


#### Question 1F
#Create a matrix of size 2 x 5 using the scores data
scores.matrix = matrix(scores, nrow = 2, ncol = 5, byrow = TRUE)
scores.matrix


#### Question 1G
#Display the first and last columns of the matrix
scores.matrix[ , c(1, ncol(scores.matrix) ) ]


#### Question 1H
#Assign column and row names for the scores.matrix 
dimnames(scores.matrix) = list( sprintf("Quiz_%d", seq(1, nrow(scores.matrix))), 
                                sprintf("Student_%d", seq(1, ncol(scores.matrix))) )

scores.matrix


#### Question 1I
#Show the result for displaying  first and last columns of  matrix, same as 1G
scores.matrix[ , c(1, ncol(scores.matrix) ) ]


#### Part 2
#### Question 2A
#Create a data frame, say dow, using the column names: Month, Open, High, Low, and Close
month.name = c("Jan", "Feb", "Mar", "Apr", "May")
month.open = c(28639, 28320, 25591, 21227, 24121)
month.high = c(29374, 29569, 27102, 24765, 24350)
month.low = c(28170, 24681, 18214, 20735, 23361)
month.close = c(28256, 25409, 21917, 24346, 24331)
dow = data.frame(Month = month.name,
                  Open = month.open,
                  High = month.high,
                  Low = month.low,
                  Close = month.close)

dow


#### Question 2B
#Show the result of the summary function for Open, High, Low, and Close.
summary( dow[c("Open", "High", "Low", "Close")] )


#### Question 2C
#Show the data frame sliced using the columns Month, Open, and Close.
dow[c("Month", "Open", "Close")]


#### Question 2D
#Show the data frame sliced using the first and last row
dow[c(1, nrow(dow)), ]


#### Question 2E
#show the data frame sliced using first and last row using cols Month, High and Low
dow[c(1, nrow(dow)), c("Month", "High", "Low")]


#### Question 2F
#Show all rows of data frame whose Low is greater than 22,000; use logical indexing
dow[c(dow$Low > 22000), ]

#Show all rows of data frame whose Low is greater than 22,000; use subset function
subset(dow, Low > 22000)


#### Question 2G
#show all rows of data frame whose Open and Low are both greater than 25,000; use logical indexing
dow[c(dow$Open & dow$Low > 25000), ]

#show all rows of data frame whose Open and Low are both greater than 25,000; use subset function
subset(dow, Open & Low > 25000)


#### Question 2H
#Modify data, add new column Volatility, showing diff between High and Low
dow$Volatility = dow$High - dow$Low
dow

#### Question 2I
#Show the row(s) of data with maximum Volatility, use subset
dow.sortmax = dow[order(-dow$Volatility),]
subset(dow, Volatility == dow.sortmax[["Volatility"]][1])


#Show the row(s) of data with maximum Volatility, use max function
dow[which.max(dow$Volatility),]


#### Question 2J
#Show row(s) of data with minimum Volatility, use logical indexing 
dow.sortmin = dow[order(dow$Volatility),]
dow[c(dow$Volatility == dow.sortmin[["Volatility"]][1]), ]

#Show row(s) of data with minimum Volatility, use min function
dow[which.min(dow$Volatility),]

