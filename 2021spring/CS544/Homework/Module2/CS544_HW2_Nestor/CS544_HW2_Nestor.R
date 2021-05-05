#Name: Kimberly Nestor 
#Class: CS 544
#Date: 02/02/2021
# Module2 Homework

library(prob)
library(DataCombine)


#### Part 1
#### Question 1A
#Probability that a randomly selected person in this survey will have a BMI of above 30
above.bmi30 = 1062 + 1710 + 656 + 189

prob.rand.above30 = above.bmi30/10000 #P(event) or prob.event
prob.rand.above30

#### BAYES THEOREM
#prob of person in certain age bracket in population of 10,000; 18-34, 35-49, 50-64, 65+
prior = c(4250/10000, 2850/10000, 1640/10000, 1260/10000)
prior

#prob that a person in each age bracket has a bmi over 30
likelihood = c(1062/4250, 1710/2850, 656/1640, 189/1260)
likelihood

#probability of no. of people with bmi above 30 in the whole population of 10,000
prob.event = sum(prior*likelihood)
prob.event

post.prob = c((prior*likelihood)/prob.event)
post.prob

#### Question 1B
#Probability randomly selected person has BMI above 30 and between 18-34 years
post.prob[1]

#### Question 1C
#Probability randomly selected person has BMI above 30 and between 35-49 years
post.prob[2]

#### Question 1D
#Probability randomly selected person has BMI above 30 and between 50-64 years
post.prob[3]

#### Question 1E
#Probability randomly selected person has BMI above 30 and between 65+ years
post.prob[4]


#### Part 2
#Consider a game which involves rolling three dice, with sample and probability space
S = rolldie(3, makespace=TRUE)
#S

#### Question 2A
#sum of rolls is greater than 10
A = subset(S, X1+X2+X3 > 10)
A

#### Question 2B
#all the three rolls are identical
B = subset(S, X1 == X2 & X2 == X3)
B

#### Question 2C
#only two of the three rolls are identical
C = subset(S, X1 == X2 | X2 == X3 | X1 == X3)
C = setdiff(C, B)
C

#### Question 2D
#none of the three rolls are identical
D = setdiff(S, B)
D

#### Question 2E
#only two of the three rolls are identical given  the sum of the rolls greater than 10.
E = intersect(A, C)
E

#### Part 3
#### With Loop
#write func returns sum of squares of first n odd numbers
sum_of_first_N_odd_squares = function(n){
  lst = c()
  count = 1
  while(length(lst) != n) {
    if(count %% 2 == 1) {
      lst = c(lst, count**2)      
    }
    count = count + 1
  }
  sum.squares = sum(lst)
  return(sum.squares)
}

sum_of_first_N_odd_squares(2)
sum_of_first_N_odd_squares(5)
sum_of_first_N_odd_squares(10)


#### Without Loop 
#write func returns sum of squares of first n odd numbers
sum_of_first_N_odd_squares_V2 = function(n) {
  lst = seq(1, 2*n, 2)
  sum.squares = sum(lst**2)
  return(sum.squares)
  
}

sum_of_first_N_odd_squares_V2(2)
sum_of_first_N_odd_squares_V2(5)
sum_of_first_N_odd_squares_V2(10)


#### Part 4
data =  read.csv("http://people.bu.edu/kalathur/datasets/DJI_2020.csv")
data

#### Question 4A
#store result of summary function for Close attribute as sm; change variable names
sm = summary(c(data[["Close"]]))
names(sm)[2] = "Q1"
names(sm)[3] = "Q2"
names(sm)[5] = "Q3"
sm

#using above data show quartile variations for 4 quartiles; print values
variation1 = sm[["Q1"]] - sm[["Min."]]
variation2 = sm[["Q2"]] - sm[["Q1"]]
variation3 = sm[["Q3"]] - sm[["Q2"]]
variation4 = sm[["Max."]] - sm[["Q3"]]

sprintf("First Quartile variation is %.1f", variation1)
sprintf("Second Quartile variation is %.1f", variation2)
sprintf("Third Quartile variation is %.1f", variation3)
sprintf("Fourth Quartile variation is %.1f", variation4)

#### Question 4B
#produce output for the minimum of the Dow closing value in dataset
min.dow = data[c(data$Close == min(data["Close"])), ] 
min.dow

sprintf("The minimum Dow value of %.0f is at row %s on %s", min.dow[,2], 
        rownames(min.dow), min.dow[,1])


#### Question 4C
#invested on minimum date, find date to sell to gain the maximum percentage gain

#subset of dataframe from min dow date onwards
min.row = as.numeric(rownames(min.dow))
df.min.dow = data[min.row:nrow(data),]
df.min.dow

#create vectors with min dow and close dow in subset
per.v1 = replicate(nrow(df.min.dow), min(data[["Close"]]) ) 
per.v2 =c(df.min.dow[["Close"]]) 

#percentage change vector
per.change = ((per.v2 - per.v1)/per.v1) *100
max(per.change)

#make dataframe with per change values
df.perchange <- data.frame(df.min.dow["Date"], per.v1, per.v2, per.change)
df.perchange

#find row in df with max perchange and print
max.perdow = df.perchange[which.max(df.perchange$per.v2),]

sprintf("I would sell on %s when Dow is at %.0f for a gain of %.2f%%", max.perdow[,1], 
        max.perdow[,3], max.perdow[,4])


#### Question 4D
#use diff func, calculate diffs between consecutive closing values, add DIFFS column
data.diff = data
diff.vec = c(0,diff(data.diff[["Close"]]))
data.diff$DIFFS = diff.vec
data.diff

#### Question 4E
#days Dow closed higher than its previous day value
high.days = nrow(data.diff[c(data.diff$DIFFS > 0), ])

#days Dow closed lower than its previous day value?
low.days = nrow(data.diff[c(data.diff$DIFFS < 0), ])

sprintf("%.0f days Dow closed higher than previous day", high.days)
sprintf("%.0f days Dow closed lower than previous day", low.days)

#### Question 4F
#show subset of data where there was a gain of at least 1000 points from previous day
data.diff[c(data.diff$DIFFS >= 1000), ]



#### Quiz
#Question 15
vec1 = c(18, 109, 113, 119, 127, 147, 186)
vec2 = c(14, 108, 110, 142, 144)
isin(vec1, c(18, 109, 113, 119, 127, 147))

oddcount = function(x){
  lst = c()
  for(i in x) {
    if(i %% 2 == 1) {
      lst = c(lst, i)      
    }
  }
  return(length(lst))
}

oddcount(vec1)
oddcount(vec2)


#Question 16
isPalindrome = function(x){
  rev.vec = x[length(x):1]
  return(isin(x, rev.vec, ordered = TRUE))
}
  
isPalindrome(c(32, 40, 86, 40, 32))
isPalindrome(c(32, 40, 40, 32))
isPalindrome(c(29, 63, 88, 94))



