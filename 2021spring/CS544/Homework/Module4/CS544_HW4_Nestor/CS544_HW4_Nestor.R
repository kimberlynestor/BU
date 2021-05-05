#Name: Kimberly Nestor 
#Class: CS 544
#Date: 02/18/2021
# Module4 Homework

library(distr) 
#X = Binom(size=n, prob=p)
#d(X)(2)

path = '/Users/kimberlynestor/Desktop/BU/2021spring/CS544/Homework/Module4/'

#### Part 1
#BINOMIAL DISTRIBUTION
n = 5 #given attempts.
p = 0.4 #chance, perfect score

#### Question 1A
#Compute and plot prob distribution for num perfect scores over 5 attempts - PMF
pmf.bd = dbinom(0:n, size=n, prob=p)
pmf.bd

png(file.path(path, 'Q1A_spikeplot_perfscores.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(0:n, pmf.bd, type="h", xlab="No. perfect scores", ylab="PMF", 
     main="Spike plot of X") #plot
points(0:n, pmf.bd, pch=16) 
dev.off() #close file

#Compute and plot prob distribution for num perfect scores over 5 attempts - CDF
cdf.bd = pbinom(0:n, size=n, prob=p) #cdf = cumsum(pmf)
cdf.bd

png(file.path(path, 'Q1A_stepplot_perfscores.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
cdfplot = stepfun(0:n, c(0, cdf.bd)) #steps
plot(cdfplot, verticals=FALSE, pch=16, xlab="No. perfect scores", ylab="CDF", 
     main="Step plot of X") #plot
dev.off() #close file

#### Question 1B
#Prob student will score a perfect score in exactly 2 out of the 5 attempts
x = 2

prob.2attempt.r = dbinom(2, size=n, prob=p)
prob.2attempt.r

prob.2attempt.math = choose(n, x) * p^x * (1 - p)^(n-x)
prob.2attempt.math

#### Question 1C
#Prob student will score a perfect score in at least 2 out of the 5 attempts
prob.2or.more = pbinom(1, size=n, prob=p, lower.tail=FALSE) #1 - P(X <= 1) or 1 - cdf(1)
prob.2or.more

#### Question 1D
#Simulate num of perfect scores over 5 attempts for 1000 students 
sim.binom = rbinom(1000, size=n, prob=p)
freq.sim = table(sim.binom)
freq.sim

#Barplot of frequencies of successes
png(file.path(path, 'Q1D_simfreq_perfscores.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
barplot(freq.sim, xlab="Attempts", ylab="Frequency", main="Simulated binome") #plot  
dev.off() #close file


#### Part 2
#NEGATIVE BINOMIAL DISTRIBUTION
p = 0.6 #prob
r = 3 #success

#### Question 2A
#Compute and plot prob distribution for scoring, three perfect scores, max of 10 failures - PMF 
xmax = 10 #maximum failures

pmf.nbd = dnbinom(0:xmax, size=r, prob=p)
pmf.nbd

png(file.path(path, 'Q2A_spikeplot_perfscores_nbinom.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(0:xmax, pmf.nbd, type="h", xlab="Failures", ylab="PMF", 
     main="Spike plot of X") #plot
points(0:xmax, pmf.nbd, pch=16) 
dev.off() #close file

#Compute and plot prob distribution for num perfect scores over 5 attempts - CDF
cdf.nbd = pnbinom(0:xmax, size=r, prob=p) #cdf = cumsum(pmf)
cdf.nbd

png(file.path(path, 'Q2A_stepplot_perfscores_nbinom.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
cdfplot = stepfun(0:xmax, c(0, cdf.nbd)) #steps
plot(cdfplot, verticals=FALSE, pch=16, xlab="Failures", ylab="CDF", 
     main="Step plot of X") #plot
dev.off() #close file

#### Question 2B
#Prob student will have three perfect scores with exactly 4 failures
x = 4 #failures

prob.4fail.r = dnbinom(x, size=r, prob=p)
prob.4fail.r

prob.4fail.math = choose((r + x -1), (r - 1)) * p^r * (1-p)^x
prob.4fail.math

#### Question 2C
#Prob student will have three perfect scores with at most 4 failures
prob.at.most4 = pnbinom(x, size=r, prob=p) #1 - P(X <= 4) or 1 - cdf(4)
prob.at.most4

#### Question 2D
#Simulate num of failures to get three perfect scores for 1000 students
sim.nbinom = rnbinom(1000, size=r, prob=p)
freq.sim = table(sim.nbinom)
freq.sim

#Barplot of frequencies of failures
png(file.path(path, 'Q2D_simfreq_perfscores_nbinom.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
barplot(freq.sim, xlab="Failures", ylab="Frequency", main="Simulated nbinome") #plot  
dev.off() #close file



#### Part 3
#HYPERGEOMETRIC DISTRIBUTION
M = 60 #multiple choice
N = 40 #programming questions
K = 20

#### Question 3A
#Compute and plot prob distribution of 0, 1,..., k multiple choice questions - PMF
pmf.hd = dhyper(0:K, m = M, n = N, k = K) 
pmf.hd

png(file.path(path, 'Q3A_spikeplot_multchoice.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(0:K, pmf.hd, type="h", xlab="No. multiple choice questions", ylab="PMF", 
     main="Spike plot of X") #plot
points(0:K, pmf.hd, pch=16) 
dev.off() #close file

#Compute and plot prob distribution of 0, 1,..., k multiple choice questions - CDF
cdf.hd = phyper(0:K, m = M, n = N, k = K) #cdf 
cdf.hd

png(file.path(path, 'Q3A_stepplot_multchoice.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
cdfplot = stepfun(0:K, c(0, cdf.hd)) #steps
plot(cdfplot, verticals=FALSE, pch=16, xlab="No. multiple choice questions", ylab="CDF", 
     main="Step plot of X") #plot
dev.off() #close file


#### Question 3B
#Prob student will have exactly 10 multiple choice questions out of 20 
x = 10

prob.10multchoice.r = dhyper(10, m = M, n = N, k = K) 
prob.10multchoice.r

prob.2attempt.math = (choose(M, x) * choose(N, K-x)) / choose(M+N, K)
prob.2attempt.math

#### Question 3C
#Prob student will have at least 10 multiple choice questions out of 20
prob.10or.more = phyper(9, m = M, n = N, k = K, lower.tail=FALSE) 
prob.10or.more

#### Question 3D
#Simulate num of multiple choice questions for 1000 students
sim.hyper = rhyper(1000, m = M, n = N, k = K)
freq.sim = table(sim.hyper)
freq.sim

#Barplot of frequencies of multiple choice questions
png(file.path(path, 'Q3D_simfreq_multchoice.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
barplot(freq.sim, xlab="No. multiple choice questions", ylab="Frequency", 
        main="Simulated hyper", ylim=c(0, 200)) #plot  
dev.off() #close file


#### Part 4
#POISSON DISTRIBUTION
lam = 10 #no. of student emails per day

#### Question 4A
#Prob the prof will have to answer exactly 8 questions per day
prob.8ans = dpois(8, lambda = lam)
prob.8ans

#### Question 4B
#Prob the prof will have to answer at most 8 questions per day
prob.at.most8 = ppois(8, lambda = lam)
prob.at.most8

#### Question 4C
#Prob the prof will have to answer between 6 and 12 questions (inclusive)
prob.betw.6and12 = diff(ppois(c(5, 12), lambda = lam))
prob.betw.6and12

#### Question 4D
#Calculate and plot PMF for first 20 questions
pmf.pd = dpois(0:20, lambda = lam)
pmf.pd

png(file.path(path, 'Q4D_spikeplot_pois.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(0:20, pmf.pd, type="h", xlab="Questions per day", ylab="PMF", 
     main="Spike plot of X") #plot
points(0:20, pmf.pd, pch=16) 
dev.off() #close file

#### Question 4E
#Simulate num of questions prof gets per day over the course run, 50 days
sim.pois = rpois(50, lambda=lam)
freq.sim = table(sim.pois)
freq.sim

#Barplot of frequencies of number of questions
png(file.path(path, 'Q4E_simfreq_pois_barplot.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
barplot(freq.sim, xlab="Questions", ylab="Frequency", 
        main="Simulated pois") #plot  
dev.off() #close file

#Boxplot of number of questions

png(file.path(path, 'Q4E_simfreq_pois_boxplot.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
boxplot(sim.pois, main="Questions to Prof per day") #plot
dev.off() #close file

#Inference from barplot and boxplot
"As noted in the question, the professor typically receives 10 emails per day
and the poisson PMF is based on this value as the lambda. Therefore in the 
simulated values barplot, the professor is shown to receive the most emails 
in the frequency of 10, 11, 12 emails per day. There are far fewer simulated 
emails received in frequencies much lower or higher than the original lambda 
input to the pois function. 

The boxplot shows a similar trend to that noted prior, 
the middle 50% of the data is located between frequencies of 8 to 12, suggesting 
that the professor has a higher probability of receiving emails within that range 
per day throughout the course. The whiskers of the plot extend to frequencies of 
5 and 16, suggesting there were fewer days where the professor received these 
number of emails.

Essentially the barplot and boxplot are qualitative confirmations of assumptions
that could have been made by analysing the PMF and CDF probability outputs. 
The assumption being that the professor will be more likely to receive emails 
at a higher probability closer to the original lambda input value."


#### Part 5
#NORMAL DISTRIBUTION
mu = 100
std = 10
dstr = seq((mu-(std*3)), (mu+(std*3)))

#### Question 5A
#Plot PDF of distribution covering three standard deviations, either side of mean
pdf.nd = dnorm(dstr, mean = mu, sd = std)

png(file.path(path, 'Q5A_distrib_3std_norm.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(dstr, pdf.nd, type="l", col="red", xlab="Money", ylab="PDF", 
     main="Amt. spend on souvenirs") #plot
dev.off() #close file


#### Question 5B
#Chance randomly selected visitor will spend more than $120
prob.more.120 = pnorm(119, mean = mu, sd = std, lower.tail=FALSE) 
prob.more.120

#### Question 5C
#Chance a randomly selected visitor will spend between $80 and $90 (inclusive)
prob.betw.80.90 = diff(pnorm(c(79, 90), mean = mu, sd = std))
prob.betw.80.90

#### Question 5D
#Chance of spending within one standard deviation
prob.1std = pnorm(mu + std, mean=mu, sd=std) - pnorm(mu - std, mean=mu, sd=std)
prob.1std

#Chance of spending within two standard deviation
prob.2std = pnorm(mu + 2*std, mean=mu, sd=std) - pnorm(mu - 2*std, mean=mu, sd=std)
prob.2std

#Chance of spending within three standard deviation
prob.3std = pnorm(mu + 3*std, mean=mu, sd=std) - pnorm(mu - 3*std, mean=mu, sd=std)
prob.3std

#### Question 5E
#What two values are the range for the middle 90% of the money spent fall
mid.90.bound = c(qnorm(0.05, mean = mu, sd = std), qnorm(0.95, mean = mu, sd = std))
mid.90.bound

#### Question 5F
#If theme park gives free T-shirt for top 5% spenders, min amt spent to get free T-shirt
top.5.bound = qnorm(0.95, mean = mu, sd = std)
top.5.bound

#### Question 5G
#Plot for 10,000 visitors using the above distribution, money on souvenirs
sim.norm = round(rnorm(10000, mean = mu, sd = std))

png(file.path(path, 'Q5G_sim_distrib_money.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
plot(table(sim.norm), xlab="Money spent", ylab="Frequency", 
     main="Amt. spend on souvenirs") #plot
dev.off() #close file


#QUIZ
"Suppose the scores of an exam follow a normal distribution with mean = 65 and 
standard deviation = 5. What is the chance that a randomly selected student will 
score at least 67? (round up the answer to nearest integer)"
pnorm(67, mean = 65, sd = 5, lower.tail=FALSE) 

"Suppose the scores of an exam follow a normal distribution with mean = 70 and 
standard deviation = 6. What is the chance that a randomly selected student will 
score in between 58 and 78? (round up the answer to nearest integer)"
diff(pnorm(c(58, 78), mean=70, sd=6))

"Suppose you go to bed at 10 PM and wake up at 6 AM and check your email the 
first thing after waking up. On an average, your inbox receives 135 emails during 
that period. What is the probability that you will have in between 129 and 136 
emails (both inclusive) to read? (round the answer to three decimals)"
diff(ppois(c(128, 136), lambda=135))

"In a game of chutes and ladders, suppose you need a 5 to win the game. You are 
rolling a 6-faced fair die. What is the probability that you will win the game 
in at most 6 tries? (round the answer to four decimals)"
prob = 0.1666667
pnbinom(0:6, size=1, prob=prob)

"Consider a slot machine with 3 wheels (reels). Each reel has 10 slots, 0, 1, …, 9. 
Suppose you have a win if the 3 slots match (0-0, 1-1, …, 9-9-9). You decided to 
play until you win 3 times. What is the probability that you will walk out after 
exactly 295 tries?"
dnbinom(295, size=3, prob=0.01) #10/10^3

"At the carnival you decided to play the balloon and dart game to pop the 
balloons with the darts. You are provided 5 darts. Your chance of popping a 
balloon with a dart is 63%. What is the probability that you will pop atmost 
4 balloons with the 5 darts."
pbinom(4, size=5, prob=0.62)

"At the carnival you decided to play the balloon and dart game to pop the 
balloons with the darts. You are provided 5 darts. Your chance of popping a 
balloon with a dart is 79%. What is the probability that you will pop at least 
3 balloons with the 5 darts."
pbinom(2, size=5, prob=0.79, lower.tail=FALSE)

"A fair coin (Heads/Tails) is tossed 13 times. What is the probability that 
you will get at least 4 Heads?"
pbinom(3, size=13, prob=0.5, lower.tail=FALSE)

"An M&M candy dispenser has 25 red, 25 blue and 25 green M&Ms. The dispenser 
dispenses 20 M&Ms each time. What is the probability, that on the first use, 
you will get 10 blue M&Ms out of the 20 dispensed."
dnbinom(1, size=10, prob=0.4) #10/25 prob selecting 10 blue from 25

pnbinom(1, size=9, prob=0.3) #10/25 prob selecting 10 blue from 25


