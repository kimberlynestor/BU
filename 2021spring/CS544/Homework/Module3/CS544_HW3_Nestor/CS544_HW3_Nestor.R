#Name: Kimberly Nestor 
#Class: CS 544
#Date: 02/12/2021
# Module3 Homework

library()
library(formattable)

path = '/Users/kimberlynestor/Desktop/BU/2021spring/CS544/Homework/Module3/'

#### Part 1
#dataframe of primes below 10,000
df = read.csv("http://people.bu.edu/kalathur/datasets/myPrimes.csv")
head(df)
tail(df)

#### Question 1A
#Barplot of the frequencies for the last digit
df.freq.ld = table(df['LastDigit']) #find frequencies
df.freq.ld
png(file.path(path, 'Q1A_barplot_freq.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 300) #save plot
barplot(df.freq.ld, xlab = "Last Digit", ylab = "Frequency") #plot
dev.off() #close file

#### Question 1B
#Barplot of the frequencies for the first digit
df.freq.fd = table(df['FirstDigit']) #find frequencies
df.freq.fd
png(file.path(path, 'Q1B_barplot_freq.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 300) #save plot
barplot(df.freq.fd, xlab = "First Digit", ylab = "Frequency") #plot
dev.off() #close file

#### Question 1C
#inferences drawn from the two plots


#### Part 2
#Dataset, number of quarters minted for 50 US states, by DenverMint and PhillyMint
us_quarters = read.csv("http://people.bu.edu/kalathur/datasets/us_quarters.csv")
head(us_quarters)

#### Question 2A
#State producing highest num quarters, DenverMint
us_quarters[which.max(us_quarters$DenverMint), 1:2]

#State producing highest num quarters, PhillyMint
us_quarters[which.max(us_quarters$PhillyMint), c('State', 'PhillyMint')]

#State producing lowest num quarters, DenverMint
us_quarters[which.min(us_quarters$DenverMint), 1:2]

#State producing lowest num quarters, PhillyMint
us_quarters[which.min(us_quarters$PhillyMint), c('State', 'PhillyMint')]

#### Question 2B
#Value of the total coins in dollars
doll.dm = currency(sum(us_quarters$DenverMint)*0.25, digits = 0L)
doll.pm = currency(sum(us_quarters$PhillyMint)*0.25, digits = 0L)
tot.doll = sum(doll.dm, doll.pm)

sprintf("Value of total coins in dollars from DenverMint is %s", toString(doll.dm))
sprintf("Value of total coins in dollars from PhillyMint is %s", toString(doll.pm))
sprintf("Value of total coins in dollars from DenverMint and PhillyMint is %s", 
        toString(tot.doll))

#### Question 2C
#Make barplot from mint data as matrix
png(file.path(path, 'Q2C_barplot_state.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 300) #save plot
mtrx <- rbind(DenverMint, PhillyMint) #matrix
barplot(mtrx, beside=T, legend=T, col=c("red","blue")) #plot
dev.off() #close file

#Write any two striking inferences from plot
plot(DenverMint, PhillyMint, pch=16, col="#69b3a2")

#### Question 2D
#Show scatter plot of num coins between two mints
#options(scipen=999) #change from scientific notation
png(file.path(path, 'Q2D_sccatterplot_coins.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 300) #save plot
plot(DenverMint, PhillyMint, pch=16, col="#69b3a2", main="Scatterplot")
dev.off() #close file

#Write any two striking inferences from plot

#### Question 2E
#Show side-by-side box plots for two mints
png(file.path(path, 'Q2E_boxplot_coins.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 300) #save plot
boxplot(DenverMint, PhillyMint, col=c("red","blue"), legend=T, 
        names=c("DenverMint", "PhillyMint"), pch=16) #plot
dev.off() #close file

#Write any two striking inferences from plot

#### Question 2F
#Using fivenum(), what states would be outliers for each of the two mints
head(us_quarters)
#min, Q1, median, Q3, max
five.num = fivenum(us_quarters$DenverMint) 
IQR = five.num[4] - five.num[2]
#outliers
low_lim = five.num[2] - 1.5 * IQR
up_lim = five.num[4] + 1.5 * IQR

#outlier states for DenverMint
outliers.dm = us_quarters[us_quarters$DenverMint < low_lim | 
                            us_quarters$DenverMint > up_lim, 1:2]
outliers.dm

#outlier states for PhillyMint
outliers.pm = us_quarters[us_quarters$PhillyMint < low_lim | 
                            us_quarters$PhillyMint > up_lim, c('State', 'PhillyMint')]
outliers.pm


#### Part 3
#FAANG stocks dataset with the April daily High values
stocks = read.csv("http://people.bu.edu/kalathur/datasets/faang.csv")
stocks

#### Question 3A
#Show pair wise plots, all 5 stocks in dataset in a single plot
png(file.path(path, 'Q3A_pairplot.jpg'), width = 4, height = 4.5, 
    units = 'in', res = 600) #save plot
plot(stocks[2:6], pch=16, col="#69b3a2") #plot
dev.off() #close file

#### Question 3B
#Show correlation matrix for 5 stocks in dataset
corr.mtrx = round(cor(stocks[2:6]), 2)
corr.mtrx

#### Question 3C
#Provide at least 4 interpretations of results


#### Part 4
#dataframe of scores
scores = read.csv("http://people.bu.edu/kalathur/datasets/scores.csv")

#### Question 4A
#Default hist, use counts and breaks, output scores and ranges

#make deafult histogram and save image 
attach(scores)
png(file.path(path, 'Q4A_hist_scores.jpg'), width = 4, height = 4, 
    units = 'in', res = 300) #save plot
hist.scores = hist(Score)
dev.off() #close file

#list of break values in histogram
hist.breaks = hist.scores$breaks
hist.breaks

#read break values as a range, with number of students in range
hbr1 = hist.breaks[1:length(hist.breaks)-1]
hbr2 = hist.breaks[2:length(hist.breaks)]

hist.counts = hist.scores$counts
hist.counts

hist.zip = as.list(paste(hbr1, hbr2, hist.counts))

#print output
writeLines(sprintf("%s students in range (%s, %s]", substring(hist.zip, 7), 
                   substring(hist.zip, 1, 2), substring(hist.zip, 4, 5)))


#### Question 4B
#Default hist, use counts and breaks, num students who get A, B, C

#find index values
idx.50 = as.numeric(match(50, hist.breaks))
idx.70 = as.numeric(match(70, hist.breaks))
idx.90 = as.numeric(match(90, hist.breaks))

#find number of students
studs.30.50 = sum(hist.counts[1:idx.50-1])
studs.50.70 = sum(hist.counts[idx.50:idx.70-1])
studs.70.90 = sum(hist.counts[idx.70:length(hist.counts)])

# print output
sprintf("%s students in C grade range (30, 50]", studs.30.50)
sprintf("%s students in B grade range (50, 70]", studs.50.70)
sprintf("%s students in A grade range (70, 90]", studs.70.90)


