"""
Name: Kimberly Nestor
Class: CS 677 - Spring 2
Date: 03/23/21
Homework #1
Description of Problem: This program analyses two datasets of stock market
trading values. Stock examined are identified by the tickers NERV and SPY.
Use of numpy and pandas is not allowed.
"""

# import sys
import stock_funcs as sf

#### QUESTION1
# all year daily returns (M-F) for R = {r1, . . . , rn}
nerv_return_mon_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Monday', \
                                        'NERV'))) for i in range(2016, 2021)]
nerv_return_tues_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Tuesday', \
                                        'NERV'))) for i in range(2016, 2021)]
nerv_return_wed_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Wednesday', \
                                        'NERV'))) for i in range(2016, 2021)]
nerv_return_thurs_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Thursday', \
                                        'NERV'))) for i in range(2016, 2021)]
nerv_return_fri_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Friday', \
                                        'NERV'))) for i in range(2016, 2021)]

# Q1 Part1 - for yrs, mean and std, sets R, R− and R+, daily returns, NERV, M-F
# all year mean - μ(R)
muR_return_mon_allyr = [sf.mean(i) for i in nerv_return_mon_all]
muR_return_tues_allyr = [sf.mean(i) for i in nerv_return_tues_all]
muR_return_wed_allyr = [sf.mean(i) for i in nerv_return_wed_all]
muR_return_thurs_allyr = [sf.mean(i) for i in nerv_return_thurs_all]
muR_return_fri_allyr = [sf.mean(i) for i in nerv_return_fri_all]

# all year std - σ(R)
stdR_return_mon_allyr = [sf.std(i) for i in nerv_return_mon_all]
stdR_return_tues_allyr = [sf.std(i) for i in nerv_return_tues_all]
stdR_return_wed_allyr = [sf.std(i) for i in nerv_return_wed_all]
stdR_return_thurs_allyr = [sf.std(i) for i in nerv_return_thurs_all]
stdR_return_fri_allyr = [sf.std(i) for i in nerv_return_fri_all]


# all years NEGATIVE daily returns
nerv_return_mon_allyr_neg = [[j for j in list if j<0] for list in nerv_return_mon_all]
nerv_return_tues_allyr_neg = [[j for j in list if j<0] for list in nerv_return_tues_all]
nerv_return_wed_allyr_neg = [[j for j in list if j<0] for list in nerv_return_wed_all]
nerv_return_thurs_allyr_neg = [[j for j in list if j<0] for list in nerv_return_thurs_all]
nerv_return_fri_allyr_neg = [[j for j in list if j<0] for list in nerv_return_fri_all]

# number of neg daily returns, |R−|
len_mon_allyr_neg = [len(i) for i in nerv_return_mon_allyr_neg]
len_tues_allyr_neg = [len(i) for i in nerv_return_tues_allyr_neg]
len_wed_allyr_neg = [len(i) for i in nerv_return_wed_allyr_neg]
len_thurs_allyr_neg = [len(i) for i in nerv_return_thurs_allyr_neg]
len_fri_allyr_neg = [len(i) for i in nerv_return_fri_allyr_neg]

# all year, negative mean, μ(R−)
muRneg_return_mon_allyr = [sf.mean(i) for i in nerv_return_mon_allyr_neg]
muRneg_return_tues_allyr = [sf.mean(i) for i in nerv_return_tues_allyr_neg]
muRneg_return_wed_allyr = [sf.mean(i) for i in nerv_return_wed_allyr_neg]
muRneg_return_thurs_allyr = [sf.mean(i) for i in nerv_return_thurs_allyr_neg]
muRneg_return_fri_allyr = [sf.mean(i) for i in nerv_return_fri_allyr_neg]

# all year, negative std, σ(R−)
stdRneg_return_mon_allyr = [sf.std(i) for i in nerv_return_mon_allyr_neg]
stdRneg_return_tues_allyr = [sf.std(i) for i in nerv_return_tues_allyr_neg]
stdRneg_return_wed_allyr = [sf.std(i) for i in nerv_return_wed_allyr_neg]
stdRneg_return_thurs_allyr = [sf.std(i) for i in nerv_return_thurs_allyr_neg]
stdRneg_return_fri_allyr = [sf.std(i) for i in nerv_return_fri_allyr_neg]


# all years POSITIVE daily returns
nerv_return_mon_allyr_pos = [[j for j in list if j>0] for list in nerv_return_mon_all]
nerv_return_tues_allyr_pos = [[j for j in list if j>0] for list in nerv_return_tues_all]
nerv_return_wed_allyr_pos = [[j for j in list if j>0] for list in nerv_return_wed_all]
nerv_return_thurs_allyr_pos = [[j for j in list if j>0] for list in nerv_return_thurs_all]
nerv_return_fri_allyr_pos = [[j for j in list if j>0] for list in nerv_return_fri_all]

# number of pos daily returns, |R+|
len_mon_allyr_pos = [len(i) for i in nerv_return_mon_allyr_pos]
len_tues_allyr_pos = [len(i) for i in nerv_return_tues_allyr_pos]
len_wed_allyr_pos = [len(i) for i in nerv_return_wed_allyr_pos]
len_thurs_allyr_pos = [len(i) for i in nerv_return_thurs_allyr_pos]
len_fri_allyr_pos = [len(i) for i in nerv_return_fri_allyr_pos]

# all year, positive mean, μ(R+)
muRpos_return_mon_allyr = [sf.mean(i) for i in nerv_return_mon_allyr_pos]
muRpos_return_tues_allyr = [sf.mean(i) for i in nerv_return_tues_allyr_pos]
muRpos_return_wed_allyr = [sf.mean(i) for i in nerv_return_wed_allyr_pos]
muRpos_return_thurs_allyr = [sf.mean(i) for i in nerv_return_thurs_allyr_pos]
muRpos_return_fri_allyr = [sf.mean(i) for i in nerv_return_fri_allyr_pos]

# all year, positive std, σ(R+)
stdRpos_return_mon_allyr = [sf.std(i) for i in nerv_return_mon_allyr_pos]
stdRpos_return_tues_allyr = [sf.std(i) for i in nerv_return_tues_allyr_pos]
stdRpos_return_wed_allyr = [sf.std(i) for i in nerv_return_wed_allyr_pos]
stdRpos_return_thurs_allyr = [sf.std(i) for i in nerv_return_thurs_allyr_pos]
stdRpos_return_fri_allyr = [sf.std(i) for i in nerv_return_fri_allyr_pos]

# list of all return data, all year, row=diff_headers and col=year
all_mon_return = [muR_return_mon_allyr, stdR_return_mon_allyr, \
                  len_mon_allyr_neg, muRneg_return_mon_allyr, stdRpos_return_mon_allyr, \
                  len_mon_allyr_pos, muRpos_return_mon_allyr, stdRpos_return_mon_allyr]
all_tues_return = [muR_return_tues_allyr, stdR_return_tues_allyr, \
                  len_tues_allyr_neg, muRneg_return_tues_allyr, stdRpos_return_tues_allyr, \
                  len_tues_allyr_pos, muRpos_return_tues_allyr, stdRpos_return_tues_allyr]
all_wed_return = [muR_return_wed_allyr, stdR_return_wed_allyr, \
                  len_wed_allyr_neg, muRneg_return_wed_allyr, stdRpos_return_wed_allyr, \
                  len_wed_allyr_pos, muRpos_return_wed_allyr, stdRpos_return_wed_allyr]
all_thurs_return = [muR_return_thurs_allyr, stdR_return_thurs_allyr, \
                  len_thurs_allyr_neg, muRneg_return_thurs_allyr, stdRpos_return_thurs_allyr, \
                  len_thurs_allyr_pos, muRpos_return_thurs_allyr, stdRpos_return_thurs_allyr]
all_fri_return = [muR_return_fri_allyr, stdR_return_fri_allyr, \
                  len_fri_allyr_neg, muRneg_return_fri_allyr, stdRpos_return_fri_allyr, \
                  len_fri_allyr_pos, muRpos_return_fri_allyr, stdRpos_return_fri_allyr]

# transpose so row=year and col=diff_headers
transpose_all_mon_return = [list(map(lambda xx: xx[i], all_mon_return)) \
                            for i in range(len(all_mon_return[0]))]
transpose_all_tues_return = [list(map(lambda xx: xx[i], all_tues_return)) \
                            for i in range(len(all_tues_return[0]))]
transpose_all_wed_return = [list(map(lambda xx: xx[i], all_wed_return)) \
                            for i in range(len(all_wed_return[0]))]
transpose_all_thurs_return = [list(map(lambda xx: xx[i], all_thurs_return)) \
                            for i in range(len(all_thurs_return[0]))]
transpose_all_fri_return = [list(map(lambda xx: xx[i], all_fri_return)) \
                            for i in range(len(all_fri_return[0]))]

#data for only 2016, row=day and col=R data
week_all_return_2016 = [transpose_all_mon_return[0], transpose_all_tues_return[0], \
                        transpose_all_wed_return[0], transpose_all_thurs_return[0], \
                        transpose_all_fri_return[0]]
#data for only 2017, row=day and col=R data
week_all_return_2017 = [transpose_all_mon_return[1], transpose_all_tues_return[1], \
                        transpose_all_wed_return[1], transpose_all_thurs_return[1], \
                        transpose_all_fri_return[1]]
#data for only 2018, row=day and col=R data
week_all_return_2018 = [transpose_all_mon_return[2], transpose_all_tues_return[2], \
                        transpose_all_wed_return[2], transpose_all_thurs_return[2], \
                        transpose_all_fri_return[2]]
#data for only 2019, row=day and col=R data
week_all_return_2019 = [transpose_all_mon_return[3], transpose_all_tues_return[3], \
                        transpose_all_wed_return[3], transpose_all_thurs_return[3], \
                        transpose_all_fri_return[3]]
#data for only 2020, row=day and col=R data
week_all_return_2020 = [transpose_all_mon_return[4], transpose_all_tues_return[4], \
                        transpose_all_wed_return[4], transpose_all_thurs_return[4], \
                        transpose_all_fri_return[4]]


# Q1 Part2 - summarize mean, std and tot num using tables for each year
#### 2 0 1 6
# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
# format_info = "{:<16.5}" * (len(tab_headers) )
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers - 2016
print("DAILY RETURN VALUES: 2016")
print(format_info.format("Day", *tab_headers))
# print table with yr2016 daily return values, mean and std
for day, val in zip(wk_days, week_all_return_2016):
    print(format_info.format(day, *val))
print("\n")

#### 2 0 1 7
# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers - 2017
print("DAILY RETURN VALUES: 2017")
print(format_info.format("Day", *tab_headers))
# print table with yr2017 daily return values, mean and std
for day, val in zip(wk_days, week_all_return_2017):
    print(format_info.format(day, *val))
print("\n")

#### 2 0 1 8
# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers - 2018
print("DAILY RETURN VALUES: 2018")
print(format_info.format("Day", *tab_headers))
# print table with yr2018 daily return values, mean and std
for day, val in zip(wk_days, week_all_return_2018):
    print(format_info.format(day, *val))
print("\n")

#### 2 0 1 9
# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers - 2019
print("DAILY RETURN VALUES: 2019")
print(format_info.format("Day", *tab_headers))
# print table with yr2019 daily return values, mean and std
for day, val in zip(wk_days, week_all_return_2019):
    print(format_info.format(day, *val))
print("\n")

#### 2 0 2 0
# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers - 2020
print("DAILY RETURN VALUES: 2020")
print(format_info.format("Day", *tab_headers))
# print table with yr2020 daily return values, mean and std
for day, val in zip(wk_days, week_all_return_2020):
    print(format_info.format(day, *val))
print("\n")

# Q1 Part3 - are there more days with negative or non-negative returns?
# Q1 Part4 - does your stock lose more on a ”down” day than it gains on an ”up” day?
# Q1 Part5 - are these results the same across days of the week?

# QUESTION 3
#### AVERAGE POS AND NEG TABLES
# avg header values for all the years, only neg and pos return map to vars
avg_neg_mon_return_allyr, avg_pos_mon_return_allyr = round([sf.mean(i) for i in \
                all_mon_return][3], 4), round([sf.mean(i) for i in all_mon_return][6], 4)
avg_neg_tues_return_allyr, avg_pos_tues_return_allyr = round([sf.mean(i) for i in \
                all_tues_return][3], 4), round([sf.mean(i) for i in all_tues_return][6], 4)
avg_neg_wed_return_allyr, avg_pos_wed_return_allyr = round([sf.mean(i) for i in \
                all_wed_return][3], 4), round([sf.mean(i) for i in all_wed_return][6], 4)
avg_neg_thurs_return_allyr, avg_pos_thurs_return_allyr = round([sf.mean(i) for i in \
                all_thurs_return][3], 4), round([sf.mean(i) for i in all_thurs_return][6], 4)
avg_neg_fri_return_allyr, avg_pos_fri_return_allyr = round([sf.mean(i) for i in \
                all_fri_return][3], 4), round([sf.mean(i) for i in all_fri_return][6], 4)

#lists of pos and neg return for the week, avg years
wk_avg_neg_return = [avg_neg_mon_return_allyr, avg_neg_tues_return_allyr, \
    avg_neg_wed_return_allyr, avg_neg_thurs_return_allyr, avg_neg_fri_return_allyr, None]

wk_avg_pos_return = [avg_pos_mon_return_allyr, avg_pos_tues_return_allyr, \
    avg_pos_wed_return_allyr, avg_pos_thurs_return_allyr, avg_pos_fri_return_allyr]

# print(all_mon_return)
# print("avg_neg_mon_return_allyr = {}".format(avg_neg_mon_return_allyr))
# print("avg_pos_mon_return_allyr = {}".format(avg_pos_mon_return_allyr))

# table format info
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri", ""]
# format_info_wk = "{:<16}" * (len(wk_days))
# format_info_val = "{:<16.4}" * 6
format_info_wk = "{:<10} {:<10} {:<10} {:<10} {:<10}"
format_info_val = "{:<10.4} {:<10.4} {:<10.4} {:<10.4} {:<10.4}"

#print tables for avg positive and negative return for the week
print("AVERAGE NEGATIVE RETURN VALUES")
print(format_info_wk.format(*wk_days))
print(format_info_wk.format(*wk_avg_neg_return))
print("\n")

print("AVERAGE POSITIVE RETURN VALUES")
print(format_info_wk.format(*wk_days))
print(format_info_val.format(*wk_avg_pos_return))
print("\n")

# Q2 Part1 - are there any patterns across days of the week?
# Q2 Part2 - are there any patterns across different years for the same day of the week?
# Q2 Part3 - what are the best and worst days of the week to be invested for each year.
# Q2 Part4 - do these days change from year to year for your stock?


# QUESTION 3
#### NERV AGREGATED YEAR TABLE
# collapsed lists by year
mon_return_agregated_yrs = [sf.mean(i) for i in all_mon_return]
tues_return_agregated_yrs = [sf.mean(i) for i in all_tues_return]
wed_return_agregated_yrs = [sf.mean(i) for i in all_wed_return]
thurs_return_agregated_yrs = [sf.mean(i) for i in all_thurs_return]
fri_return_agregated_yrs = [sf.mean(i) for i in all_fri_return]

# printable versions of the lists
print_agstr = ["%.4f" % i for i in mon_return_agregated_yrs] #string
print_agfloat = [float(ii) for ii in ["%.4f" % i for i in mon_return_agregated_yrs]] #float
# print("\n", print_agstr)

nerv_agregated_lst = [mon_return_agregated_yrs, tues_return_agregated_yrs, \
                      wed_return_agregated_yrs, thurs_return_agregated_yrs, \
                      fri_return_agregated_yrs]

# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers
print("NERV AGGREGATED TABLE")
print(format_info.format("Day", *tab_headers))
# print table with aggregated return values, num, mean and std
for day, val in zip(wk_days, nerv_agregated_lst):
    print(format_info.format(day, *val))
print("\n")


#### SPY DATA
# all year daily returns (M-F) for R = {r1, . . . , rn}
spy_return_mon_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Monday', \
                                        'SPY'))) for i in range(2016, 2021)]
spy_return_tues_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Tuesday', \
                                        'SPY'))) for i in range(2016, 2021)]
spy_return_wed_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Wednesday', \
                                        'SPY'))) for i in range(2016, 2021)]
spy_return_thurs_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Thursday', \
                                        'SPY'))) for i in range(2016, 2021)]
spy_return_fri_all = [ list(map(lambda x: x[-3], sf.td_yrdy(i, 'Friday', \
                                        'SPY'))) for i in range(2016, 2021)]

# Q1 Part1 - for yrs, mean and std, sets R, R− and R+, daily returns, NERV, M-F
# all year mean - μ(R)
muR_return_mon_allyr_spy = [sf.mean(i) for i in spy_return_mon_all]
muR_return_tues_allyr_spy = [sf.mean(i) for i in spy_return_tues_all]
muR_return_wed_allyr_spy = [sf.mean(i) for i in spy_return_wed_all]
muR_return_thurs_allyr_spy = [sf.mean(i) for i in spy_return_thurs_all]
muR_return_fri_allyr_spy = [sf.mean(i) for i in spy_return_fri_all]

# all year std - σ(R)
stdR_return_mon_allyr_spy = [sf.std(i) for i in spy_return_mon_all]
stdR_return_tues_allyr_spy = [sf.std(i) for i in spy_return_tues_all]
stdR_return_wed_allyr_spy = [sf.std(i) for i in spy_return_wed_all]
stdR_return_thurs_allyr_spy = [sf.std(i) for i in spy_return_thurs_all]
stdR_return_fri_allyr_spy = [sf.std(i) for i in spy_return_fri_all]


# all years NEGATIVE daily returns
spy_return_mon_allyr_neg = [[j for j in list if j<0] for list in spy_return_mon_all]
spy_return_tues_allyr_neg = [[j for j in list if j<0] for list in spy_return_tues_all]
spy_return_wed_allyr_neg = [[j for j in list if j<0] for list in spy_return_wed_all]
spy_return_thurs_allyr_neg = [[j for j in list if j<0] for list in spy_return_thurs_all]
spy_return_fri_allyr_neg = [[j for j in list if j<0] for list in spy_return_fri_all]

# number of neg daily returns, |R−|
len_mon_allyr_neg_spy = [len(i) for i in spy_return_mon_allyr_neg]
len_tues_allyr_neg_spy = [len(i) for i in spy_return_tues_allyr_neg]
len_wed_allyr_neg_spy = [len(i) for i in spy_return_wed_allyr_neg]
len_thurs_allyr_neg_spy = [len(i) for i in spy_return_thurs_allyr_neg]
len_fri_allyr_neg_spy = [len(i) for i in spy_return_fri_allyr_neg]

# all year, negative mean, μ(R−)
muRneg_return_mon_allyr_spy = [sf.mean(i) for i in spy_return_mon_allyr_neg]
muRneg_return_tues_allyr_spy = [sf.mean(i) for i in spy_return_tues_allyr_neg]
muRneg_return_wed_allyr_spy = [sf.mean(i) for i in spy_return_wed_allyr_neg]
muRneg_return_thurs_allyr_spy = [sf.mean(i) for i in spy_return_thurs_allyr_neg]
muRneg_return_fri_allyr_spy = [sf.mean(i) for i in spy_return_fri_allyr_neg]

# all year, negative std, σ(R−)
stdRneg_return_mon_allyr_spy = [sf.std(i) for i in spy_return_mon_allyr_neg]
stdRneg_return_tues_allyr_spy = [sf.std(i) for i in spy_return_tues_allyr_neg]
stdRneg_return_wed_allyr_spy = [sf.std(i) for i in spy_return_wed_allyr_neg]
stdRneg_return_thurs_allyr_spy = [sf.std(i) for i in spy_return_thurs_allyr_neg]
stdRneg_return_fri_allyr_spy = [sf.std(i) for i in spy_return_fri_allyr_neg]


# all years POSITIVE daily returns
spy_return_mon_allyr_pos = [[j for j in list if j>0] for list in spy_return_mon_all]
spy_return_tues_allyr_pos = [[j for j in list if j>0] for list in spy_return_tues_all]
spy_return_wed_allyr_pos = [[j for j in list if j>0] for list in spy_return_wed_all]
spy_return_thurs_allyr_pos = [[j for j in list if j>0] for list in spy_return_thurs_all]
spy_return_fri_allyr_pos = [[j for j in list if j>0] for list in spy_return_fri_all]

# number of pos daily returns, |R+|
len_mon_allyr_pos_spy = [len(i) for i in spy_return_mon_allyr_pos]
len_tues_allyr_pos_spy = [len(i) for i in spy_return_tues_allyr_pos]
len_wed_allyr_pos_spy = [len(i) for i in spy_return_wed_allyr_pos]
len_thurs_allyr_pos_spy = [len(i) for i in spy_return_thurs_allyr_pos]
len_fri_allyr_pos_spy = [len(i) for i in spy_return_fri_allyr_pos]

# all year, positive mean, μ(R+)
muRpos_return_mon_allyr_spy = [sf.mean(i) for i in spy_return_mon_allyr_pos]
muRpos_return_tues_allyr_spy = [sf.mean(i) for i in spy_return_tues_allyr_pos]
muRpos_return_wed_allyr_spy = [sf.mean(i) for i in spy_return_wed_allyr_pos]
muRpos_return_thurs_allyr_spy = [sf.mean(i) for i in spy_return_thurs_allyr_pos]
muRpos_return_fri_allyr_spy = [sf.mean(i) for i in spy_return_fri_allyr_pos]

# all year, positive std, σ(R+)
stdRpos_return_mon_allyr_spy = [sf.std(i) for i in spy_return_mon_allyr_pos]
stdRpos_return_tues_allyr_spy = [sf.std(i) for i in spy_return_tues_allyr_pos]
stdRpos_return_wed_allyr_spy = [sf.std(i) for i in spy_return_wed_allyr_pos]
stdRpos_return_thurs_allyr_spy = [sf.std(i) for i in spy_return_thurs_allyr_pos]
stdRpos_return_fri_allyr_spy = [sf.std(i) for i in spy_return_fri_allyr_pos]

# list of all return data, all year, row=diff_headers and col=year
spy_all_mon_return = [muR_return_mon_allyr_spy, stdR_return_mon_allyr_spy, \
                      len_mon_allyr_neg_spy, muRneg_return_mon_allyr_spy, \
                      stdRpos_return_mon_allyr_spy, len_mon_allyr_pos_spy, \
                      muRpos_return_mon_allyr_spy, stdRpos_return_mon_allyr_spy]
spy_all_tues_return = [muR_return_tues_allyr_spy, stdR_return_tues_allyr_spy, \
                       len_tues_allyr_neg_spy, muRneg_return_tues_allyr_spy, \
                       stdRpos_return_tues_allyr_spy, len_tues_allyr_pos_spy, \
                       muRpos_return_tues_allyr_spy, stdRpos_return_tues_allyr_spy]
spy_all_wed_return = [muR_return_wed_allyr_spy, stdR_return_wed_allyr_spy, \
                      len_wed_allyr_neg_spy, muRneg_return_wed_allyr_spy, \
                      stdRpos_return_wed_allyr_spy, len_wed_allyr_pos_spy, \
                      muRpos_return_wed_allyr_spy, stdRpos_return_wed_allyr_spy]
spy_all_thurs_return = [muR_return_thurs_allyr_spy, stdR_return_thurs_allyr_spy, \
                        len_thurs_allyr_neg_spy, muRneg_return_thurs_allyr_spy, \
                        stdRpos_return_thurs_allyr_spy, len_thurs_allyr_pos_spy, \
                        muRpos_return_thurs_allyr_spy, stdRpos_return_thurs_allyr_spy]
spy_all_fri_return = [muR_return_fri_allyr_spy, stdR_return_fri_allyr_spy, \
                      len_fri_allyr_neg_spy, muRneg_return_fri_allyr_spy, \
                      stdRpos_return_fri_allyr_spy, len_fri_allyr_pos_spy, \
                      muRpos_return_fri_allyr_spy, stdRpos_return_fri_allyr_spy]


#### SPY AGREGATED YEAR TABLE
# collapsed lists by year
mon_return_agregated_yrs_spy = [sf.mean(i) for i in spy_all_mon_return]
tues_return_agregated_yrs_spy = [sf.mean(i) for i in spy_all_tues_return]
wed_return_agregated_yrs_spy = [sf.mean(i) for i in spy_all_wed_return]
thurs_return_agregated_yrs_spy = [sf.mean(i) for i in spy_all_thurs_return]
fri_return_agregated_yrs_spy = [sf.mean(i) for i in spy_all_fri_return]

# printable versions of the lists
print_agstr_spy = ["%.4f" % i for i in mon_return_agregated_yrs_spy] #string
print_agfloat_spy = [float(ii) for ii in ["%.4f" % i for i in mon_return_agregated_yrs_spy]] #float
# print("\n", print_agstr_spy)

spy_agregated_lst = [mon_return_agregated_yrs_spy, tues_return_agregated_yrs_spy, \
                      wed_return_agregated_yrs_spy, thurs_return_agregated_yrs_spy, \
                      fri_return_agregated_yrs_spy]

# table header info
tab_headers = ["μ(R)", "σ(R)", "|R−|", "μ(R−)", "σ(R−)", "|R+|", "μ(R+)", "σ(R+)", ""]
wk_days = ["Mon", "Tues", "Wed", "Thurs", "Fri"]
# table format info
format_info = "{:<10} {:<14.5} {:<12.5} {:<8} {:<14.5} {:<14.5} {:<8} {:<14.5} {:<14.5}"
# print headers
print("SPY AGGREGATED TABLE")
print(format_info.format("Day", *tab_headers))
# print table with aggregated return values, num, mean and std
for day, val in zip(wk_days, spy_agregated_lst):
    print(format_info.format(day, *val))
print("\n")


#### QUESTION 3
# Q3 Part1 & 2


#### QUESTION 4 - ORACLE
# use the oracle to determine what your final stock value will be, buy low sell high

# lists of all adj close values for five years, NERV and SPY stock
nerv_all_adjclose = [i[-4] for i in sf.ticker_data('NERV')]   #can unlist td_yr
spy_all_adjclose = [i[-4] for i in sf.ticker_data('SPY')]

# oracle output values for stock, 5years, buy in value of $100 each
nerv_oracle_predict = sf.oracle(100, nerv_all_adjclose)
print("NERV stock oracle output = ${:,.2f}".format(nerv_oracle_predict))

spy_oracle_predict = sf.oracle(100, spy_all_adjclose)
print("SPY stock oracle output = ${:,.2f}".format(spy_oracle_predict)) #comma sep thousand
print("\n")


#### QUESTION 5
# Buy and hold strategy, buy first day sell last day, no oracle
nerv_buy_hold = (100/nerv_all_adjclose[0]) * nerv_all_adjclose[-1]
print("NERV buy and hold strategy output = ${:,.2f}".format(nerv_buy_hold))

spy_buy_hold = (100/spy_all_adjclose[0]) * spy_all_adjclose[-1]
print("SPY buy and hold strategy output = ${:,.2f}".format(spy_buy_hold))
print("\n")


#### QUESTION 6 - ORACLE REVENGE
# Oracle relays incorrect information for best and worst days to trade
#sorted adjusted close list - NERV, SPY
nerv_all_adjclose_sorted = sorted(nerv_all_adjclose)
spy_all_adjclose_sorted = sorted(spy_all_adjclose)


#Q6 Part1 A - missed best 10 trading days
# best 10 trading days - NERV, SPY
nerv_adjclose_best10 = nerv_all_adjclose_sorted[-10:]
spy_adjclose_best10 = spy_all_adjclose_sorted[-10:]

# filtered adjusted close list, remove best 10 days
nerv_filter_best10 = list(filter(lambda x: x not in nerv_adjclose_best10, nerv_all_adjclose))
spy_filter_best10 = list(filter(lambda x: x not in spy_adjclose_best10, spy_all_adjclose))

# print output
print("NERV stock oracle output, filter best10 = ${:,.2F}".format(sf.oracle(100, nerv_filter_best10)))
print("SPY stock oracle output, filter best10 = ${:,.2F}".format(sf.oracle(100, spy_filter_best10)))
print("\n")



#Q6 Part1 B - missed  worst 10 trading days
# worst 10 trading days - NERV, SPY
nerv_adjclose_worst10 = nerv_all_adjclose_sorted[0:10]
spy_adjclose_worst10 = spy_all_adjclose_sorted[0:10]

# filtered adjusted close list, remove worst 10 days
nerv_filter_worst10 = list(filter(lambda x: x not in nerv_adjclose_worst10, nerv_all_adjclose))
spy_filter_worst10 = list(filter(lambda x: x not in spy_adjclose_worst10, spy_all_adjclose))

# print output
print("NERV stock oracle output, filter worst10 = ${:,.2F}".format(sf.oracle(100, nerv_filter_worst10)))
print("SPY stock oracle output, filter worst10 = ${:,.2F}".format(sf.oracle(100, spy_filter_worst10)))
print("\n")


#Q6 Part1 C - missed worst 5 and best 5 trading days
#worst and best 5 trading days - NERV, SPY
nerv_adjclose_best5_worst5 = sf.unlist([nerv_all_adjclose_sorted[-5:], \
                                        nerv_all_adjclose_sorted[0:5]])
spy_adjclose_best5_worst5 = sf.unlist([spy_all_adjclose_sorted[-5:], \
                                       spy_all_adjclose_sorted[0:5]])


# filtered adjusted close list, remove worst 5 and best 5 days
nerv_filter_best5_worst5 = list(filter(lambda x: x not in nerv_adjclose_best5_worst5, \
                                       nerv_all_adjclose))
spy_filter_best5_worst5 = list(filter(lambda x: x not in spy_adjclose_best5_worst5, \
                                      spy_all_adjclose))

# print output
print("NERV stock oracle output, filter best5 and worst5 = ${:,.2F}".format(sf.oracle(100, \
                                                    nerv_filter_best5_worst5)))
print("SPY stock oracle output, filter best5 and worst5 = ${:,.2F}".format(sf.oracle(100, \
                                                    spy_filter_best5_worst5)))
print("\n")


