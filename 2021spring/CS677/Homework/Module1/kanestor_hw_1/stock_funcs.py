"""
Name: Kimberly Nestor
Class: CS 677 - Spring 2
Date: 03/23/21
Homework #1
Description of Problem: This is a module of homemade functions to get the ticker
stock data from a csv file. Also contains an unlist func for nested lists,
std, mean funcs and the oracle predictive function.
"""

import os

def ticker_data(name):
    """This function opens the csv file of a ticker and retrieves stock market
    data to be analysed. The func splits the data, sets fields to proper data
    types and returns a nested list of data from all five sample years. The
    input for the func is the ticker of downloaded stock e.g. NERV."""
    # symbol
    ticker = name
    # for relative path locate symbol file
    here = os.path.abspath(__file__)
    input_dir = os.path.abspath(os.path.join(here, os.pardir))
    ticker_file = os.path.join(input_dir, ticker + '.csv')

    try:
        # open symbol file, each row as one csv
        with open(ticker_file) as f:
            lines = f.read().splitlines()
        # print('opened file for ticker: ', ticker)

        # separate items in list based on comma, remove header row
        lines_sep = [i.split(sep=",") for i in lines[1:]]

        # loop converts certain string vals to numbers for each row of data
        LINES_LST = []
        for i in lines_sep:
            i[1:4], i[5], i[7:len(i)] = map(int, i[1:4]), int(i[5]), map(float, \
                                                                    i[7:len(i)])
            i[-5] = round(i[-5])
            LINES_LST.append(i)
        # print(LINES_LST[0:5])
        return LINES_LST
    except Exception as e:
        print(e)
        print('failed to read stock data for ticker: ', ticker)


# function to obtain list with all data from specific year, 2016-2020
td_yr = lambda year, stock: [i for i in ticker_data(stock) if i[1] == year] # td_yr(year, stock)
# print(ln_lst_yr(2017, 'NERV'))

# function to obtain list with specific year and day data, M-F,
td_yrdy = lambda year, day, stock: [ii for ii in [i for i in ticker_data(stock) \
                        if i[1] == year] if ii[4] == day] # td_yrdy(year, day, stock)

# function to unlist into sep years, day or all data
unlist = lambda nest: [i for list in nest for i in list]

# func to find mean of list
mean = lambda list: sum(list)/len(list)

# func to find standard deviation of list
std = lambda list: ((sum([r**2 for r in list]) / len(list)) - mean(list)**2)**0.5
# mu^2 = (sum(r^2 .. rn^2)/n) - mean^2   #sqrt = x**(0.5)


# print(ticker_data('NERV'))
# print(td_yr(2020, 'NERV'))
# print(td_yrdy(2020, 'Tuesday', 'SPY'))

# print(unlist([td_yrdy(2020, 'Tuesday', 'NERV'), td_yrdy(2020, 'Wednesday', 'NERV')]))
# print([td_yrdy(2020, 'Tuesday', 'NERV'), td_yrdy(2020, 'Wednesday', 'NERV')])


# func to answer Q4, predict next day close value and sell high, buy low next day
def oracle(stock, adjclose_lst):
    """This is the function for the oracle that predicts your profits based
    on using foresight to look ahead and see whether the next trading day will
    result in a profit or loss. Func takes the starting stock value as input and
    a list of adjusted close values."""
    share_lst = [stock/adjclose_lst[0]] # starting stock
    i = 0
    while i < len(adjclose_lst):
        try:
            if adjclose_lst[i + 1] < adjclose_lst[i]:
                sell = adjclose_lst[i] * share_lst.pop()
                buy = sell / adjclose_lst[i + 1]
                share_lst.append(buy)
        except IndexError:
            final_stock = adjclose_lst[-1] * share_lst.pop()
            return final_stock
        i+=1

# test_lst = [5, 10, 7, 6, 10]
# print(oracle(100, test_lst))

