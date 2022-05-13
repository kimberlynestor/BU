# Name: Kimberly Nestor 
# Class: CS 688
# Date: 04/26/2022
# Module6 Homework 
# This program gets the following information from twitter of 4 politicians.

library(rtweet)
library(dplyr)
library(MASS)

# get following list of target twitter usernames
following = get_friends(c('JoeBiden', 'KamalaHarris', 'BarackObama', 'AOC'))

# eliminate people who appear less than twice
counts = table(following$user_id)
following.reduced = following %>% filter(user_id %in% names(counts[counts > 2]))
following.reduced

# convert key names to usernames
following.names = left_join(following.reduced, distinct(following.reduced,user_id) %>%
                    mutate(names = lookup_users(user_id)$screen_name), by = 'user_id') %>% 
                      select(-user_id) 
following.matrix = as.matrix(following.names)

# set path to curr dir
path = rstudioapi::getActiveDocumentContext()$path
Encoding(path) <- "UTF-8"
setwd(dirname(path))

# save matrix to csv
write.matrix(following.matrix, file='twit_foll.csv')

