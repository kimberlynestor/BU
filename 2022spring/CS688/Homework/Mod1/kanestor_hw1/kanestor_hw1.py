"""
Name: Kimberly Nestor
Class: CS 688 - Spring 2
Date: 03/22/22
Homework Problem: Hw 1 Q B
Description of Problem: This program uses BeautifulSoup to scrap data from a BU
course webpage with tag class = level_1 .
Webpage: https://www.bu.edu/academics/met/courses/met-cs-688/
"""

import sys
import requests
from bs4 import BeautifulSoup


# BU webpage for cs688
weblink = 'https://www.bu.edu/academics/met/courses/met-cs-688/'
page = requests.get(weblink)

# BeautifulSoup object
soup = BeautifulSoup(page.text, 'html.parser')

# find a class with tag "level_1"
mycls = soup.find_all("a", class_="level_1")
print("BeautifulSOUP web scraping from BU CS688 course page with tag class=level_1\n")
print(mycls, "\n")
print(f'Number of blocks extracted: {len(mycls)}')

# print(soup("level_1"))
# sys.exit()
