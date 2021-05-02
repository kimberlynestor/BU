"""
Name: Kimberly Nestor
Class: CS 521 - Summer 2
Date: 08/22/20
Final Project
Description of Problem: This program takes semi structured text data of \
                        publications and automatically constructs SQL commands \
                        that can be used to update a database. Commands are \
                        stored in text files as output: ComdSqlPaper.csv, \
                        ComdSqlJournal.csv, ComdSqlResearcher.csv.
"""


import pandas as pd
import class_researcher as cr
# cr.Researcher()
import sys


try:
    # input files, submit data
    # Read all publication data from these files
    pub_all_data = pd.read_excel('FischlPubsSamp.xlsx')
    # read journals and researcher names from these files
    researcher_name_data = open('ResearcherNameData.csv', 'r')
    pub_journal_data = open('FischlPubJournals.csv', 'r')

except(FileNotFoundError, OverflowError, TypeError):
    print("Error: Could not locate the input files needed to run this program\n" +\
          "Files needed are: ResearcherNameData.csv, FischlPubJournals.csv," + \
          " FischlPubsSamp.xlsx")


# Write constructed commands saved to these files
# output file
comdsql_paper = open('ComdSqlPaper.csv', 'w')
comdsql_paper = open('ComdSqlPaper.csv', 'a')

comdsql_journal = open('ComdSqlJournal.csv', 'w')
comdsql_journal = open('ComdSqlJournal.csv', 'a')

comdsql_researcher = open('ComdSqlResearcher.csv', 'w')
comdsql_researcher = open('ComdSqlResearcher.csv', 'a')


def journal_insert_func(journal_str, key="0"):
    "takes a list of journals and returns a SQL command as a string"
    assert type(journal_str) == str, \
        "Error: journal_insert_func() input was not a character!"
    journal_insert_str = "INSERT INTO Journal VALUES(" + str(key) + ", NULL, " \
                           + "'" + journal_str + "'" + ", NULL);\n"
    return(journal_insert_str)

# user function unit test
# print(journal_insert_func(journal_str="str"))
# print(journal_insert_func(journal_str=1))
"""
try:
    print(journal_insert_func(journal_str="string", key=1))
    print(journal_insert_func(journal_str=1, key=1))
except(NameError, AssertionError, ValueError, TypeError):
    print("Error: Please double check your input file and input values.")
# can be used to ignore rest of code and only test function
"""
# sys.exit()

# JOURNALS
# user defined function, container type, iteration type, conditional, try
try:
    if __name__ == '__main__':

        # list of all journals, with repeats
        journal_list_all = [((i.replace(',,,,\n', '\n')).replace(',,,,', '\n')).\
                            replace('"', '') for i in pub_journal_data]

        # creates list of only unique journals, sort list
        journal_list_unique = []
        for i in journal_list_all:
            if i not in journal_list_unique:
                journal_list_unique.append(i)
        journal_list_unique.sort()

        # replace all ( ' ) with ( '' ) so that commands will be recognised in SQL
        journal_list_unique = [(i.replace("\n", '')).replace("'", "''") for i in \
                               journal_list_unique]
        # print(journal_list_unique)

        # takes a list of journals and uses journal_insert_func to make SQL commands
        count = 1101
        for journal in journal_list_unique:
            # print(journal)
            journal_insert_sqlcomd = journal_insert_func(str(journal), str(count))
            count+=1
            # print(journal_insert_sqlcomd)
            comdsql_journal.writelines(journal_insert_sqlcomd)
except(NameError, AssertionError, ValueError):
    print("Error: Please double check your input file and input values.")


# PAPERS
try:
    # read excel sheet and make a dataframe
    df_paper_info = pd.DataFrame(pub_all_data)

    # read aspects of paper data from dataframe
    paper_title = [str(i.replace("'", "''")) for i in df_paper_info['Title']]
    paper_authors = [str(i) for i in df_paper_info['Authors']]
    paper_year = [str(i) for i in df_paper_info['Year']]
    paper_jkey = [str(i) for i in df_paper_info['Journal_key']]
    paper_citations = [str(i) for i in df_paper_info['Citations']]
    paper_doi = [str(i) for i in df_paper_info['DOI']]
    paper_eid = [str(i) for i in df_paper_info['EID']]
    paper_ref = [str(i.replace("'", "''")) for i in df_paper_info['Reference']]
    paper_link = [str(i) for i in df_paper_info['Abstract']]

    # takes a lists of paper attributes, read all at once and make SQL commands
    paper_insert_lst = []
    # used to create new key values for each loop in the string
    count = 20001
    for jkey, title, year, reference, doi, link, citation in zip(paper_jkey,
                                                                 paper_title,
                                                                 paper_year,
                                                                 paper_ref,
                                                                 paper_doi,
                                                                 paper_link,
                                                                 paper_citations):
        # paper SQL command string variable
        paper_insert_str = "INSERT INTO Paper VALUES(" + str(
            count) + ", " + jkey + ", '" + title + "', " + year + ", '" + \
                           reference + "', '" + doi + "', '" + link + "', " + \
                           citation + ", '', '');\n"

        # append to list and write to text file
        paper_insert_lst.append(paper_insert_str)
        comdsql_paper.writelines((paper_insert_str + "\n"))
        count += 1
    # print(paper_insert_lst)
    # print(paper_insert_lst[0])
except(NameError, AssertionError, TypeError, ValueError):
    print("Error: Please double check your input file and input values.")


# RESEARCHER
# imported user defined classes
if __name__ == '__main__':
    try:
        # read input file with names and make a list of researchers in file
        researcher_lst = [i.replace('\n', '') for i in researcher_name_data.readlines()]

        # lists of researchs of specific categories
        lcn_faculty = ["Fischl", "Aganj", "van der Kouwe", "Augustinack", "Yendiki",
                       "Greve", "Zöllei", "Magnain", "Wang", "Frost", "Dalca"]

        lcn_member = ["Stevens", "Frau-Pascual", "Hoffmann", "Siless",
                      "Varadarajan",
                      "Maffei", "Cheng", "Hoopes", "Wang", "DiCamillo", "Wighton",
                      "Diamond", "Morgan", "Nestor", "Larrabee", "Robert", "Jones",
                      "Freeman", "Vera", "Cordero", "Williams", "Kim"]

        lcn_collaborator = ["Edlow", "Salat", "Reuter", "Tisdall", "Iglesias",
                            "Van Leemput", "Desbordes", "Yeo", "Boas", "Knudsen",
                            "Buckner", "Konukoglu", "Ganz", "Olesen", "Gollub",
                            "Mareyam", "Grant", "Pienaar", "Helmer", "Polimeni",
                            "Pedemente", "Saygin", "Kliemann"]

        lcn_alumni = ["Adolphs", "Agartz"]

        # loop through the list of all researchers and print sql commands specific to
        # their category
        researcher_insert_lst = []
        researcher_subtype_insert_lst = []
        count = 30001
        for i in researcher_lst:
            # separate researcher name at the comma into last and first, mid initials
            idx_comma = i.index(",")
            last_name_only = i[0:idx_comma]
            first_initial = i[idx_comma + 2:idx_comma + 4]
            mid_initial = i[idx_comma + 4:]
            mid_initial.replace(' ', '')

            # create research insert command for all researchers
            researcher_insert_str = cr.Researcher(last_name=last_name_only, \
                                                  first_initial=first_initial, \
                                                  mid_initial=mid_initial, key=count)
            comdsql_researcher.write(str(researcher_insert_str))

            # these statements will create subtype commands for specific researchers
            # commands are saved to ComdSqlResearcher output file
            if last_name_only in lcn_faculty:
                researcher_subtype_insert_str = cr.LcnFaculty(key=count)
                comdsql_researcher.writelines(str(researcher_subtype_insert_str) + "\n")

            elif last_name_only in lcn_member:
                researcher_subtype_insert_str = cr.LcnMember(key=count)
                comdsql_researcher.writelines(str(researcher_subtype_insert_str) + "\n")

            elif last_name_only in lcn_collaborator:
                researcher_subtype_insert_str = cr.LcnCollaborator(key=count)
                comdsql_researcher.writelines(str(researcher_subtype_insert_str) + "\n")

            # private reearcher class method being called
            elif last_name_only in lcn_alumni:
                print(str(researcher_insert_str), str(researcher_insert_str.lcn_alumni()))
    except(NameError, AssertionError, ValueError, TypeError):
        print("Error: Please double check your input values.")


print("\nAll testing was successful!")
# CLASS UNIT TESTS? evaluate using assert statements

pub_journal_data.close()
researcher_name_data.close()
comdsql_paper.close()
comdsql_journal.close()
comdsql_researcher.close()

