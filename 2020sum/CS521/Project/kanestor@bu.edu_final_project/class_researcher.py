"""
Name: Kimberly Nestor
Class: CS 521 - Summer 2
Date: 08/22/20
Final Project
Description of Problem: This program contains classes, takes a name as an input \
                        and returns a string that can be used as a sql command \
                        to update a database.
"""

class Researcher(object):
    """
    These are the sql insert commands for all researchers
    """
    def __init__(self, last_name="", first_initial='', mid_initial='', key=0):
        # init() method that takes at least 1 arg
        """
        Constructor for Researcher
        """
        # public attributes
        self.last_name = last_name
        self.first_initial = first_initial
        self.mid_initial = mid_initial
        # private attribute
        self._key = key

    def __str__(self):
        """
        Method returns sql command used to update database for researcher
        """
        researcher_insert_str = "INSERT INTO Researcher VALUES(" + str(self._key) +\
                                ", '" + self.first_initial + "', '" + \
                                self.mid_initial + "', '" + self.last_name + \
                                "');\n"
        # return the constructed sql command
        return researcher_insert_str


    def __repr__(self):
        # repr method
        """
        Method returns a string representation of the class and input value
        """
        return "{self.__class__.__name__}(last_name={self.last_name})"\
            .format(self=self)


    def lcn_alumni(self): # public method
        """
        This is a public methof that calls the private method __alum()
        """
        return self.__alum()


    def __alum(self): # private method
        """
        Returns whether the researcher was an lcn alumni
        """
        return "HA! You are an LCN alumnus!"

class LcnFaculty(Researcher):
    """
    These are the sql commands specific to LCN faculty
    """
    def __str__(self):
        """
        Method returns sql command used to update database for lcn faculty
        """
        researcher_subtype_insert_str = "INSERT INTO LCN_faculty VALUES(" + str( \
            self._key) + ");\n"
        # return the constructed sql command
        return researcher_subtype_insert_str


class LcnMember(Researcher):
    """
    These are the sql commands specific to LCN members
    """
    def __str__(self):
        """
        Method returns sql command used to update database for lcn members
        """
        researcher_subtype_insert_str = "INSERT INTO LCN_member VALUES(" + str( \
            self._key) + ");\n"
        # return the constructed sql command
        return researcher_subtype_insert_str


class LcnCollaborator(Researcher):
    """
    These are the sql commands specific to LCN collaborators
    """
    def __str__(self):
        """
        Method returns sql command used to update database for lcn members
        """
        researcher_subtype_insert_str = "INSERT INTO LCN_collaborator VALUES(" + str( \
            self._key) + ");\n"
        # return the constructed sql command
        return researcher_subtype_insert_str


# test section
# test = Researcher(last_name="Rover", first_initial="A", mid_initial="B")
# test = LcnCollaborator(last_name="Rover")
# print(test.__repr__())