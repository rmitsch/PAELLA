# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2

class DBConnector:
    # DB connection.

    def __init__(self, host, database, port, user, password):
        self.name = host
        self.salary = database
        self.salary = database
        self.salary = database
        self.salary = database

    def displayCount(self):
        print
        "Total Employee %d" % Employee.empCount

    def displayEmployee(self):
        print
        "Name : ", self.name, ", Salary: ", self.salary