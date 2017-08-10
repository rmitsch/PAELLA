# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2
import sys

# Class for connecting to postgres database and executing queries.
class DBConnector:
    # Init instance of DBConnector.
    # @param host
    # @param database
    # @param port
    # @param user
    # @param password
    def __init__(self, host, database, port, user, password):
        try:
            # Init DB connection.
            self.connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
            print("connected")
        except:
            print("unsuccessful")

    # Execute DDL.
    def constructDatabase(self):
        # Create cursor.
        cursor = self.connection.cursor()
        # Execute ddl.sql.
        try:
            cursor.execute(open("backend/database/ddl.sql", "r").read())
            self.connection.commit()
        except:
            print(sys.exc_info()[1])


    # Init instance of DBConnector.
    # @param host
    # @param database
    # @param port
    # @param user
    # @param password
    def import_corpus(self):
        print("dla")