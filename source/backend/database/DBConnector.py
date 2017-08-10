# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2

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

    # Init instance of DBConnector.
    # @param host
    # @param database
    # @param port
    # @param user
    # @param password
    def import_corpus(self):
        print("dla")