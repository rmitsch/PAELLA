# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2

class DBConnector:
    def __init__(self, host, database, port, user, password):
        # Init DB connection.
        self.connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
