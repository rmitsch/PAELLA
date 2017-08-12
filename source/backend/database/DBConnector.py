# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2
import sys

class DBConnector:
    """
    Class for connecting to postgres database and executing queries.
    """

    def __init__(self, host, database, port, user, password):
        """
        Init instance of DBConnector.
        :param host:
        :param database:
        :param port:
        :param user:
        :param password:
        """
        try:
            # Init DB connection.
            self.connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
            print("connected")
        except:
            print("unsuccessful")

    def constructDatabase(self):
        """
        Execute DDL.
        :return:
        """
        # Create cursor.
        cursor = self.connection.cursor()
        # Execute ddl.sql.
        try:
            cursor.execute(open("backend/database/ddl.sql", "r").read())
            self.connection.commit()
        except:
            print(sys.exc_info()[1])
