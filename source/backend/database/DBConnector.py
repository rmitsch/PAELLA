# @author RM
# @date 2017-08-08
# Class for connecting to database. Uses exactly one connection.
#

import psycopg2
import sys
import logging

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
        self.logger = logging.getLogger("topac")

        try:
            # Init DB connection.
            self.connection = psycopg2.connect(host=host, database=database, port=port, user=user, password=password)
        except:
            self.logger.critical("Connection to database failed. Check parameters.")

    def construct_database(self):
        """
        Execute DDL.
        :return:
        """
        # Create cursor.
        cursor = self.connection.cursor()
        # Execute ddl.sql.
        try:
            # todo Remove: For testing purposes - database reset at startup.
            cursor.execute("drop schema if exists topac cascade")
            self.connection.commit()
            cursor.execute(open("backend/database/ddl.sql", "r").read())
            self.connection.commit()
        except:
            print(sys.exc_info()[1])

    def load_terms_in_corpus(self, corpus_id):
        """
        Loads all terms in corpus and returns it as map.
        :param corpus_id:
        :return: Map in the form of "word: {terms_ID, terms_in_corpora_ID}"
        """

        cursor = self.connection.cursor()

        cursor.execute("select "
                       "    t.term, "
                       "    t.id, "
                       "    tic.id "
                       "from "
                       "    topac.terms t "
                       "inner join topac.terms_in_corpora tic on"
                       "    tic.terms_id    = t.id and "
                       "    tic.corpora_id  = %s",
                       (corpus_id,))

        # Transform result in map.
        term_dict = {}
        for row in cursor.fetchall():
            term_dict[row[0]] = {}
            term_dict[row[0]]["terms_id"] = row[1]
            term_dict[row[0]]["terms_in_corpora_id"] = row[2]

        return term_dict
