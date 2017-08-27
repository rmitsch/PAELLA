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
        :return: Map in the form of "term: {terms_ID, terms_in_corpora_ID}"
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

    def load_facets_in_corpus(self, corpus_id):
        """
        Loads all facets in corpus and returns it as map.
        :param corpus_id:
        :return: Map in the form of "id: {facet_label_key}"
        """

        cursor = self.connection.cursor()

        cursor.execute("select "
                       "    cfa.id "
                       "from "
                       "    topac.corpus_facets cfa "
                       # Exclude all corpus features and facets not associated with this corpus.
                       "inner join topac.corpus_features cfe on "
                       "    cfe.id          = cfa.corpus_features_id and "
                       "    cfe.corpora_id  = %s",
                       (corpus_id,))

        # Transform result in map.
        facet_dict = {}
        for row in cursor.fetchall():
            facet_dict[row[0]] = {}
            facet_dict[row[0]]["facet_label_key"] = "t:" + str(row[0])

        return facet_dict

    def fetch_corpus_id(self, corpus_title):
        """
        Reads corpus_id from database for given corpus_title.
        :param corpus_title:
        :return: Corpus ID for this title.
        """

        cursor = self.connection.cursor()
        cursor.execute("select "
                       "   id "
                       "from "
                       "   topac.corpora c "
                       "where "
                       "   c.title = %s",
                       (corpus_title,))
        res = cursor.fetchone()

        # # FOR TESTING PURPOSES: Empty databases topac model tables on start.
        # cursor.execute("truncate table  topac.topic_models, "
        #                "                topac.topics, "
        #                "                topac.terms_in_topics, "
        #                "                topac.corpus_facets_in_topics "
        #                "restart identity")
        # self.connection.commit()

        # Return corpus ID.
        return res[0]

    def fetch_corpus_feature_id(self, corpus_id, corpus_feature_title):
        """
        Reads corpus_feature_id from database for determined corpus ID and corpus_feature_title.
        :param corpus_id:
        :param corpus_feature_title:
        :return: Corpus feature ID for this corpus feature title in this corpus.
        """

        cursor = self.connection.cursor()
        cursor.execute("select "
                       "   cf.id "
                       "from "
                       "   topac.corpus_features cf "
                       "inner join topac.corpora c on "
                       "    c.id = cf.corpora_id and "
                       "    c.id = %s"
                       "where "
                       "   cf.title = %s",
                       (corpus_id, corpus_feature_title))

        # Return corpus feature ID.
        return cursor.fetchone()[0]
