# @author RM
# @date 2017-08-17
# Class for handling topic models.
#

import logging
import os
import gensim

class TopicModel:
    """
    Classic LDA topic models.
    Contains data and logic for processing, storing, retrieving and analyzing topic-model related data.
    """

    def __init__(self, db_connector, corpus_title, corpus_feature_title, alpha=None, beta=None, kappa=None):
        """
        Set up topic model parameters.
        Note: Uses title instead of IDs as arguments to simplify usage (and because performance isn't critical
        for a one-time lookup).
        :param db_connector:
        :param corpus_title:
        :param alpha:
        :param beta:
        :param kappa:
        """
        self.logger = logging.getLogger("topac")
        self.db_connector = db_connector

        self.corpus_title = corpus_title
        self.corpus_feature_title = corpus_feature_title
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Initialize dictionary and corpus as empty, since they aren't always needed.
        self.gensim_dictionary = None
        self.gensim_corpus = None

    def compile(self):
        """
        Initializes TopicModel instance by fetching preprocessed data from the database and then calculating the desired
        results. Persists results in DB.
        Note: Should only be execute after preprocessing the corpus and _before_ actual usage of the topic model.
        If the topic model results were already generated and stored in the DB, call method load().
        :return:
        """
        # Initialize cursor.
        cursor = self.db_connector.connection.cursor()

        # Load dictionary and corpus from database.
        self.load_gensim_models(cursor=cursor)
        print("loaded successfully")

        # Commit changes.
        self.db_connector.connection.commit()

    def load_gensim_models(self, cursor):
        """
        Loads preprocessed gensim models (dictionary and corpus) from database.
        Sets s dictionary and corpus for this corpus and corpus feature.
        :param cursor:
        :return:
        """
        cursor.execute("select "
                       "    cf.gensim_dictionary as gensim_dictionary, "
                       "    cf.gensim_corpus as gensim_corpus, "
                       "    cf.feature_value_sequence as feature_value_sequence "
                       "from "
                       "    topac.corpus_features cf "
                       "inner join topac.corpora c on "
                       "    c.title = %s and "
                       "    c.id    = cf.corpora_id "
                       "where "
                       "   cf.title   = %s",
                       (self.corpus_title, self.corpus_feature_title))
        res = cursor.fetchone()

        # gensim can only read from file, so we dump the data do files.
        with open("tmp_dict_file.dict", "wb") as tmp_dict_file:
            tmp_dict_file.write(res[0])
        with open("tmp_corpus_file.mm", "wb") as tmp_corpus_file:
            tmp_corpus_file.write(res[1])

        self.gensim_dictionary = gensim.corpora.Dictionary.load("tmp_dict_file.dict")
        self.gensim_corpus = gensim.corpora.MmCorpus("tmp_corpus_file.mm")

        # Clean up files.
        os.remove("tmp_dict_file.dict")
        os.remove("tmp_corpus_file.mm")



