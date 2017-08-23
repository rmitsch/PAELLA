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

    def __init__(self, db_connector, corpus_title, corpus_feature_title, n_workers=1, alpha=None, eta=None, kappa=None, n_iterations=None):
        """
        Set up topic model parameters.
        Note: Uses title instead of IDs as arguments to simplify usage (and because performance isn't critical
        for a one-time lookup).
        :param db_connector:
        :param corpus_title:
        :param corpus_feature_title:
        :param n_workers:
        :param alpha:
        :param eta:
        :param kappa:
        :param n_iterations:
        """
        self.logger = logging.getLogger("topac")
        self.db_connector = db_connector

        self.logger.info("Initializing topic model.")

        self.corpus_title = corpus_title
        self.corpus_feature_title = corpus_feature_title
        # Use gensim's default values if none are provided.
        # See https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
        self.kappa = kappa if kappa is not None else 100
        self.alpha = alpha if alpha is not None else 1.0 / self.kappa
        self.eta = eta if eta is not None else 1.0 / self.kappa
        self.n_iterations = n_iterations if n_iterations is not None else 50
        self.n_workers = n_workers
        # Initialize runtime with 0.
        self.runtime = 0
        # Initialize coordinates with empty array.
        self.coordinates = []

        # Initialize dictionary and corpus as empty, since they aren't always needed.
        self.gensim_dictionary = None
        self.gensim_corpus = None

        # Get corpus ID.
        self.corpus_id = self.read_corpus_id()
        # Get corpus feature ID.
        self.corpus_feature_id = self.read_corpus_feature_id()

    def read_corpus_id(self):
        """
        Reads corpus_id from database for given corpus_title.
        :return: Corpus ID for this title.
        """

        cursor = self.db_connector.connection.cursor()
        cursor.execute("select "
                       "   id "
                       "from "
                       "   topac.corpora c "
                       "where "
                       "   c.title = %s",
                       (self.corpus_title,))
        res = cursor.fetchone()

        # FOR TESTING PURPOSES: Empty databases topac model tables on start.
        cursor.execute("truncate table topac.topic_models cascade")
        self.db_connector.connection.commit()

        # Return corpus ID.
        return res[0]

    def read_corpus_feature_id(self):
        """
        Reads corpus_feature_id from database for determined corpus ID and corpus_feature_title.
        :return: Corpus feature ID for this corpus feature title in this corpus.
        """

        cursor = self.db_connector.connection.cursor()
        cursor.execute("select "
                       "   cf.id "
                       "from "
                       "   topac.corpus_features cf "
                       "inner join topac.corpora c on "
                       "    c.id = cf.corpora_id and "
                       "    c.id = %s"
                       "where "
                       "   cf.title = %s",
                       (self.corpus_id, self.corpus_feature_title))

        # Return corpus feature ID.
        return cursor.fetchone()[0]

    def compile(self):
        """
        Initializes TopicModel instance by fetching preprocessed data from the database and then calculating the desired
        results. Persists results in DB.
        Note: Should only be execute after preprocessing the corpus and _before_ actual usage of the topic model.
        If the topic model results were already generated and stored in the DB, call method load().
        :return:
        """

        self.logger.info("Compiling topic model.")

        # Initialize cursor.
        cursor = self.db_connector.connection.cursor()

        # Load dictionary and corpus from database.
        self.load_gensim_models(cursor=cursor)

        # Train LDA model.
        topic_model = gensim.models.LdaMulticore(corpus=self.gensim_corpus,
                                                 workers=self.n_workers,
                                                 id2word=self.gensim_dictionary,
                                                 alpha=self.alpha,
                                                 eta=self.eta,
                                                 num_topics=self.kappa,
                                                 iterations=self.n_iterations)

        # Store results in database.
        self.import_topic_model(cursor, topic_model)

        # Delete gensim files from disk (apparently used during training, maybe for updating the file
        # with results).
        TopicModel.delete_gensim_model_files()

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

    @staticmethod
    def delete_gensim_model_files():
        """
        Clean up files when they aren't needed any longer.
        :param self:
        :return:
        """
        os.remove("tmp_dict_file.dict")
        os.remove("tmp_corpus_file.mm")

    def import_topic_model(self, cursor, topic_model):
        """
        Import topic model into database.
        :param cursor:
        :param topic_model:
        :return:
        """

        self.logger.info("Importing topic model.")

        # 0. Load topics.
        topics = topic_model.show_topic()

        # 1. Import topic model entry.
        cursor.execute("insert into "
                       "    topac.topic_models (alpha, "
                       "                        eta, "
                       "                        kappa, "
                       "                        n_iterations, "
                       "                        corpora_id, "
                       "                        corpus_features_id,"
                       "                        runtime,"
                       "                        coordinates) "
                       "values "
                       "    (%s, %s, %s, %s, %s, %s, %s, %s) "
                       "returning id",
                       (self.alpha,
                        self.eta,
                        self.kappa,
                        self.n_iterations,
                        self.corpus_id,
                        self.corpus_feature_id,
                        self.runtime,
                        self.coordinates))

        # 2. Import term-in-topic probability matrix.







