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
        # Initialize topic model ID as empty.
        self.topic_model_id = None

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
        cursor.execute("truncate table  topac.topic_models, "
                       "                topac.topics, "
                       "                topac.terms_in_topics, "
                       "                topac.corpus_facets_in_topics "
                       "restart identity")
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
                                                 minimum_probability=0.00000001,
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
                       "    cf.gensim_corpus as gensim_corpus "
                       "from "
                       "    topac.corpus_features cf "
                       "inner join topac.corpora c on "
                       "    c.title = %s and "
                       "    c.id    = cf.corpora_id "
                       "where "
                       "   cf.title   = %s",
                       (self.corpus_title, self.corpus_feature_title))
        res = cursor.fetchone()

        # gensim can only read from file, so we dump the data to files.
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

        # 1. Load topics.
        topics = topic_model.show_topics(num_topics=topic_model.num_topics,
                                         num_words=topic_model.num_terms,
                                         formatted=False)

        # 2. Load terms in corpus as map.
        term_dict = self.db_connector.load_terms_in_corpus(corpus_id=self.corpus_id)

        # 3. Import topic model entry.
        cursor.execute(
            "insert into "
            "    topac.topic_models ( "
            "       alpha, "
            "       eta, "
            "       kappa, "
            "       n_iterations, "
            "       corpora_id, "
            "       corpus_features_id, "
            "       runtime, "
            "       coordinates "
            ") "
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
        self.topic_model_id = cursor.fetchone()[0]

        # 4. Import topics.
        sequence_number = 1
        for topic in topics:
            self.import_topic(cursor=cursor,
                              topic=topic,
                              sequence_number=sequence_number,
                              term_dict=term_dict)
            sequence_number += 1

        # 5. Import topic-in-document matrix.
        self.import_topics_in_documents_distribution(cursor=cursor,
                                                     topic_model=topic_model)

    def import_topic(self, cursor, topic, sequence_number, term_dict):
        """
        Import topic (including topic-in-document and term-in-topic distribution) into DB.
        :param cursor:
        :param topic:
        :param sequence_number:
        :param term_dict:
        :return:
        """

        # 1. Insert new topic entry into DB.
        cursor.execute("insert into "
                       "    topac.topics("
                       "        sequence_number, "
                       "        title, "
                       "        topic_models_id, "
                       "        quality, "
                       "        coordinates"
                       ") "
                       "values (%s, %s, %s, %s, %s) "
                       "returning id",
                       (sequence_number,
                        "",
                        self.topic_model_id,
                        0,
                        []))
        topic_id = cursor.fetchone()[0]

        # 2. Import term-in-topic probability matrix.
        tuples_to_insert = []
        for word, prob in topic[1]:
            tuples_to_insert.append((topic_id,
                                     prob,
                                     term_dict[word]["terms_in_corpora_id"]))
        # Import term-in-topics data.
        cursor.execute(cursor.mogrify("insert into "
                                      "  topac.terms_in_topics ( "
                                      "     topics_id, "
                                      "     probability,"
                                      "     terms_in_corpora_id "
                                      "     ) "
                                      " values " +
                                      ','.join(["%s"] * len(tuples_to_insert)), tuples_to_insert))

    def import_topics_in_documents_distribution(self, cursor, topic_model):
        """
        Imports topic-in-document probability matrix.
        :param cursor:
        :param topic_model:
        :return:
        """

        # 1. Fetch IDs of facets in correct order.
        cursor.execute("select "
                       "    id "
                       "from "
                       "    topac.corpus_facets "
                       "where"
                       "    corpus_features_id = %s "
                       "order by"
                       "    sequence_number asc",
                       (self.corpus_feature_id,))
        facet_ids_res = cursor.fetchall()

        # 2. Fetch IDs of topics in correct order.
        cursor.execute("select "
                       "    id "
                       "from "
                       "    topac.topics "
                       "where"
                       "    topic_models_id = %s "
                       "order by"
                       "    sequence_number asc",
                       (self.topic_model_id,))
        topic_ids_res = cursor.fetchall()

        # 2. Apply transformation to corpus to retrieve topic-document probabilities.
        # See
        # https://stackoverflow.com/questions/25803267/retrieve-topic-word-array-document-topic-array-from-lda-gensim.
        # Document number/ID (sequence of documents in corpusMM is equivalent with sequence of feature values in DB.

        # Clear collection of tuples to insert.
        tuples_to_insert = []
        # Store row indices.
        document_row_index = 0
        topic_row_index = 0

        # Iterate through all topic-document edges.
        for topic_in_docs_datum in topic_model[self.gensim_corpus]:
            # Fetch current facet ID (facet is equivalent to document here, since a facet is comprised of all documents
            # with the same value for the selected corpus feature.
            corpus_facet_id = facet_ids_res[document_row_index][0]

            # Deconstruct in individual topic -> document relations.
            for topic_in_doc_datum in topic_in_docs_datum:
                # Fetch current topic ID.
                # Assumption: Since facets/documents were imported in the same order as they were provided to gensim's
                # models and gensim's show_topics() sequence of topics is acknowledged as a consequence of the sort by
                # sequence_number, both the sequence of documents and the sequence of topics should agree with gensim's
                # sort order in the topic-document matrix.
                topic_id = topic_ids_res[topic_row_index][0]

                # Append new record.
                tuples_to_insert.append((topic_id, corpus_facet_id, topic_in_doc_datum[1]))

                # Keep track of processed topic entries in topic-document matrix.
                topic_row_index += 1

            # Keep track of processed facet/document entries in topic-document matrix.
            document_row_index += 1

            # Reset topic row index.
            topic_row_index = 0

        # Persist topic-document matrix.
        cursor.execute(cursor.mogrify("insert into "
                                      "  topac.corpus_facets_in_topics ( "
                                      "     topics_id, "
                                      "     corpus_facets_id,"
                                      "     probability "
                                      ") "
                                      " values " +
                                      ','.join(["%s"] * len(tuples_to_insert)), tuples_to_insert))