# @author RM
# @date 2017-08-17
# Class for handling topic models.
#

import logging
import os
import gensim
import numpy


class TopicModel:
    """
    Classic LDA topic models.
    Contains data and logic for processing, storing, retrieving and analyzing topic-model related data.
    Use gensim's default values if none are provided.
    See https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
    """

    def __init__(self,
                 db_connector,
                 corpus_title,
                 corpus_feature_title,
                 doc2vec_model,
                 term_coordinates_dict,
                 n_workers=1,
                 alpha=None,
                 eta=None,
                 kappa=100,
                 n_iterations=50,
                 n_top_words_for_coherence=10,
                 n_top_words_for_projection=10):
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
        :param n_top_words_for_coherence: Specifies n for the n most relevant words used to calculate cohesion.
        :param n_top_words_for_projection: Specifies n for the n most relevant words used to project topic model into
        flattened word embedding space.
        :param doc2vec_model: Instance of gensim's doc2vec model supposed for calculation of word embedding-related
        properties in this topic model. Note: If said properties are to be calculted in dependence of multiple doc2vec
        models, some refactoring is necessary. For now, exactly one doc2vec model per x topic models is presumed (in
        implementation, not in DB model, which is more flexible in this regard).
        :param term_coordinates_dict: Dictionary (term => coordinate array) containing low-dimensional coordinates as
        calculated by t-SNE.
        """
        self.logger = logging.getLogger("topac")
        self.db_connector = db_connector

        self.logger.info("Initializing topic model.")

        self.corpus_title = corpus_title
        self.corpus_feature_title = corpus_feature_title
        # Use gensim's default values if none are provided.
        # See https://radimrehurek.com/gensim/models/ldamodel.html#gensim.models.ldamodel.LdaModel
        # todo Set kappa to number of possible values for selected feature.
        self.kappa = kappa
        self.alpha = alpha if alpha is not None else 1.0 / self.kappa
        self.eta = eta if eta is not None else 1.0 / self.kappa
        self.n_iterations = n_iterations
        self.n_workers = n_workers
        # Initialize runtime with 0.
        self.runtime = 0
        # Initialize coordinates with empty array.
        self.coordinates = []
        # Set application-specific hyperparameters.
        self.n_top_words_for_coherence = n_top_words_for_coherence
        self.n_top_words_for_projection = n_top_words_for_projection

        # Set doc2vec model.
        self.doc2vec_model = doc2vec_model
        # Set term-coordinate matrix.
        self.term_coordinates_dict = term_coordinates_dict

        # Initialize dictionary and corpus as empty, since they aren't always needed.
        self.gensim_dictionary = None
        self.gensim_corpus = None

        # Get corpus ID.
        self.corpus_id = self.db_connector.fetch_corpus_id(corpus_title=self.corpus_title)
        # Get corpus feature ID.
        self.corpus_feature_id = self.db_connector.fetch_corpus_feature_id(
            corpus_id=self.corpus_id, corpus_feature_title=self.corpus_feature_title
        )
        # Initialize topic model as empty.
        self.model = None
        # Initialize topic model ID in DB as empty.
        self.topic_model_id = None

    def compile(self):
        """
        Initializes TopicModel instance by fetching preprocessed data from the database and then calculating the desired
        results. Persists results in DB.
        Note: Should only be execute after preprocessing the corpus and _before_ actual usage of the topic model.
        If the topic model results were already generated and stored in the DB, call method load().
        :return:
        """

        self.logger.info("Compiling topic model.")

        # 1. Initialize cursor.
        cursor = self.db_connector.connection.cursor()

        # 2. Load dictionary and corpus from database.
        self.load_gensim_models(cursor=cursor)

        # 3. Train LDA model.
        self.model = gensim.models.LdaMulticore(corpus=self.gensim_corpus,
                                                workers=self.n_workers,
                                                id2word=self.gensim_dictionary,
                                                alpha=self.alpha,
                                                eta=self.eta,
                                                minimum_probability=0.000000001,
                                                num_topics=self.kappa,
                                                iterations=self.n_iterations)

        # 4. Store results in database.
        self.import_topic_model(cursor, self.model)

        # 5. Commit changes.
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
        self.logger.info("Loading terms in TopicModel.import_topic_model(...).")
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
        self.logger.info("Importing topics.")
        sequence_number = 1
        for topic in topics:
            self.import_topic(cursor=cursor,
                              topic=topic,
                              sequence_number=sequence_number,
                              term_dict=term_dict)
            sequence_number += 1

        self.logger.info("Importing topic-in-document distribution.")
        # 5. Import topic-in-document matrix.
        self.import_topics_in_documents_distribution(cursor=cursor,
                                                     topic_model=topic_model)

        # Delete gensim files from disk (apparently used during training, maybe for updating the file
        # with results).
        TopicModel.delete_gensim_model_files()

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
                       "        coordinates, "
                       "        coherence "
                       ") "
                       "values (%s, %s, %s, %s, %s, %s) "
                       "returning id",
                       (sequence_number,
                        "",
                        self.topic_model_id,
                        0,
                        # Calculate topic coordinates in reduced word embedding space.
                        self.calculate_coordinates(top_topic_words_with_probabilities=
                                                   topic[1][:self.n_top_words_for_projection]),
                        # Calculate topic cohesion for topic (cohesion for topic model can be derived from that).
                        # Use only self.n_top_words_for_coherence most relevant words for that.
                        self.calculate_coherence(top_topic_words_with_probabilities=
                                                 topic[1][:self.n_top_words_for_coherence])
                        ))
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

    def calculate_coherence(self, top_topic_words_with_probabilities):
        """
        Calculates cohesion for specified topic. Uses pairwise euclidean distance.
        :param top_topic_words_with_probabilities:
        :return:
        """

        coherence = 0
        # number_of_words should always equal self.n_top_words_for_coherence.
        number_of_words = len(top_topic_words_with_probabilities)
        for index in range(0, number_of_words):
            for index2 in range(index, number_of_words):
                # Use euclidean distance between doc2vec vectors for corresponding words, use topic probabilities to
                # weigh them.
                coherence += numpy.linalg.norm(
                                numpy.multiply(
                                    self.doc2vec_model[top_topic_words_with_probabilities[index][0]],
                                    top_topic_words_with_probabilities[index][1]) -
                                numpy.multiply(
                                    self.doc2vec_model[top_topic_words_with_probabilities[index2][0]],
                                    top_topic_words_with_probabilities[index2][1]
                                )
                )

        # Normalize coherence (sum divided by number of pairs).
        return coherence / (number_of_words * number_of_words - number_of_words) / 2

    def calculate_coordinates(self, top_topic_words_with_probabilities):
        """
        Projects topic into word embedding space using probabilities and doc2vec coorediates of
        top_top_topic_words_with_probabilities most relevant words.
        :param top_topic_words_with_probabilities:
        :return:
        """

        # CONTINUE HERE:
        #     - Sum up weighted coordinate vectors (use numpy.array)
        #     - Cast vector elements to int (?)
        #     - Return array
        #     - How to proceed after that? Projection and cohesion should be done
        coordinates = None
        # Number of words should always equal self.n_top_words_for_projection.
        for index in range(0, len(top_topic_words_with_probabilities)):
            # Use probability-weighted average over word coordinates.
            if coordinates is None:
                coordinates = numpy.multiply(
                    self.term_coordinates_dict[top_topic_words_with_probabilities[index][0]],
                    top_topic_words_with_probabilities[index][1]
                )
            else:
                coordinates += numpy.multiply(
                    self.term_coordinates_dict[top_topic_words_with_probabilities[index][0]],
                    top_topic_words_with_probabilities[index][1]
                )

        return coordinates.tolist()

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