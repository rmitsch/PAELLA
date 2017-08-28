# @author RM
# @date 2017-08-24
# Entity representing a doc2vec model, including functionality to preprocess the relevant data.
#

import logging
import gensim
from backend.algorithm.IterableTextCorpusForDoc2Vec import IterableTextCorpusForDoc2Vec
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy
from MulticoreTSNE import MulticoreTSNE as MulticoreTSNE


class Doc2VecModel:
    """
    Class representing a doc2vec model.
    Used for accessing preprocessed data from the database and generating the actual model in accordance with the
    specified hyperparameters.
    See https://rare-technologies.com/doc2vec-tutorial/.
    Use gensim's default values if none are provided.
    See https://radimrehurek.com/gensim/models/doc2vec.html.
    """

    def __init__(self,
                 db_connector,
                 corpus_title,
                 feature_vector_size=100,
                 alpha=0.025,
                 min_alpha=0.0001,
                 n_window=5,
                 n_workers=1,
                 n_epochs=5):
        """
        Set up properties.
        :param db_connector:
        :param corpus_title:
        """
        self.logger = logging.getLogger("topac")
        self.logger.info("Initializing Doc2VecModel object. doc2vec.FAST_VERSION = " + str(gensim.models.doc2vec.FAST_VERSION))

        self.db_connector = db_connector
        self.corpus_title = corpus_title
        self.runtime = 0

        self.feature_vector_size = feature_vector_size
        self.alpha = alpha
        self.min_alpha = min_alpha
        self.n_window = n_window
        self.n_workers = n_workers
        self.n_epochs = n_epochs

        # Get corpus ID.
        self.corpus_id = self.db_connector.fetch_corpus_id(corpus_title=self.corpus_title)
        # Initialize model as empty.
        self.model = None
        # Initialize topic model ID as empty.
        self.doc2vec_model_id = None

    def compile(self):
        """
        Compile doc2vec model and persist it in DB.
        :return:
        """

        self.logger.info("Compiling lda2vec model for corpus " + self.corpus_title)

        # Prepare cursor.
        cursor = self.db_connector.connection.cursor()

        # FOR TEST PURPOSES: Truncate contents of doc2vec tables.
        cursor.execute("truncate table  topac.doc2vec_models, "
                       "                topac.corpus_facets_in_doc2vec_models, "
                       "                topac.terms_in_doc2vec_model "
                       "restart identity")
        self.db_connector.connection.commit()

        # 1. Load all documents including their labels.
        cursor.execute("select "
                       "    d.id, "
                       "    d.refined_text, "
                       "    array_agg('t:' || cfa.id::varchar) as feature_labels "
                       "from "
                       "    topac.documents as d "
                       # Fetch available features available for this corpus.
                       "inner join topac.corpus_features as cf on "
                       "    cf.corpora_id = d.corpora_id "
                       # Get document's values for these features.
                       "inner join topac.corpus_features_in_documents as cfid on "
                       "    cfid.corpus_features_id = cf.id and "
                       "    cfid.documents_id       = d.id "
                       # Get corpus facet matching this value for this corpus feature value - reason:
                       # we want to use IDs instead of actual feature values for gensim preprocessing in order to
                       # reduce memory requirements.
                       "inner join topac.corpus_facets cfa on "
                       "    cfa.corpus_features_id      = cf.id and "
                       "    cfa.corpus_feature_value    = cfid.value "
                       # Get only documents in the specified corpus.
                       "where "
                       "    d.corpora_id = %s "
                       "group by "
                       "    d.id, "
                       "    d.refined_text; ",
                       (self.corpus_id,))

        # 2. Transfrom document collection to iterable corpus.
        iterable_text_corpus = IterableTextCorpusForDoc2Vec(db_result_set=cursor.fetchall())

        # 3. Initialize and train doc2vec model.
        # See https://rare-technologies.com/doc2vec-tutorial/ for reference (note that tutorial version is slightly
        # deprecated).
        # Include training of word embeddings.
        self.model = gensim.models.doc2vec.Doc2Vec(alpha=self.alpha,
                                                   min_alpha=self.min_alpha,
                                                   dbow_words=1,
                                                   size=self.feature_vector_size,
                                                   window=self.n_window,
                                                   workers=self.n_workers,
                                                   iter=self.n_epochs,
                                                   dm_tag_count=iterable_text_corpus.number_of_document_tags,
                                                   # Don't prune any words not removed during preprocessing. See
                                                   # https://stackoverflow.com/questions/45420466/gensim-keyerror-word-not-in-vocabulary?rq=1
                                                   min_count=1
                                                 )
        self.model.build_vocab(iterable_text_corpus)

        self.logger.info("Training model.")

        # 4. Train doc2vec model in accordance with specified parameters.
        self.model.train(sentences=iterable_text_corpus,
                         total_examples=self.model.corpus_count,
                         start_alpha=self.model.alpha,
                         end_alpha=self.model.min_alpha,
                         epochs=self.model.iter)

        # 5. Prepare for persisting model.

        # Delete temporary training data before persisting model (assuming we don't want to train the model with
        # more documents, since this won't be possible anymore - other than restarting training completely).
        self.model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

        # Dump model to temp file.
        self.model.save("tmp_model.d2v")

        # 6. Persist model.
        self.logger.info("persisting model")
        self.import_doc2vec_model(cursor=cursor)

        # Get rid of temporary file.
        os.remove("tmp_model.d2v")

        # Commit changes.
        self.db_connector.connection.commit()

    def import_doc2vec_model(self, cursor):
        """

        :param self:
        :param cursor:
        :return:
        """

        # 1. Store model in database.
        cursor.execute("insert into "
                       "    topac.doc2vec_models( "
                       "        corpora_id, "
                       "        feature_vector_size, "
                       "        n_window, "
                       "        alpha, "
                       "        min_alpha, "
                       "        n_epochs, "
                       "        gensim_model "
                       ") "
                       "values (%s, %s, %s, %s, %s, %s, %s) "
                       "returning id",
                       (self.corpus_id,
                        self.feature_vector_size,
                        self.n_window,
                        self.alpha,
                        self.min_alpha,
                        self.n_epochs,
                        open("tmp_model.d2v", "rb").read()))
        self.doc2vec_model_id = cursor.fetchone()[0]

        # 2. Retrieve all terms in corpus.
        term_dict = self.db_connector.load_terms_in_corpus(corpus_id=self.corpus_id)

        # 3. Retrieve all facets in corpus.
        facet_dict = self.db_connector.load_facets_in_corpus(corpus_id=self.corpus_id)

        # 4. Prepare coordinate matrix to be used by t-SNE.
        embedded_coordinates_matrix = numpy.zeros((len(self.model.wv.vocab) + len(self.model.docvecs),
                                                  self.feature_vector_size))
        i = 0
        for term, term_values in term_dict.items():
            embedded_coordinates_matrix[i] = self.model[term].astype(numpy.float64)
            term_dict[term]["index"] = i
            i += 1
        for facet_id, facet_data in facet_dict.items():
            embedded_coordinates_matrix[i] = self.model.docvecs[facet_data["facet_label"]].astype(numpy.float64)
            facet_dict[facet_id]["index"] = i
            i += 1

        # 5. Apply t-SNE.
        # See http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html.
        # Metric can be chosen on initialization using parameter 'metric' (default: euclidean). See
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html.
        #
        # todo Note that currently an inofficial multithreaded version of t-SNE is used (see
        # https://github.com/DmitryUlyanov/Multicore-TSNE). There are reports of incosistencies/worse results than
        # with the (horribly slow) scikit-learn implementation. Pay attention to that - if necessary, either
        #   (1) Increase accurady-related variables (angle, iterations, etc.),
        #   (2) issue commands to a fast t-SNE installation in other languages (e.g.
        #       * https://github.com/lvdmaaten/bhtsneor/https://github.com/maximsch2/bhtsne
        #       * https://github.com/rappdw/tsne -
        #       have python wrappers and seems to support multi-threading) or
        #   (3) switch back to scikit-learn,.
        self.logger.info("Applying TSNE to reduce dimensionality.")

        tsne = MulticoreTSNE(n_components=2,
                             method='barnes_hut',
                             metric='euclidean',
                             # todo Remove after tests are done.
                             perplexity=2,
                             # todo Remove after tests are done.
                             angle=0.9,
                             verbose=1,
                             n_jobs=self.n_workers)
        # Train TSNE on gensim's model.
        tsne_results = tsne.fit_transform(embedded_coordinates_matrix)

        plt.scatter(tsne_results[:, 0], tsne_results[:, 1])
        plt.show()

        # 5. Persist term coordinates.
        self.logger.info("Persisting TSNE results.")

        # print(len(tsne_results), len(embedded_coordinates_matrix), self.feature_vector_size)

        # Bulk update:
        # https://stackoverflow.com/questions/33256345/how-to-efficiently-update-a-column-in-a-large-postgresql-table-using-python-ps

        # Prepare term coordinates.
        tuples_to_insert = [None] * len(self.model.wv.vocab)
        i = 0
        for term, term_values in term_dict.items():
            tuples_to_insert[i] = ((self.doc2vec_model_id,
                                    term_values["terms_in_corpora_id"],
                                    tsne_results[term_values["index"]].tolist()))
            i += 1
        # Insert term coordinates into DB.
        cursor.execute(cursor.mogrify("insert into "
                                      "     topac.terms_in_doc2vec_model ( "
                                      "         doc2vec_models_id, "
                                      "         terms_in_corpora_id, "
                                      "         coordinates"
                                      ") "
                                      " values " +
                                      ','.join(["%s"] * len(tuples_to_insert)), tuples_to_insert))

    @staticmethod
    def plot_tsne_results(word_vectors):
        for i in range(1, 4):
            tsne = MulticoreTSNE(n_components=2,
                                 method='barnes_hut',
                                 metric='euclidean',
                                 verbose=1,
                                 random_state=7)
            # Train TSNE on gensim's model.
            tsne_result = tsne.fit_transform(word_vectors)

            plt.subplot(2, 3, i)
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
        for i in range(1, 4):
            tsne = TSNE(n_components=2,
                                 method='barnes_hut',
                                 metric='euclidean',
                                 verbose=1,
                                 random_state=7)
            # Train TSNE on gensim's model.
            tsne_result = tsne.fit_transform(word_vectors)

            plt.subplot(2, 3, 3 + i)
            plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
        plt.show()

