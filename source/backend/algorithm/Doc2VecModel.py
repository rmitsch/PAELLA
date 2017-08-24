# @author RM
# @date 2017-08-24
# Entity representing a doc2vec model, including functionality to preprocess the relevant data.
#

import logging
import gensim.models.doc2vec as doc2vec
from backend.algorithm.IterableTextCorpusForDoc2Vec import IterableTextCorpusForDoc2Vec

class Doc2VecModel:
    """
    Class representing a doc2vec model.
    Used for accessing preprocessed data from the database and generating the actual model in accordance with the
    specified hyperparameters.
    See https://rare-technologies.com/doc2vec-tutorial/.
    """

    def __init__(self, db_connector, corpus_title):
        """
        Set up properties.
        :param db_connector:
        :param corpus_title:
        """
        self.logger = logging.getLogger("topac")
        self.logger.info("Initializing Doc2VecModel object.")

        self.db_connector = db_connector
        self.corpus_title = corpus_title
        self.runtime = 0

        # Get corpus ID.
        self.corpus_id = self.db_connector.fetch_corpus_id(corpus_title=self.corpus_title)
        # Initialize topic model ID as empty.
        self.doc2vec_model_id = None

        print(doc2vec.FAST_VERSION)

    def compile(self):
        """
        Compile doc2vec model and persist it in DB.
        :return:
        """

        self.logger.info("Compiling lda2vec model for corpus " + self.corpus_title)

        # Prepare cursor.
        cursor = self.db_connector.connection.cursor()

        # 1. Load all documents including their labels.
        cursor.execute("select "
                       "    d.id, "
                       "    d.refined_text, "
                       "    array_agg(cf.id::varchar || ':' || cfa.id::varchar) as feature_labels "
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

        # Transfrom document collection to iterable corpus.
        iterable_text_corpus = IterableTextCorpusForDoc2Vec(db_result_set=cursor.fetchall())

        # Initialize doc2vec model.
        model = doc2vec.Doc2Vec(alpha=0.025, min_alpha=0.025)
        model.build_vocab(iterable_text_corpus)


