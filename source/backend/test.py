# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#
#
# NEXT STEPS
#   - Calculate and persist t-SNE positions for words and documents
#       * Store term coordinates in DB.
#       * Make sure every term has coordinates (and that there are no terms having coordinates but are not persisted).
#   - Calculate and persist t-SNE positions for topics
#   - Description of use cases
#   - How to handle summarization per facet?
#     https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/
#     https://github.com/miso-belica/sumy/

import logging
import backend.database.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus
import backend.algorithm.TopicModel as TopicModel
import backend.algorithm.Doc2VecModel as Doc2VecModel


# Note: Reserved keywords for feature columns are id, raw_text.

# Create logger.
Utils.create_logger()
logger = logging.getLogger("topac")
logger.info("Starting test.py.")

# Create database connection.
db_connector = DBConnector(host="localhost",
                           database="topac",
                           port="8001",
                           user="admin",
                           password="password")

# Set corpus title.
corpus_title = "nltk-reuters"

# # Create database.
# db_connector.construct_database()
#
# # Import nltk-reuters corpus.
# # Define which corpus-features should be used.
# corpus_features = [
#     {"name": "categories", "type": "text"}
# ]
# nltk_reuters_corpus = Corpus(name=corpus_title,
#                              corpus_type="nltk-reuters",
#                              stopwords=[],
#                              corpus_features=corpus_features)
# nltk_reuters_corpus.compile("", db_connector)

# # Create new topic model. Omit hyperparameters for now.
# topic_model = TopicModel(db_connector=db_connector,
#                          corpus_title=corpus_title,
#                          corpus_feature_title="document_id",
#                          n_iterations=10,
#                          n_workers=2)
# # Calculate/compile topic model.
# topic_model.compile()

# Create new doc2vec model. Omit hyperparameters for now.
doc2vec_model = Doc2VecModel(db_connector=db_connector,
                             corpus_title=corpus_title,
                             alpha=0.05,
                             n_workers=2,
                             n_epochs=15,
                             n_window=10)
# Compile doc2vec model.
doc2vec_model.compile()

logger.info("Finished test.py.")