# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#

import backend.database.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus
import backend.algorithm.TopicModel as TopicModel


# Note: Reserved keywords for feature columns are id, raw_text.

# Create logger.
Utils.create_logger()

# Create database connection.
db_connector = DBConnector(host="localhost",
                           database="topac",
                           port="8001",
                           user="admin",
                           password="password")

# Set corpus title.
corpus_title = "nltk-reuters"

# # Create database.
# db_connector.constructDatabase()
#
# # Import nltk-reuters corpus.
# stopwords = []
# # Define which corpus-features should be used.
# corpus_features = [
#     {"name": "categories", "type": "text"}
# ]
# nltk_reuters_corpus = Corpus(name=corpus_title,
#                              corpus_type="nltk-reuters",
#                              stopwords=stopwords,
#                              corpus_features=corpus_features)
# nltk_reuters_corpus.compile("", db_connector)

# Create new topic model. Omit hyperparameters for now.
topic_model = TopicModel(db_connector=db_connector,
                         corpus_title=corpus_title,
                         corpus_feature_title="document_id")
# Calculate/compile topic model.
topic_model.compile()