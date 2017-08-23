# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#
#
# NEXT STEPS
#   - How to handle summarization per facet?
#     https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/
#     https://github.com/miso-belica/sumy/
#   - Sentiment analysis - per document? Per facet? At all?
#     Calculate per document and then average per facet? -> nltk + VADER
#     http://www.nltk.org/howto/sentiment.html
#     http://www.geeksforgeeks.org/twitter-sentiment-analysis-using-python/
#   - Import topic models
#   - Description of use cases
#   - Generate and import doc2vec model
#   - Calculate and persist t-SNE positions for words and documents
#   - Calculate and persist t-SNE positions for topics

import backend.database.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus
import logging
import backend.algorithm.TopicModel as TopicModel


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

# Create database.
# db_connector.construct_database()

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
                         corpus_feature_title="document_id",
                         n_iterations=10)
# Calculate/compile topic model.
topic_model.compile()

logger.info("Finished test.py.")