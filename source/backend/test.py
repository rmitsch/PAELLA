# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#

import backend.database.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus

# Note: Reserved keywords for feature columns are id, raw_text.

# Create logger.
Utils.create_logger()

# Create database connection.
dbConnector = DBConnector(host="localhost",
                          database="topac",
                          port="8001",
                          user="admin",
                          password="password")

# Create database.
dbConnector.constructDatabase()

# Import nltk-reuters corpus.
stopwords = []
# Define which corpus-features should be used.
corpus_features = [
    {"name": "categories", "type": "text"}
]
nltk_reuters_corpus = Corpus(name="nltk-reuters",
                             corpus_type="nltk-reuters",
                             stopwords=stopwords,
                             corpus_features=corpus_features)
nltk_reuters_corpus.compile_corpus("", dbConnector)
