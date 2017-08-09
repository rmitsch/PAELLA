# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#

import backend.datamodel.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus

# Create logger.
Utils.create_logger()

# Create database connection.
dbConnector = DBConnector(host="localhost", database="topac", port="8001", user="admin", password="password")

# Import nltk-reuters corpus.
stopwords = []
nltk_reuters_corpus = Corpus(name="reuters", corpus_type="nltk-reuters", stopwords=stopwords)
nltk_reuters_corpus.import_corpus("")
