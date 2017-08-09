# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#

import datamodel.DBConnector as DBConnector
import utils.Utils as Utils
import algorithm.Corpus as Corpus

# Create logger.
Utils.create_logger()

# Create database connection.
dbConnector = DBConnector(host="localhost", database="topac", port="8001", user="admin", password="password")

# Import nltk-reuters corpus.
nltk_reuters_corpus = Corpus(name="reuters", corpus_type="nltk-reuters")
nltk_reuters_corpus.import_corpus("")
