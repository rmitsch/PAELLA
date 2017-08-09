# @author RM
# @date 2017-08-09
# Class for connecting to database. Uses exactly one connection.
#

import utils.Utils as Utils


# Class representing a corpus in its entirety.
# Used for importing corpora.
class Corpus:
    # Supported corpora types.
    # Should ultimately boil down to one single, standardized CSV template.
    supportedCorporaTypes = ['nltk-reuters']

    # Init instance of Corpus.
    # @param name
    def __init__(self, name, corpus_type):
        logger = Utils.logger

        self.name = name
        self.corpus_type = corpus_type
        # Initialize document collection with empty dataframe.
        self.documents = {}

        # Check if corpus_type is supported.
        if corpus_type not in Corpus.supportedCorporaTypes:
            logger.error("Corpus type not supported")

    # Imports specified corpus.
    # @param corpus_type Type of corpus relating to its structure (different corpus structures have to be processed
    # differently. Ultimately, only one standardized corpus structure should be used).
    # @param path Path to file/folder where corpus is located.
    def import_corpus(self, path):
        if self.corpus_type == "nltk-reuters":
            self.import_nltk_reuters_corpus(path)
        else:
            print("blab")

    # Import nltk-reuter corpus.
    # @param path
    def import_nltk_reuters_corpus(self, path):
        print("importing")
