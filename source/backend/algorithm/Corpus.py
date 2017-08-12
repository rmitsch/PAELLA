# @author RM
# @date 2017-08-09
# Class for connecting to database. Uses exactly one connection.
#

import backend.utils.Utils as Utils
import nltk as nltk
import logging
import re


class Corpus:
    """
    Class representing a corpus in its entirety.
    Used for importing corpora.
    """
    # Supported corpora types.
    # Should ultimately boil down to one single, standardized CSV template.
    supportedCorporaTypes = ['nltk-reuters']

    def __init__(self, name, corpus_type, stopwords):
        """
        Init instance of Corpus.
        :param name:
        :param corpus_type:
        :param stopwords:
        """
        logger = logging.getLogger("topac")

        self.name = name
        self.corpus_type = corpus_type
        # Initialize document collection with empty dataframe.
        self.documents = {}
        self.stopwords = stopwords

        # Check if corpus_type is supported.
        if corpus_type not in Corpus.supportedCorporaTypes:
            logger.error("Corpus type not supported")

    # Imports specified corpus.
    # @param corpus_type Type of corpus relating to its structure (different corpus structures have to be processed
    # differently. Ultimately, only one standardized corpus structure should be used).
    # @param path Path to file/folder where corpus is located.
    # @param dbConnector
    def import_corpus(self, path, db_connector):
        if self.corpus_type == "nltk-reuters":
            self.import_nltk_reuters_corpus(path, db_connector)
        else:
            print("blab")

    def import_nltk_reuters_corpus(self, path, db_connector):
        """
        Import nltk-reuter corpus.
        :param path:
        :param db_connector:
        :return:
        """

        # Prepare cursor.
        cursor = db_connector.connection.cursor()

        # Import corpus.
        cursor.execute("insert into topac.corpora (title) values (%s) returning id", (self.name,))
        corpus_id = cursor.fetchone()[0]

        # Import corpus features.
        cursor.execute("insert into "
                       "    topac.corpus_features ("
                       "        title, "
                       "        type, "
                       "        corpora_id"
                       ")"
                       "values ('categories', 'text', %s) "
                       "returning id",
                       (corpus_id,))
        corpus_feature_id = cursor.fetchone()[0]

        for fileID in nltk.corpus.reuters.fileids():
            # Fetch document.
            doc = nltk.corpus.reuters.raw(fileids=[fileID]).strip()

            # Import documents in DB.
            cursor.execute("insert into "
                           "topac.documents ("
                           "    title, "
                           "    raw_text, "
                           "    corpora_id"
                           ")"
                           "values (%s, %s, %s) "
                           "returning id",
                           (fileID, doc, corpus_id))
            document_id = cursor.fetchone()[0]

            # Import document feature values.
            cursor.execute("insert into "
                           "topac.corpus_features_in_documents ("
                           "    corpus_features_id, "
                           "    documents_id, "
                           "    value"
                           ") "
                           "values (%s, %s, %s)",
                           (corpus_feature_id, document_id, nltk.corpus.reuters.categories(fileids=[fileID])))

            # Exclude one-letter words
            # Next steps:
            #   - Text preprocessing
            #   - Creating and storing tfidf-models (persist where/how - blob in db?)

        # Commit and store corpus with raw text in DB.
            db_connector.connection.commit()

        # Refine corpus text.
        self.refine_corpus_text(db_connector)

    def refine_corpus_text(self, db_connector):
        """
        Preprocess and refine raw texts for entire corpus.
        :param dbConnector:
        :return:
        """

        # 1. Prepare cursor.
        cursor = db_connector.connection.cursor()

        # 2. Load all documents from database.
        cursor.execute("select"
                       "    d.id, "
                       "    d.raw_text "
                       "from "
                       "    topac.documents d "
                       "inner join topac.corpora c on"
                       "    c.title = %s and"
                       "    c.id    = d.corpora_id ",
                       (self.name,))
        documents = cursor.fetchall()

        # Exclude special signs: All ; & > < = numbers : , . ' "
        #refined_text = re.sub(r"([;]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|(\d+)|[']|[\"])", "", doc)

        #return refined_text
