# @author RM
# @date 2017-08-09
# Class for connecting to database. Uses exactly one connection.
#

import nltk as nltk
from pattern3 import vector as pattern_vector
import logging
import re
import gensim
from backend.algorithm.IterableGensimDoc2BowCorpus import IterableGensimDoc2BowCorpus


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

        # Prepare list of documents. Used for preprocessing.
        documents = []

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

            # Add to list of documents.
            documents.append({"db_id": document_id, "raw_text": doc})

        # Commit and store corpus with raw text in DB.
        db_connector.connection.commit()

        # Refine corpus text.
        self.refine_corpus_text(db_connector, documents)

    def refine_corpus_text(self, db_connector, documents):
        """
        Preprocess and refine raw texts for entire corpus.
        :param dbConnector:
        :param documents:
        :return:
        """

        # 1. Prepare cursor.
        cursor = db_connector.connection.cursor()

        # 2. Loop over documents, remove unwanted characters.
        # Note: Assuming performance isn't critical for document import.
        for doc in documents:
            # Make everything lowercase and exclude special signs: All ; & > < = numbers : , . ' "
            doc["text"] = re.sub(r"([;]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|(\d+)|[']|[\"])", "", doc["raw_text"].lower())
            # Tokenize text.
            doc["tokenized_text"] = [pattern_vector.stem(word, stemmer=pattern_vector.LEMMA) for word in doc["text"].split()]

        # 3. Gather tokens.
        tokenized_documents = [doc["tokenized_text"] for doc in documents]

        # 4. Build dictionary.
        dictionary = gensim.corpora.Dictionary(tokenized_documents)
        # Remove stop words and words appearing only once.
        stopword_list = nltk.corpus.stopwords.words('english')
        # Add manual stopword list to default one.
        self.stopwords = ["automotive"]
        stopword_list += self.stopwords
        # Filter stopwords out of dictionary.
        ids_to_remove = [dictionary.token2id[stopword] for stopword in stopword_list if stopword in dictionary.token2id]
        dictionary.filter_tokens(ids_to_remove)
        # Filter words occuring only once (reasonable?).
        ids_to_remove = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(ids_to_remove)
        # Remove gaps in sequence after removing stopwords.
        dictionary.compactify()
        # todo Save dictionary in database as blob.

        # 5. Build iterable doc2bow corpus.
        doc2bow_corpus = IterableGensimDoc2BowCorpus(tokenized_documents=tokenized_documents, dictionary=dictionary)

        # Build tfidf-matrix.
        corpus_tfidf = gensim.models.TfidfModel(doc2bow_corpus)[doc2bow_corpus]
        print(corpus_tfidf)

        # 3. Store results in database.
            # cursor.execute("update "
            #                "topac.documents "
            #                "set"
            #                "    refined_text = %s "
            #                "where"
            #                "    id = %s",
            #                (refined_text, doc["db_id"]))



        # X. Commit transaction.
        db_connector.connection.commit()
