# @author RM
# @date 2017-08-09
# Class for connecting to database. Uses exactly one connection.
#

import os
import nltk as nltk
from pattern3 import vector as pattern_vector
import logging
import re
import gensim
from backend.algorithm.IterableGensimDoc2BowCorpus import IterableGensimDoc2BowCorpus
from psycopg2 import Binary
from psycopg2 import InternalError

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
        cursor.execute("insert into "
                       "    topac.corpora ("
                       "    title"
                       ") "
                       "values (%s) "
                       "returning id",
                       (self.name,))
        corpus_id = cursor.fetchone()[0]

        # todo Refactor into individual functions for each table.
        # Import corpus features.
        # Categories.
        cursor.execute("insert into "
                       "    topac.corpus_features ("
                       "        title, "
                       "        type, "
                       "        corpora_id"
                       ")"
                       "values ('categories', 'text', %s) "
                       "returning id",
                       (corpus_id,))
        corpus_categories_feature_id = cursor.fetchone()[0]
        # Document IDs.
        cursor.execute("insert into "
                       "    topac.corpus_features ("
                       "        title, "
                       "        type, "
                       "        corpora_id"
                       ")"
                       "values ('document_ids', 'int', %s) "
                       "returning id",
                       (corpus_id,))
        corpus_document_ids_feature_id = cursor.fetchone()[0]

        # Define and import stopwords.
        self.generate_and_import_stopwords(cursor, corpus_id)

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
            # Categories.
            cursor.execute("insert into "
                           "topac.corpus_features_in_documents ("
                           "    corpus_features_id, "
                           "    documents_id, "
                           "    value"
                           ") "
                           "values (%s, %s, %s)",
                           (corpus_categories_feature_id, document_id, nltk.corpus.reuters.categories(fileids=[fileID])))
            # Contextual document ID / title.
            cursor.execute("insert into "
                           "topac.corpus_features_in_documents ("
                           "    corpus_features_id, "
                           "    documents_id, "
                           "    value"
                           ") "
                           "values (%s, %s, %s)",
                           (corpus_document_ids_feature_id, document_id, document_id))

            # Add to list of documents.
            documents.append({"db_id": document_id, "raw_text": doc})

        # Commit and store corpus with raw text in DB.
        db_connector.connection.commit()

        # Refine corpus text.
        self.preprocess_corpus(db_connector, documents, corpus_id)

    def generate_and_import_stopwords(self, cursor, corpus_id):
        """
        Define and import stopwords.
        :param cursor:
        :param cursor_id:
        :return:
        """

        self.stopwords = nltk.corpus.stopwords.words('english')
        self.stopwords.extend(self.stopwords)

        for stopword in self.stopwords:
            cursor.execute("insert into "
                           "    topac.stopwords ("
                           "        word, "
                           "        corpora_id "
                           ") "
                           "values(%s, %s) "
                           "on conflict "
                           "do nothing",
                           (stopword, corpus_id))

    def preprocess_corpus(self, db_connector, documents, corpus_id):
        """
        Preprocess and refine raw texts for entire corpus.
        :param db_connector:
        :param documents:
        :param corpus_id:
        :return:
        """
        # ---------------------
        # 1. Prepare cursor.
        # ---------------------
        cursor = db_connector.connection.cursor()

        # ---------------------
        # 2. Loop over documents, remove unwanted characters.
        # ---------------------
        # Note: Assuming performance isn't critical for document import.
        for doc in documents:
            # Make everything lowercase and exclude special signs: All ; & > < = numbers : , . ' "
            doc["text"] = re.sub(r"([;]|[(]|[)]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|[-]|(\d+)|[']|[\"])", "", doc["raw_text"].lower())
            # Tokenize text.
            doc["tokenized_text"] = [pattern_vector.stem(word, stemmer=pattern_vector.LEMMA) for word in doc["text"].split()]
            # Remove stopwords from text.
            doc["text"] = ' '.join(filter(lambda x: x not in self.stopwords, doc["tokenized_text"]))

            # Update documents in DB. Again: Performance could be sped up, but bottleneck is most likely not
            # document preprocessing.
            cursor.execute("update topac.documents "
                           "set"
                           "    refined_text = %s "
                           "where"
                           "    id = %s ",
                           (doc["text"], doc["db_id"]))

        # ---------------------
        # 3. Gather tokens.
        # ---------------------
        tokenized_documents = [doc["tokenized_text"] for doc in documents]

        # ---------------------
        # 4. Build dictionary.
        # ---------------------
        dictionary = gensim.corpora.Dictionary(tokenized_documents)
        # Filter stopwords out of dictionary.
        ids_to_remove = [dictionary.token2id[stopword] for stopword in self.stopwords if stopword in dictionary.token2id]
        dictionary.filter_tokens(ids_to_remove)
        # Filter words occuring only once (reasonable?).
        ids_to_remove = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(ids_to_remove)
        # Remove gaps in sequence after removing stopwords.
        dictionary.compactify()

        # ---------------------
        # 5. Import terms and term-corpus associations.
        # ---------------------
        Corpus.import_terms(cursor=cursor, dictionary=dictionary, corpus_id=corpus_id)

        # ---------------------
        # 6. Build and save corpus for LDA-based topic modeling.
        # ---------------------
        Corpus.generate_and_import_corpus(cursor=cursor,
                                          tokenized_documents=tokenized_documents,
                                          dictionary=dictionary,
                                          corpus_id=corpus_id)

        # ---------------------
        # x. Commit transaction.
        # ---------------------
        db_connector.connection.commit()

    @staticmethod
    def import_terms(cursor, dictionary, corpus_id):
        """
        Import terms into database.
        :param cursor:
        :param dictionary:
        :param corpus_id:
        :return:
        """

        # 1. Get all existing terms in database, store in dictionary.
        cursor.execute("select"
                       "    id, "
                       "    term "
                       "from"
                       "    topac.terms")
        res = cursor.fetchall()

        existing_terms = {}
        for row in res:
            existing_terms[row[1]] = row[0]

        # 2. Gather words to store in DB.
        # Check which words are not in table yet.
        tuples_to_insert = []
        for id, word in dictionary.iteritems():
            if word not in existing_terms.keys():
                tuples_to_insert.append((word,))

        # 3. Write terms to DB.
        cursor.execute(cursor.mogrify("insert into "
                                      "  topac.terms (term)"
                                      " values " +
                                      ','.join(["%s"] * len(tuples_to_insert)), tuples_to_insert))

        # 4. Generate lookup table for terms.
        cursor.execute("select"
                       "    id, "
                       "    term "
                       "from "
                       "    topac.terms")
        res = cursor.fetchall()

        term_dict = {}
        for row in res:
            term_dict[row[1]] = row[0]

        # 5. Generate tuples.
        del tuples_to_insert [:]
        for id, word in dictionary.iteritems():
            tuples_to_insert.append((corpus_id, term_dict[word]))

        # 6. Write terms_in_corpora to DB.
        cursor.execute(cursor.mogrify("insert into "
                                      "  topac.terms_in_corpora (corpora_id, terms_id)"
                                      " values " +
                                      ','.join(["%s"] * len(tuples_to_insert)), tuples_to_insert))

    @staticmethod
    def generate_and_import_corpus(cursor, tokenized_documents, dictionary, corpus_id):
        """
        Generates and imports corpus in DB.
        :param cursor: Cursor to current DB transaction.
        :param tokenized_documents:
        :param dictionary:
        :param corpus_id:
        :return:
        """

        # ---------------------
        # 1. Build corpus usable by gensim.
        # ---------------------
        doc2bow_corpus = IterableGensimDoc2BowCorpus(tokenized_documents=tokenized_documents, dictionary=dictionary)
        # Build tfidf-matrix.
        tfidf_corpus = gensim.models.TfidfModel(doc2bow_corpus)[doc2bow_corpus]

        # ---------------------
        # 2. Save dictionary and doc2bow-corpus to database.
        # ---------------------

        # Intermediate step: Save dictionary and corpus to temporary files.
        dictionary.save("tmp_dict_file.dict")
        gensim.corpora.MmCorpus.serialize("tmp_corpus_file.mm", tfidf_corpus)

        # Transfer files to database.
        cursor.execute("update topac.corpora "
                       "set "
                       "    gensim_dictionary = %s, "
                       "    gensim_corpus = %s "
                       "where "
                       "    id = %s",
                       (
                           Binary(open("tmp_dict_file.dict", "rb").read()),
                           Binary(open("tmp_corpus_file.mm", "rb").read()),
                           corpus_id
                       ))

        # Remove temporary files.
        os.remove("tmp_dict_file.dict")
        os.remove("tmp_corpus_file.mm")
        os.remove("tmp_corpus_file.mm.index")


# Snippet for loading and executing LDA from data stored in DB
# # test loading
# cursor.execute("select"
#                "    gensim_dictionary, gensim_corpus "
#                "from topac.corpora "
#                "where "
#                "    id = %s",
#                (corpus_id,))
# res = cursor.fetchone()
# open("tmp_dict_file.dict", "wb+").write(res[0])
# open("tmp_corpus_file.mm", "wb+").write(res[1])
# testdict = gensim.corpora.Dictionary.load("tmp_dict_file.dict")
# testcorpus = gensim.corpora.MmCorpus("tmp_corpus_file.mm")
#
# # Train LDA model.
# lda = gensim.models.LdaMulticore(corpus=testcorpus,
#                                  workers=2,
#                                  id2word=testdict,
#                                  num_topics=10)
# for topic in lda.show_topics():
#     print(topic)