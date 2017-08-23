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
from sumy.summarizers.text_rank import TextRankSummarizer as Sumy_TextRankSummarizer
from sumy.nlp.stemmers import Stemmer as Sumy_Stemmer
from sumy.parsers.plaintext import PlaintextParser as Sumy_PlaintextParser
from sumy.nlp.tokenizers import Tokenizer as Sumy_Tokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class Corpus:
    """
    Class representing a corpus in its entirety.
    Used for importing corpora.
    """
    # Supported corpora types.
    # Should ultimately boil down to one single, standardized CSV template.
    supportedCorporaTypes = ['nltk-reuters']

    def __init__(self, name, corpus_type, corpus_features, stopwords, summarization_word_count=50):
        """
        Init instance of Corpus.
        :param name:
        :param corpus_type: Structural type of corpus. Should ultimately converge to unified format.
        :param corpus_features: List of dictionaries containing values describing corpus features: name, type.
                                A unique ID is inferred automatically from DB IDs and doesn't have to be specified.
                                Note that names have to coincide with column names containing those features.
        :param stopwords: List of stopwords to remove before generating models.
        :param summarization_word_count: Number of words to use for summarization of document in this corpus.
        """
        self.logger = logging.getLogger("topac")

        self.logger.info("Initializing Corpus object.")

        self.name = name
        self.corpus_type = corpus_type
        # Initialize document collection with empty dataframe.
        self.documents = {}
        self.stopwords = stopwords
        self.summarization_word_count = summarization_word_count
        # Create list of corpus_features.
        self.corpus_features = {
            "document_id": {"name": "document_id", "type": "int"}
        }
        for corpus_feature in corpus_features:
            self.corpus_features[corpus_feature["name"]] = {"name": corpus_feature["name"],
                                                            "type": corpus_feature["type"]}

        # Check if corpus_type is supported.
        if corpus_type not in Corpus.supportedCorporaTypes:
            self.logger.error("Corpus type not supported")

    def compile(self, path, db_connector):
        """
        Preprocesses and mports specified corpus.
        :param path: Path to file/folder where corpus is located.
        :param db_connector:
        :return:
        """

        self.logger.info("Compiling and importing corpus of type " + self.corpus_type)

        if self.corpus_type == "nltk-reuters":
            self.import_nltk_reuters_corpus(db_connector)
        else:
            print("blab")

    def import_nltk_reuters_corpus(self, db_connector):
        """
        Import nltk-reuter corpus.
        :param db_connector:
        :return:
        """

        # Prepare cursor.
        cursor = db_connector.connection.cursor()

        # Preprare VADER's sentiment analyzer.
        sentiment_analyzer = SentimentIntensityAnalyzer()

        # Prepare list of documents. Used for preprocessing.
        documents = []

        # Import corpus.
        corpus_id = Corpus.import_corpus(cursor, self.name)

        # Import corpus features.
        for corpus_feature_name, corpus_feature in self.corpus_features.items():
            corpus_feature_id = Corpus.import_corpus_feature(cursor=cursor,
                                                             corpus_id=corpus_id,
                                                             feature_title=corpus_feature_name,
                                                             feature_type=corpus_feature["type"])
            # Update dict for corpus features.
            self.corpus_features[corpus_feature_name]["id"] = corpus_feature_id

        # Define and import stopwords.
        self.generate_and_import_stopwords(cursor, corpus_id)

        # Preprocess and persist documents.
        for fileID in nltk.corpus.reuters.fileids():
            # Fetch document.
            doc = nltk.corpus.reuters.raw(fileids=[fileID]).strip()

            # Import document in DB.
            document_id = Corpus.import_document(cursor=cursor,
                                                 corpus_id=corpus_id,
                                                 title=fileID,
                                                 raw_text=doc,
                                                 sentiment_analyzer=sentiment_analyzer)

            # Import document feature values.
            # Contextual document ID / title.
            Corpus.import_corpus_feature_in_document(cursor=cursor,
                                                     corpus_feature_id=self.corpus_features["document_id"]["id"],
                                                     document_id=document_id,
                                                     value=document_id)
            # Categories.
            Corpus.import_corpus_feature_in_document(cursor=cursor,
                                                     corpus_feature_id=self.corpus_features["categories"]["id"],
                                                     document_id=document_id,
                                                     value=nltk.corpus.reuters.categories(fileids=[fileID]))

            # Add to list of documents.
            # Note: With generic corpus, data has to be loaded from corresponding column in dataframe/dict.
            new_document = {"id": document_id,
                            "raw_text": doc,
                            "features": {
                                self.corpus_features["document_id"]["name"]: document_id,
                                self.corpus_features["categories"]["name"]:
                                    nltk.corpus.reuters.categories(fileids=[fileID])[0]
                            }}
            documents.append(new_document)

        # Commit and store corpus with raw text in DB.
        db_connector.connection.commit()

        # Refine corpus text.
        self.preprocess_corpus(db_connector=db_connector,
                               documents=documents,
                               corpus_id=corpus_id,
                               corpus_features=self.corpus_features)

    @staticmethod
    def import_corpus(cursor, corpus_name):
        """
        Import corpus in database.
        :param cursor:
        :param corpus_name:
        :return: ID of new corpus.
        """

        cursor.execute("insert into "
                       "    topac.corpora ("
                       "    title"
                       ") "
                       "values (%s) "
                       "returning id",
                       (corpus_name,))

        return cursor.fetchone()[0]

    @staticmethod
    def import_corpus_feature(cursor, corpus_id, feature_title, feature_type):
        """
        Adds entry in topac.corpus_features.
        :param cursor:
        :param corpus_id:
        :param feature_title:
        :param feature_type:
        :return: ID of newly added corpus feature.
        """
        cursor.execute("insert into "
                       "    topac.corpus_features ("
                       "        title, "
                       "        type, "
                       "        corpora_id"
                       ")"
                       "values (%s, %s, %s) "
                       "returning id",
                       (feature_title, feature_type, corpus_id))

        return cursor.fetchone()[0]

    @staticmethod
    def import_document(cursor, corpus_id, title, raw_text, sentiment_analyzer):
        """
        Adds entry in topac.documents.
        :param cursor:
        :param corpus_id:
        :param title:
        :param raw_text:
        :param sentiment_analyzer: VADER's sentiment analyzer.
        :return: ID of newly added document.
        """

        cursor.execute("insert into "
                       "topac.documents ("
                       "    title, "
                       "    raw_text, "
                       "    corpora_id, "
                       "    sentiment_score "
                       ")"
                       "values (%s, %s, %s, %s) "
                       "returning id",
                       (title,
                        raw_text,
                        corpus_id,
                        sentiment_analyzer.polarity_scores(raw_text)["compound"]))

        return cursor.fetchone()[0]

    @staticmethod
    def import_corpus_feature_in_document(cursor, corpus_feature_id, document_id, value):
        """
        Adds entry in topac.corpus_features_in_documents.
        :param cursor:
        :param corpus_feature_id:
        :param document_id:
        :param value:
        :return: ID of newly generated entry.
        """

        cursor.execute("insert into "
                       "topac.corpus_features_in_documents ("
                       "    corpus_features_id, "
                       "    documents_id, "
                       "    value"
                       ") "
                       "values (%s, %s, %s) "
                       "returning id",
                       (corpus_feature_id, document_id, value))

        return cursor.fetchone()[0]

    def generate_and_import_stopwords(self, cursor, corpus_id):
        """
        Define and import stopwords.
        :param cursor:
        :param corpus_id:
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

    def preprocess_corpus(self, db_connector, documents, corpus_id, corpus_features):
        """
        Preprocess and refine raw texts for entire corpus and all corpus features.
        :param db_connector:
        :param documents:
        :param corpus_id:
        :param corpus_features:
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
            doc["text"] = re.sub(r"([;]|[(]|[)]|[/]|[\\]|[$]|[&]|[>]|[<]|[=]|[:]|[,]|[.]|[-]|(\d+)|[']|[\"])", "",
                                 doc["raw_text"].lower())
            # Tokenize text.
            doc["tokenized_text"] = [pattern_vector.stem(word, stemmer=pattern_vector.LEMMA)
                                     for word in doc["text"].split()]
            # Remove stopwords from text.
            doc["text"] = ' '.join(filter(lambda x: x not in self.stopwords, doc["tokenized_text"]))

            # Update documents in DB. Again: Performance could be sped up, but bottleneck is most likely not
            # document preprocessing.
            cursor.execute("update topac.documents "
                           "set"
                           "    refined_text = %s "
                           "where"
                           "    id = %s ",
                           (doc["text"], doc["id"]))

        # ---------------------
        # 3. For topic models: Preprocess corpus and store relevant information for each corpus feature.
        # ---------------------
        for corpus_feature_name, corpus_feature in corpus_features.items():
            dictionary = self.preprocess_corpus_feature(cursor=cursor,
                                                        documents=documents,
                                                        corpus_feature=corpus_feature,
                                                        stopwords=self.stopwords)

            # If current feature (and dictionary) is document ID: Store terms and term-corpus associations
            # in database.
            if corpus_feature["name"] == "document_id":
                Corpus.import_terms(cursor=cursor,
                                    dictionary=dictionary,
                                    corpus_id=corpus_id)

        # ---------------------
        # x. Commit transaction.
        # ---------------------
        db_connector.connection.commit()

    def preprocess_corpus_feature(self, cursor, documents, corpus_feature, stopwords):
        """
        Preprocess and refine raw texts for selected corpus feature.
        :param cursor:
        :param documents:
        :param corpus_feature:
        :param stopwords:
        :return: gensim-dictionary created from aggregated documents.
        """

        # ---------------------
        # 1. Concatenate documents with same value for current corpus_feature to one document.
        # ---------------------
        tokenized_merged_documents_dict, raw_merged_documents_dict = \
            Corpus.merge_tokenized_document_texts_by_feature_value(
                documents=documents,
                corpus_feature=corpus_feature
            )

        # Note that feature values and merged documents are sorted in the same order -
        # hence it's possible later on to use the sequence of feature values to determine
        # topic-probability associations.
        feature_values = list(tokenized_merged_documents_dict.keys())
        tokenized_merged_documents = list(tokenized_merged_documents_dict.values())

        # ---------------------
        # 2. Store sequence feature_values (for later retrieval of topic-document probabilities after TM building.
        # ---------------------
        cursor.execute("update topac.corpus_features "
                       "set "
                       "    feature_value_sequence = %s "
                       "where id = %s",
                       (feature_values, corpus_feature["id"]))

        # ---------------------
        # 3. Store corpus facets.
        # ---------------------
        # Prepare sumy summarizer.
        # summarizer = Sumy_TextRankSummarizer(Sumy_Stemmer("english"))
        # summarizer.stop_words = self.stopwords

        for feature_value, raw_merged_document in raw_merged_documents_dict.items():
            # parser = Sumy_PlaintextParser.from_string(raw_merged_document, Sumy_Tokenizer("english"))
            # blub = []
            # for sentence in summarizer(parser.document, 10):
            #    blub.append(sentence)

            # Insert facet.
            cursor.execute("insert into "
                           "    topac.corpus_facets ("
                           "    corpus_features_id, "
                           "    corpus_feature_value, "
                           "    summarized_text"
                           ") "
                           "values (%s, %s, %s)",
                           (corpus_feature["id"],
                            feature_value,
                            # todo Which summarizer to use? Runtime not critical, but relevant.
                            ""
                            # gensim.summarization.summarize(raw_merged_document, self.summarization_word_count)
                            ))

        # ---------------------
        # 4. Build dictionary.
        # ---------------------
        dictionary = gensim.corpora.Dictionary(tokenized_merged_documents)
        # Filter stopwords out of dictionary.
        ids_to_remove = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
        dictionary.filter_tokens(ids_to_remove)
        # Filter words occuring only once (reasonable?).
        ids_to_remove = [tokenid for tokenid, docfreq in dictionary.dfs.items() if docfreq == 1]
        dictionary.filter_tokens(ids_to_remove)
        # Remove gaps in sequence after removing stopwords.
        dictionary.compactify()

        # ---------------------
        # 5. Build and save corpus for LDA-based topic modeling.
        # ---------------------
        Corpus.generate_and_import_gensim_corpus(cursor=cursor,
                                                 tokenized_documents=tokenized_merged_documents,
                                                 dictionary=dictionary,
                                                 corpus_feature_id=corpus_feature["id"])

        return dictionary

    @staticmethod
    def merge_tokenized_document_texts_by_feature_value(documents, corpus_feature):
        """
        Merges tokenized document texts by their documents' values for the specified feature.
        :param documents:
        :param corpus_feature:
        :return: Lists of merged (1) tokenized document texts and (2) raw document texts.
        """

        merged_tokenized_document_texts_by_feature = {}
        merged_raw_document_texts_by_feature = {}

        # Iterate over all documents and group their tokenized_texts by feature value.
        for document in documents:
            feature_value = document["features"][corpus_feature["name"]]

            # Group documents by feature value.
            if feature_value in merged_tokenized_document_texts_by_feature:
                merged_tokenized_document_texts_by_feature[feature_value].extend(document["tokenized_text"])
                merged_raw_document_texts_by_feature[feature_value] += " " + document["raw_text"]
            else:
                merged_tokenized_document_texts_by_feature[feature_value] = document["tokenized_text"]
                merged_raw_document_texts_by_feature[feature_value] = document["raw_text"]

        return merged_tokenized_document_texts_by_feature, merged_raw_document_texts_by_feature

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
    def generate_and_import_gensim_corpus(cursor, tokenized_documents, dictionary, corpus_feature_id):
        """
        Generates and imports corpus in DB.
        :param cursor: Cursor to current DB transaction.
        :param tokenized_documents:
        :param dictionary:
        :param corpus_feature_id:
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
        # todo Some space might be saved by using gensim's binary save/load interface
        # (http://radimrehurek.com/gensim/utils.html#gensim.utils.SaveLoad) for the corpus as well.
        dictionary.save("tmp_dict_file.dict")
        gensim.corpora.MmCorpus.serialize("tmp_corpus_file.mm", tfidf_corpus)

        # Transfer files to database.
        cursor.execute("update topac.corpus_features "
                       "set "
                       "    gensim_dictionary = %s, "
                       "    gensim_corpus = %s "
                       "where "
                       "    id = %s",
                       (
                           open("tmp_dict_file.dict", "rb").read(),
                           open("tmp_corpus_file.mm", "rb").read(),
                           corpus_feature_id
                       ))

        # Remove temporary files.
        os.remove("tmp_dict_file.dict")
        os.remove("tmp_corpus_file.mm")
        os.remove("tmp_corpus_file.mm.index")