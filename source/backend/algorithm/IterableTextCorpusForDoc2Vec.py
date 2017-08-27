import gensim


class IterableTextCorpusForDoc2Vec(object):
    """
    Iterable used for gensim's doc2vec processing - to avoid holding all documents at once in memory.
    """

    def __init__(self,  db_result_set):
        """
        Initializes instance.
        :param db_result_set: Set of documents with the refined text in the first and an array of corpus feature IDs
        and facet IDs for the corresponding document in the second column.
        """

        self.db_result_set = db_result_set
        # Determine number of document tags based on first row.
        self.number_of_document_tags = len(db_result_set[0][2])

    def __iter__(self):
        """
        Yield words and labels (feature values) for all documents in provided DB set.
        :return:
        """

        for row in self.db_result_set:
            yield gensim.models.doc2vec.TaggedDocument(words=row[1].split(), tags=row[2])
