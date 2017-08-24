class IterableGensimDoc2BowCorpus(object):
    """
    Iterable used for gensim's streaming processing - to avoid holding all documents at once in memory.
    """

    def __init__(self,  tokenized_documents, dictionary):
        """
        Initializes instance.
        :param tokenized_documents: Tokenized documents.
        :param dictionary: Gensim dictionary generated using the tokenized documents; filtered and preprocessed.
        """
        self.tokenized_documents = tokenized_documents
        self.dictionary = dictionary

    def __iter__(self):
        """
        Yield doc2bow for tokenized documents when iterated over.
        :return:
        """
        for line in self.tokenized_documents:
            # Assume there's one document per line, tokens separated by spaces.
            yield self.dictionary.doc2bow([x.strip() for x in line])
