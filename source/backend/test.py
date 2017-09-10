# @author RM
# @date 2017-08-08
# Playground for testing import and other functionalities.
#
#
# NEXT STEPS
#   - Calculate and persist t-SNE positions for words and documents
#       * Append facet vectors to word vectors; store index of appended line (important!) in dictionary.
#       * Apply t-SNE.
#       * Iterate over term and word dicitionaries, create tupels for DB updates/inserts with coordinates in t-SNE
#         result matrix' corresponding line.
#       * Store term coordinates in DB.
#       * Make sure every term has coordinates (and that there are no terms having coordinates but are not persisted).
#   - Calculate and persist t-SNE positions for topics
#   - Description of use cases
#   - How to handle summarization per facet?
#     https://rare-technologies.com/text-summarization-in-python-extractive-vs-abstractive-techniques-revisited/
#     https://github.com/miso-belica/sumy/

import logging
import backend.database.DBConnector as DBConnector
import backend.utils.Utils as Utils
import backend.algorithm.Corpus as Corpus
import backend.algorithm.TopicModel as TopicModel
import backend.algorithm.Doc2VecModel as Doc2VecModel
import gensim

# Note: Reserved keywords for feature columns are id, raw_text.

# Create logger.
Utils.create_logger()
logger = logging.getLogger("topac")
logger.info("Starting test.py.")

# Create database connection.
db_connector = DBConnector(host="localhost",
                           database="topac",
                           port="8001",
                           user="admin",
                           password="password")

# Set corpus title.
corpus_title = "nltk-reuters"

# # Create database.
# db_connector.construct_database()
# # Truncate database.
# db_connector.truncate_database()
db_connector.truncate_topic_tables()
#
# # Import nltk-reuters corpus.
# # Define which corpus-features should be used.
# corpus_features = [
#     {"name": "categories", "type": "text"}
# ]
# nltk_reuters_corpus = Corpus(name=corpus_title,
#                              corpus_type="nltk-reuters",
#                              stopwords=['they'],
#                              corpus_features=corpus_features)
# nltk_reuters_corpus.compile("", db_connector)

# # Create new doc2vec model. Omit most hyperparameters for now.
# doc2vec_model = Doc2VecModel(db_connector=db_connector,
#                              corpus_title=corpus_title,
#                              alpha=0.05,
#                              n_workers=2,
#                              n_epochs=1,
#                              n_window=1,
#                              feature_vector_size=5)
# # Compile doc2vec model.
# doc2vec_model.compile()

# Fetch term-coordinate dictionary.
term_coordinates_dict = db_connector.load_term_coordinates(doc2vec_model_id=1)
# Fetch and instantiate doc2vec model (for testing purposes; in production: Fetch once before generating batch of topic
# models).
with open("tmp_word_embedding.d2v", "wb") as tmp_d2v_file:
    tmp_d2v_file.write(db_connector.fetch_doc2vec_gensim_model(doc2vec_model_id=1))
doc2vec_gensim_model = gensim.models.Doc2Vec.load("tmp_word_embedding.d2v")

# Create new topic model. Omit most hyperparameters for now.
topic_model = TopicModel(db_connector=db_connector,
                         corpus_title=corpus_title,
                         corpus_feature_title="document_id",
                         n_iterations=10,
                         n_workers=2,
                         doc2vec_model=doc2vec_gensim_model,
                         term_coordinates_dict=term_coordinates_dict)
# Calculate/compile topic model.
topic_model.compile()

# Close database connection.
db_connector.connection.close()

logger.info("Finished test.py.")