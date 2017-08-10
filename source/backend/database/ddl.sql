-- Created by Vertabelo (http://vertabelo.com)
-- Last modification date: 2017-08-10 15:31:33.892

create schema topac;

-- tables
-- Table: corpora
CREATE TABLE topac.corpora (
    id serial  NOT NULL,
    title text  NOT NULL,
    comment text  NULL,
    CONSTRAINT c_u_corpora_title UNIQUE (title) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT corpora_pk PRIMARY KEY (id)
);

-- Table: document_features
CREATE TABLE topac.document_features (
    id serial  NOT NULL,
    title text  NOT NULL,
    comment text  NULL,
    CONSTRAINT c_u_document_features UNIQUE (title) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT document_features_pk PRIMARY KEY (id)
);

-- Table: document_features_in_documents
CREATE TABLE topac.document_features_in_documents (
    documents_id int  NOT NULL,
    document_features_id int  NOT NULL,
    CONSTRAINT document_features_in_documents_pk PRIMARY KEY (documents_id,document_features_id)
);

-- Table: documents
CREATE TABLE topac.documents (
    id serial  NOT NULL,
    title text  NOT NULL,
    raw_text text  NOT NULL,
    refined_text text  NOT NULL,
    coordinates integer[]  NOT NULL,
    comment int  NULL,
    corpora_id int  NOT NULL,
    CONSTRAINT c_u_documents_title_corpora_id UNIQUE (title) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT documents_pk PRIMARY KEY (id)
);

-- Table: stopwords
CREATE TABLE topac.stopwords (
    id serial  NOT NULL,
    word text  NOT NULL,
    corpora_id int  NOT NULL,
    comment text  NULL,
    CONSTRAINT c_u_stopwords_word_corpora_id UNIQUE (word, corpora_id) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT stopwords_pk PRIMARY KEY (id)
);

-- Table: terms
CREATE TABLE topac.terms (
    id serial  NOT NULL,
    term text  NOT NULL,
    CONSTRAINT c_u_terms_term UNIQUE (term) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT terms_pk PRIMARY KEY (id)
);

-- Table: terms_in_corpora
CREATE TABLE topac.terms_in_corpora (
    corpora_id int  NOT NULL,
    terms_id int  NOT NULL,
    coordinates integer[]  NOT NULL,
    frequency int  NOT NULL,
    CONSTRAINT terms_in_corpora_pk PRIMARY KEY (corpora_id,terms_id)
);

-- Table: terms_in_topics
CREATE TABLE topac.terms_in_topics (
    terms_id int  NOT NULL,
    topics_id int  NOT NULL,
    probability real  NOT NULL CHECK (probability > 0),
    CONSTRAINT terms_in_topics_pk PRIMARY KEY (terms_id,topics_id)
);

-- Table: topic_models
CREATE TABLE topac.topic_models (
    id serial  NOT NULL,
    alpha real  NOT NULL,
    beta real  NOT NULL,
    kappa real  NOT NULL,
    corpora_id int  NOT NULL,
    runtime int  NOT NULL,
    coordinates integer[]  NOT NULL,
    comment text  NULL,
    CONSTRAINT c_u_topic_models_alpha_beta_kappa_corpora_id UNIQUE (alpha, beta, kappa, corpora_id) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT topic_models_pk PRIMARY KEY (id)
);

-- Table: topic_probabilities_in_documents
CREATE TABLE topac.topic_probabilities_in_documents (
    documents_id int  NOT NULL,
    topics_id int  NOT NULL,
    probability real  NOT NULL CHECK (probability > 0),
    CONSTRAINT topic_probabilities_in_documents_pk PRIMARY KEY (documents_id,topics_id)
);

-- Table: topics
CREATE TABLE topac.topics (
    id serial  NOT NULL,
    topic_number int  NOT NULL,
    title text  NOT NULL,
    topic_models_id int  NOT NULL,
    quality int  NOT NULL,
    coordinates integer[]  NOT NULL,
    document_features_id int  NOT NULL,
    comment text  NULL,
    CONSTRAINT c_u_topics_topic_number_document_feature_topic_models_id UNIQUE (topic_models_id, topic_number, document_features_id) NOT DEFERRABLE  INITIALLY IMMEDIATE,
    CONSTRAINT topics_pk PRIMARY KEY (id)
);

-- foreign keys
-- Reference: document_features_in_documents_document_features (table: document_features_in_documents)
ALTER TABLE topac.document_features_in_documents ADD CONSTRAINT document_features_in_documents_document_features
    FOREIGN KEY (document_features_id)
    REFERENCES topac.document_features (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: document_features_in_documents_documents (table: document_features_in_documents)
ALTER TABLE topac.document_features_in_documents ADD CONSTRAINT document_features_in_documents_documents
    FOREIGN KEY (documents_id)
    REFERENCES topac.documents (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: documents_corpora (table: documents)
ALTER TABLE topac.documents ADD CONSTRAINT documents_corpora
    FOREIGN KEY (corpora_id)
    REFERENCES topac.corpora (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: stopwords_corpora (table: stopwords)
ALTER TABLE topac.stopwords ADD CONSTRAINT stopwords_corpora
    FOREIGN KEY (corpora_id)
    REFERENCES topac.corpora (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: terms_in_corpora_corpora (table: terms_in_corpora)
ALTER TABLE topac.terms_in_corpora ADD CONSTRAINT terms_in_corpora_corpora
    FOREIGN KEY (corpora_id)
    REFERENCES topac.corpora (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: terms_in_corpora_terms (table: terms_in_corpora)
ALTER TABLE topac.terms_in_corpora ADD CONSTRAINT terms_in_corpora_terms
    FOREIGN KEY (terms_id)
    REFERENCES topac.terms (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: terms_in_topics_terms (table: terms_in_topics)
ALTER TABLE topac.terms_in_topics ADD CONSTRAINT terms_in_topics_terms
    FOREIGN KEY (terms_id)
    REFERENCES topac.terms (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: terms_in_topics_topics (table: terms_in_topics)
ALTER TABLE topac.terms_in_topics ADD CONSTRAINT terms_in_topics_topics
    FOREIGN KEY (topics_id)
    REFERENCES topac.topics (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: topic_models_corpora (table: topic_models)
ALTER TABLE topac.topic_models ADD CONSTRAINT topic_models_corpora
    FOREIGN KEY (corpora_id)
    REFERENCES topac.corpora (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: topics_document_features (table: topics)
ALTER TABLE topac.topics ADD CONSTRAINT topics_document_features
    FOREIGN KEY (document_features_id)
    REFERENCES topac.document_features (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: topics_in_documents_documents (table: topic_probabilities_in_documents)
ALTER TABLE topac.topic_probabilities_in_documents ADD CONSTRAINT topics_in_documents_documents
    FOREIGN KEY (documents_id)
    REFERENCES topac.documents (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: topics_in_documents_topics (table: topic_probabilities_in_documents)
ALTER TABLE topac.topic_probabilities_in_documents ADD CONSTRAINT topics_in_documents_topics
    FOREIGN KEY (topics_id)
    REFERENCES topac.topics (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- Reference: topics_topic_models (table: topics)
ALTER TABLE topac.topics ADD CONSTRAINT topics_topic_models
    FOREIGN KEY (topic_models_id)
    REFERENCES topac.topic_models (id)  
    NOT DEFERRABLE 
    INITIALLY IMMEDIATE
;

-- End of file.

