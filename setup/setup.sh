#!/bin/sh

## Manual setup steps. ##

# Download NLTK's default stopwords us.
python -c "import nltk; nltk.download('stopwords')"
# Download NLTK tokenizers.
python -c "import nltk; nltk.download('punkt')"
# Download VADER's lexicon.
python -c "import nltk; nltk.download('vader_lexicon')"
