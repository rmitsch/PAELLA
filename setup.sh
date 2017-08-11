# @author RM
# @date 2017-08-11
# Additional setup necessary for TOPAC.

## Set up lda2vec.
# Execute install script.
python lda2vec/setup.py install
# Download spacy's english language model.
sudo python -m spacy.en.download