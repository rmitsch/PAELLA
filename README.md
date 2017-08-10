# TOPAC: Topic Prism Analytics Core

### Frontend
Plotting and crossfiltering.

### Backend
Handles backend calls from the frontend, aggregates analysis data and issues instructions to the TM engine. 

### Topic model engine
Using lda2vec. Spawned as distinct process using Python 2.7 (since lda2vec is written in that).