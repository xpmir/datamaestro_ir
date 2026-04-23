Information Retrieval Datasets
==============================

This section lists native IR dataset definitions.

MS MARCO Passage
----------------

The `MS MARCO <https://microsoft.github.io/msmarco/>`_ (Microsoft Machine Reading
Comprehension) Passage Ranking dataset. One of the most widely used benchmarks
for neural IR research.

Contains ~8.8M passages and ~500K training queries with sparse relevance judgments.

.. dm:datasets:: com.microsoft.msmarco.passage ir

Example usage:

.. code-block:: python

   from datamaestro import prepare_dataset
   from datamaestro.record import IDItem, TextItem

   # Load the full adhoc dataset
   adhoc = prepare_dataset("com.microsoft.msmarco.passage")

   # Iterate over documents
   for doc in adhoc.documents.iter_documents():
       doc_id = doc[IDItem].id
       text = doc[TextItem].text

   # Load training triplets
   triplets = prepare_dataset("com.microsoft.msmarco.passage.train.idstriples.small")
   for triplet in triplets.iter():
       query = triplet.query
       pos_doc = triplet.positive
       neg_doc = triplet.negative


BEIR Benchmark
--------------

The `BEIR <https://github.com/beir-cellar/beir>`_ (Benchmarking IR) benchmark
is a heterogeneous collection of diverse IR tasks for evaluating zero-shot
retrieval models. It includes datasets from question answering, fact
verification, citation prediction, and more.

.. dm:datasets:: org.beir ir

Example usage:

.. code-block:: python

   from datamaestro import prepare_dataset

   # Load a single-split dataset
   adhoc = prepare_dataset("org.beir.scidocs")

   # Load a multi-split dataset
   adhoc = prepare_dataset("org.beir.nfcorpus_test")

   # Access components
   for doc in adhoc.documents.iter():
       print(doc["id"], doc["text_item"].text)


LoTTE Benchmark
---------------

The `LoTTE <https://github.com/stanford-futuredata/ColBERT>`_ (Long-Tail
Topic-stratified Evaluation) benchmark from ColBERTv2. Contains 6 domains
(lifestyle, recreation, science, technology, writing, pooled) with dev/test
splits and two query types (search, forum) per split.

.. dm:datasets:: edu.stanford.lotte ir

Example usage:

.. code-block:: python

   from datamaestro import prepare_dataset

   # Load a specific task
   adhoc = prepare_dataset("edu.stanford.lotte.science_test_search")

   # Access components
   for doc in adhoc.documents.iter():
       print(doc["id"], doc["text_item"].text)


TIPSTER Collections
-------------------

The TIPSTER document collections used in TREC evaluations, organized by source.

.. dm:datasets:: gov.nist.trec.tipster ir


AQUAINT
-------

The AQUAINT Corpus consists of newswire text data in English from three sources:
Xinhua News Service, New York Times, and Associated Press.

.. dm:datasets:: edu.upenn.ldc.aquaint ir


TREC Ad Hoc
-----------

Classic TREC Ad Hoc test collections from NIST. These collections have been
fundamental benchmarks in IR research since the 1990s.

.. dm:datasets:: gov.nist.trec.adhoc ir

Example usage:

.. code-block:: python

   from datamaestro import prepare_dataset

   # Load TREC Adhoc dataset (e.g., TREC-8)
   adhoc = prepare_dataset("gov.nist.trec.adhoc.8")

   # Access components
   documents = adhoc.documents
   topics = adhoc.topics
   assessments = adhoc.assessments


TREC CAR
--------

The `TREC Complex Answer Retrieval <http://trec-car.cs.unh.edu/>`_ paragraph
corpus — ~29.8M paragraphs extracted from Wikipedia, used as a document
collection in several TREC tracks including CaST.

.. dm:datasets:: gov.nist.trec.car ir


Washington Post
---------------

`Washington Post <https://trec.nist.gov/data/wapost/>`_ document collections
used in several TREC tracks. These collections require a data-use agreement
with NIST and must be provided locally via ``DatafolderPath``.

.. dm:datasets:: gov.nist.trec.wapo ir
