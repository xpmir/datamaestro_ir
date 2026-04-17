Information Retrieval API
=========================

This module provides data types for Information Retrieval datasets and experiments.

The core abstractions are:

- **Documents** - Collections of documents to be searched
- **Topics** - Queries or information needs
- **Assessments** - Relevance judgments (qrels) linking topics to relevant documents
- **Adhoc** - A complete IR test collection combining documents, topics, and assessments

For training neural rankers:

- **TrainingTriplets** - Training data as (query, positive_doc, negative_doc) triplets
- **PairwiseSampleDataset** - General pairwise training data


Data objects
------------

.. automodule:: datamaestro_ir.data.base
   :members:

Collection
----------

.. autoxpmconfig:: datamaestro_ir.data.Adhoc

Topics
------

.. autoxpmconfig:: datamaestro_ir.data.Topics
    :members: iter, count

.. autoxpmconfig:: datamaestro_ir.data.csv.Topics
.. autoxpmconfig:: datamaestro_ir.data.FilteredTopics

.. autoxpmconfig:: datamaestro_ir.transforms.TopicWrapper

Dataset-specific Topics
-----------------------

.. autoxpmconfig:: datamaestro_ir.data.beir.BeirTopics
.. autoxpmconfig:: datamaestro_ir.data.beir.BeirParquetTopics
.. autoxpmconfig:: datamaestro_ir.data.lotte.LotteTopics
.. autoxpmconfig:: datamaestro_ir.data.trec.TrecTopics
.. autoxpmconfig:: datamaestro_ir.data.cord19.Topics

Documents
---------

.. autoxpmconfig:: datamaestro_ir.data.Documents
    :members: iter_documents, iter_ids, documentcount
.. autoxpmconfig:: datamaestro_ir.data.csv.Documents


Dataset-specific documents
**************************

.. autoxpmconfig:: datamaestro_ir.data.cord19.Documents
.. autoxpmconfig:: datamaestro_ir.data.trec.TipsterCollection
.. autoxpmconfig:: datamaestro_ir.data.beir.BeirDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.lotte.LotteDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.stores.MsMarcoPassagesStore
.. autoxpmconfig:: datamaestro_ir.data.stores.MsMarcoPassageV2Store
.. autoxpmconfig:: datamaestro_ir.data.stores.CarParagraphStore
.. autoxpmconfig:: datamaestro_ir.data.stores.WapoDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.stores.WapoPassageStore
.. autoxpmconfig:: datamaestro_ir.data.stores.KiltDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.stores.MsMarcoDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.stores.MsMarcoDocumentV2Store
.. autoxpmconfig:: datamaestro_ir.data.stores.CastSegmentedPassageStore
.. autoxpmconfig:: datamaestro_ir.data.stores.TipsterDocumentStore
.. autoxpmconfig:: datamaestro_ir.data.PrefixedDocumentStore

Assessments
-----------

.. autoxpmconfig:: datamaestro_ir.data.AdhocAssessments
    :members:

.. autoxpmconfig:: datamaestro_ir.data.beir.BeirAssessments
.. autoxpmconfig:: datamaestro_ir.data.beir.BeirParquetAssessments
.. autoxpmconfig:: datamaestro_ir.data.lotte.LotteAssessments
.. autoxpmconfig:: datamaestro_ir.data.trec.TrecAdhocAssessments

.. autoclass:: datamaestro_ir.data.AdhocAssessedTopic
.. autoclass:: datamaestro_ir.data.AdhocAssessment

Runs
----

.. autoxpmconfig:: datamaestro_ir.data.AdhocRun
.. autoxpmconfig:: datamaestro_ir.data.csv.AdhocRunWithText
.. autoxpmconfig:: datamaestro_ir.data.trec.TrecAdhocRun


Results
-------

.. autoxpmconfig:: datamaestro_ir.data.AdhocResults
.. autoxpmconfig:: datamaestro_ir.data.trec.TrecAdhocResults
    :members: get_results

Evaluation
----------

.. autoxpmconfig:: datamaestro_ir.data.Measure


Reranking
---------

.. autoxpmconfig:: datamaestro_ir.data.RerankAdhoc

Document Index
---------------

.. autoxpmconfig:: datamaestro_ir.data.DocumentStore
    :members: documentcount, docid_internal2external, document_int, document_ext, iter_sample

.. autoxpmconfig:: datamaestro_ir.data.CompressedDocumentStore

.. autoxpmconfig:: datamaestro_ir.data.AdhocIndex
    :members: termcount, term_df

.. autoxpmconfig:: datamaestro_ir.data.anserini.Index


Training triplets
-----------------


.. autoxpmconfig:: datamaestro_ir.data.TrainingTriplets
    :members: iter

.. autoxpmconfig:: datamaestro_ir.data.PairwiseSampleDataset
    :members: iter

.. autoxpmconfig:: datamaestro_ir.data.TrainingTripletsLines

.. autoxpmconfig:: datamaestro_ir.data.huggingface.HuggingFacePairwiseSampleDataset

Distillation
************

.. autoxpmconfig:: datamaestro_ir.data.distillation.PairwiseDistillationSamples
.. autoxpmconfig:: datamaestro_ir.data.distillation.PairwiseDistillationSamplesTSV
.. autoxpmconfig:: datamaestro_ir.data.distillation.ListwiseDistillationSamples
.. autoxpmconfig:: datamaestro_ir.data.distillation.ListwiseDistillationSamplesTSV
.. autoxpmconfig:: datamaestro_ir.data.distillation.ListwiseDistillationSamplesTSVWithAnnotations

Transforms
**********

.. autoxpmconfig:: datamaestro_ir.transforms.StoreTrainingTripletTopicAdapter

.. autoxpmconfig:: datamaestro_ir.transforms.StoreTrainingTripletDocumentAdapter

.. autoxpmconfig:: datamaestro_ir.transforms.ShuffledTrainingTripletsLines
