Distillation Datasets
=====================

This section lists datasets for knowledge distillation in neural IR,
where teacher model scores are used to train student rankers.


Pairwise Distillation
---------------------

Pairwise distillation datasets contain triples of (query, positive document,
negative document) with teacher model scores for each document.


Hofstaetter Neural Ranking KD
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Teacher scores for MS MARCO passage ranking from
`neural-ranking-kd <https://github.com/sebastian-hofstaetter/neural-ranking-kd>`_.
Contains ~40M triples with BERT-based teacher scores in TSV format
(pos_score, neg_score, query_id, pos_passage_id, neg_passage_id).

.. dm:datasets:: com.github.hofstaetter.distillation ir


Listwise Distillation
---------------------

Listwise distillation datasets contain ranked lists of documents for each query,
produced by a teacher model.


Rank-DistilLM
~~~~~~~~~~~~~~

Ranked passage lists from `rank-distillm <https://github.com/webis-de/rank-distillm>`_
for MS MARCO training queries. Includes BM25 and ColBERTv2 first-stage retrieval
results, as well as RankZephyr reranked lists at various cutoffs.

.. dm:datasets:: com.github.webis-de.rank-distillm ir
