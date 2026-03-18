Conversational IR Datasets
==========================

This section lists datasets for conversational information retrieval
and contextual query understanding tasks.


Conversational Search
---------------------

These datasets evaluate multi-turn conversational search systems where
users engage in conversations to satisfy complex information needs.


TREC CaST
~~~~~~~~~~

The `TREC Conversational Assistance Track <https://www.treccast.ai/>`_ (CaST)
evaluates conversational information seeking over multi-turn dialogues.
Runs from 2019 to 2022 with evolving document collections across versions.

.. dm:datasets:: gov.nist.trec.cast ir


iKAT
~~~~

The `iKAT <https://github.com/irlabamsterdam/iKAT>`_ (Interactive Knowledge
Assistance Track) datasets for conversational search and query rewriting,
using the ClueWeb22 document collection. Runs from 2023 to 2025.

.. dm:datasets:: com.github.ikat ir


Contextual Query Rewriting
--------------------------

These datasets contain conversational queries that need to be rewritten
to be self-contained (decontextualization), resolving coreferences and
ellipses from the conversation context.


CANARD
~~~~~~

Context-dependent Query Rewriting dataset for conversational question answering.
Contains queries from QuAC that have been manually rewritten to be self-contained.

.. dm:datasets:: com.github.aagohary.canard ir

Example:

.. code-block:: python

   from datamaestro import prepare_dataset

   canard = prepare_dataset("com.github.aagohary.canard.train")
   for entry in canard.iter():
       print(f"Original: {entry.source}")
       print(f"Rewritten: {entry.rewrite}")


OrConvQA
~~~~~~~~

Open-Retrieval Conversational Question Answering dataset.
Contains multi-turn QA conversations with passage retrieval.

.. dm:datasets:: com.github.prdwb.orconvqa ir


QReCC
~~~~~

Question Rewriting in Conversational Context dataset.
Contains conversations with human rewrites of questions.

.. dm:datasets:: com.github.apple.ml-qrecc ir
