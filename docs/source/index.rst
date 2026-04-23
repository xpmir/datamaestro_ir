Datamaestro IR
==============

**datamaestro-ir** is a `datamaestro <https://github.com/bpiwowar/datamaestro>`_ plugin that provides
access to Information Retrieval datasets for research in:

* **Information Retrieval (IR)** - Document collections, topics, relevance judgments, training triplets
* **Conversational IR** - Query rewriting, conversational search datasets

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   api/index
   datasets/index
   search-datasets


Installation
------------

Install from PyPI:

.. code-block:: bash

   pip install datamaestro-ir

For development:

.. code-block:: bash

   git clone https://github.com/experimaestro/datamaestro_ir.git
   cd datamaestro_ir
   pip install -e ".[dev]"


Quick Start
-----------

List available datasets:

.. code-block:: bash

   # List all datasets in the IR repository
   datamaestro search ir

   # Search for specific datasets
   datamaestro search "msmarco"

Load a dataset in Python:

.. code-block:: python

   from datamaestro import prepare_dataset

   # Load MS MARCO passage dataset
   dataset = prepare_dataset("ir.com.microsoft.msmarco.passage")

Key Concepts
------------

**Data Types**
   Schema classes that define the structure of datasets (e.g., ``Documents``, ``Topics``, ``Adhoc``).
   See the :doc:`api/index` for the complete API reference.

**Dataset Configurations**
   Specific dataset definitions that implement data types with download URLs and processing logic.
   See :doc:`datasets/index` for available datasets.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
