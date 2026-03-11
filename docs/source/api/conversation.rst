Conversation API
================

This module provides data types for conversational information retrieval
and query understanding tasks.

.. currentmodule:: datamaestro_ir.data.conversation.base


Core Data Classes
-----------------

Entry types for conversation turns:

.. autoclass:: AnswerEntry
   :members:

.. autoclass:: RetrievedEntry
   :members:

.. autoclass:: ClarifyingQuestionEntry
   :members:

.. autoclass:: DecontextualizedItem
   :members:

Conversation structures:

.. autoclass:: ConversationHistory
   :members:

.. autoclass:: ConversationHistoryItem
   :members:


Conversational IR
-----------------

.. autoxpmconfig:: datamaestro_ir.data.conversation.base.ConversationUserTopics


Contextual Query Reformulation
------------------------------

Base class for conversation datasets:

.. autoxpmconfig:: datamaestro_ir.data.conversation.base.ConversationDataset


CANARD Dataset
~~~~~~~~~~~~~~

.. autoxpmconfig:: datamaestro_ir.data.conversation.canard.CanardDataset


OrConvQA Dataset
~~~~~~~~~~~~~~~~

.. autoxpmconfig:: datamaestro_ir.data.conversation.orconvqa.OrConvQADataset


QReCC Dataset
~~~~~~~~~~~~~

.. autoxpmconfig:: datamaestro_ir.data.conversation.qrecc.QReCCDataset


iKAT Dataset
~~~~~~~~~~~~

.. autoxpmconfig:: datamaestro_ir.data.conversation.ikat.IkatConversations
