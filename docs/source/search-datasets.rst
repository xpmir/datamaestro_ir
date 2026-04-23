Dataset Search
==============

Search across all datasets indexed in this documentation by id, name,
tag, or task. Click a result to see its details (description,
experimaestro type, variants, external link). Use the link in the
details panel to jump to the full dataset entry.

You can scope a query to a specific field using ``key:value`` syntax:

- ``tag:distillation`` — only datasets tagged ``distillation``
- ``task:"learning to rank"`` — quoted multi-word values
- ``id:lighton`` — match against dataset ids only
- ``name:msmarco`` — match against the human-readable name
- ``description:beir`` — match against the description text

Multiple clauses combine with AND, and free-text tokens are still
searched across all fields. Example: ``tag:ir name:msmarco passage``.

.. dm:search::
