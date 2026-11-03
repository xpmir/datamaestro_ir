"""IR-specific dataset factory.

:class:`DatasetFactory` extends :class:`datamaestro.factory.DatasetFactory`
(which provides ``register`` and ``file``) with builders for document stores
and ad hoc tasks::

    datasets = DatasetFactory(__name__, prefix="org.example")

    docs = datasets.docstore(
        "x.documents",
        FileDownloader("corpus.parquet", url),
        iter_factory=beir_parquet_docstore_iter,
    )
    topics = datasets.file(
        "x.queries", BeirParquetTopics, QUERIES=FileDownloader("q.parquet", url)
    )
    qrels = datasets.file(
        "x.qrels", BeirParquetAssessments, QRELS=FileDownloader("r.parquet", url)
    )
    datasets.adhoc("x", docs, topics, qrels)
"""

from typing import Callable, Optional, Sequence, Union

from datamaestro.download import Resource, reference
from datamaestro.factory import DatasetClass
from datamaestro.factory import DatasetFactory as BaseDatasetFactory

from datamaestro_ir.data import Adhoc, AdhocCollection
from datamaestro_ir.data.beir import BeirDocumentStore
from datamaestro_ir.download.docstore import docstore_builder


class DatasetFactory(BaseDatasetFactory):
    """Creates and registers IR datasets in a module (see
    :class:`datamaestro.factory.DatasetFactory`)"""

    def docstore(
        self,
        ds_id: str,
        source: Union[Resource, dict[str, Resource]],
        iter_factory: Callable,
        *,
        keys: Sequence[str] = ("id",),
        doc_count: Optional[int] = None,
        name: Optional[str] = None,
        doc: Optional[str] = None,
        url: Optional[str] = None,
        languages: Optional[Sequence[str]] = None,
    ) -> DatasetClass:
        """Registers a document store (:class:`BeirDocumentStore`) built from
        one or several source files

        :param source: The source resource (named ``CORPUS``), or a dictionary
            of resources (attribute name -> resource) when the corpus is
            split into several files; in that case, ``iter_factory`` receives
            the list of paths
        :param iter_factory: Iterates over (keys, content) pairs from the
            source path(s)
        :param languages: Languages of the documents (ISO 639-1 codes)
        """
        sources = source if isinstance(source, dict) else {"CORPUS": source}
        store = docstore_builder(
            list(sources.values()) if isinstance(source, dict) else source,
            iter_factory=iter_factory,
            keys=list(keys),
            doc_count=doc_count,
        )

        def config(self) -> BeirDocumentStore:
            return BeirDocumentStore.C(
                id=self.__dataset__.id,
                path=self.STORE.path,
                languages=list(languages) if languages else None,
            )

        return self.register(
            ds_id,
            BeirDocumentStore,
            config,
            resources={**sources, "STORE": store},
            name=name,
            doc=doc,
            url=url,
        )

    def adhoc(
        self,
        ds_id: str,
        documents: DatasetClass,
        topics: DatasetClass,
        assessments: Optional[DatasetClass] = None,
        *,
        name: Optional[str] = None,
        doc: Optional[str] = None,
        url: Optional[str] = None,
    ) -> DatasetClass:
        """Registers an ad hoc task referencing other datasets

        This is an :class:`~datamaestro_ir.data.Adhoc` dataset, or an
        :class:`~datamaestro_ir.data.AdhocCollection` if there are no
        assessments (e.g. test sets with hidden relevance judgments).
        """
        resources = {"DOCS": reference(documents), "QUERIES": reference(topics)}

        if assessments is None:

            def config(self) -> AdhocCollection:
                return AdhocCollection.C(
                    id=self.__dataset__.id,
                    documents=self.DOCS.config(),
                    topics=self.QUERIES.config(),
                )

            base = AdhocCollection
        else:
            resources["QRELS"] = reference(assessments)

            def config(self) -> Adhoc:
                return Adhoc.C(
                    id=self.__dataset__.id,
                    documents=self.DOCS.config(),
                    topics=self.QUERIES.config(),
                    assessments=self.QRELS.config(),
                )

            base = Adhoc

        return self.register(
            ds_id, base, config, resources=resources, name=name, doc=doc, url=url
        )
