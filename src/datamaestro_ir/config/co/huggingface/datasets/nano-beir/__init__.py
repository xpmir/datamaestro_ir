from datamaestro.definitions import dataset, Dataset, datatasks
from datamaestro.download.single import FileDownloader
from datamaestro.download import reference
from datamaestro_ir.data.beir import (
    BeirDocumentStore,
    BeirParquetTopics,
    BeirParquetAssessments,
    beir_parquet_docstore_iter,
)
from datamaestro_ir.download.docstore import docstore_builder
from datamaestro_ir.data import Adhoc

NANO_BEIR_DATA = {
    "arguana": "NanoArguAna",
    "climate-fever": "NanoClimateFEVER",
    "dbpedia-entity": "NanoDBPedia",
    "fever": "NanoFEVER",
    "fiqa": "NanoFiQA2018",
    "hotpotqa": "NanoHotpotQA",
    "msmarco": "NanoMSMARCO",
    "nfcorpus": "NanoNFCorpus",
    "nq": "NanoNQ",
    "quora": "NanoQuoraRetrieval",
    "scidocs": "NanoSCIDOCS",
    "scifact": "NanoSciFact",
    "webis-touche2020": "NanoTouche2020",
}


def register_nano_beir_subsets():
    for name, hf_name in NANO_BEIR_DATA.items():
        base_url = (
            f"https://huggingface.co/datasets/zeta-alpha-ai/{hf_name}/resolve/main"
        )

        # --- Documents
        def make_docs_class(
            n=name, url=f"{base_url}/corpus/train-00000-of-00001.parquet"
        ):
            class Docs(Dataset):
                CORPUS = FileDownloader(
                    "corpus.parquet",
                    url,
                )
                STORE = docstore_builder(
                    source=CORPUS,
                    iter_factory=beir_parquet_docstore_iter,
                    keys=["id"],
                )

                def config(self) -> BeirDocumentStore:
                    return BeirDocumentStore.C(
                        id=self.__dataset__.id, path=self.STORE.path
                    )

            Docs.__name__ = f"NanoBeir_{n.replace('-', '_')}_Documents"
            Docs.__module__ = __name__
            return dataset(id=f"co.huggingface.nano-beir.{n}.documents")(Docs)

        Docs = make_docs_class()
        globals()[Docs.__name__] = Docs

        # --- Queries
        def make_queries_class(
            n=name, url=f"{base_url}/queries/train-00000-of-00001.parquet"
        ):
            class Queries(Dataset):
                QUERIES = FileDownloader(
                    "queries.parquet",
                    url,
                )

                def config(self) -> BeirParquetTopics:
                    return BeirParquetTopics.C(
                        id=self.__dataset__.id, path=self.QUERIES.path
                    )

            Queries.__name__ = f"NanoBeir_{n.replace('-', '_')}_Queries"
            Queries.__module__ = __name__
            return dataset(id=f"co.huggingface.nano-beir.{n}.queries")(Queries)

        Queries = make_queries_class()
        globals()[Queries.__name__] = Queries

        # --- Qrels
        def make_qrels_class(
            n=name, url=f"{base_url}/qrels/train-00000-of-00001.parquet"
        ):
            class Qrels(Dataset):
                QRELS = FileDownloader(
                    "qrels.parquet",
                    url,
                )

                def config(self) -> BeirParquetAssessments:
                    return BeirParquetAssessments.C(
                        id=self.__dataset__.id, path=self.QRELS.path
                    )

            Qrels.__name__ = f"NanoBeir_{n.replace('-', '_')}_Qrels"
            Qrels.__module__ = __name__
            return dataset(id=f"co.huggingface.nano-beir.{n}.qrels")(Qrels)

        Qrels = make_qrels_class()
        globals()[Qrels.__name__] = Qrels

        # --- Full Adhoc
        def make_full_class(n=name, d=Docs, q=Queries, r=Qrels):
            class Full(Dataset):
                DOCS = reference(d)
                QUERIES = reference(q)
                QRELS = reference(r)

                def config(self) -> Adhoc:
                    return Adhoc.C(
                        id=self.__dataset__.id,
                        documents=self.DOCS.config(),
                        topics=self.QUERIES.config(),
                        assessments=self.QRELS.config(),
                    )

            Full.__name__ = f"NanoBeir_{n.replace('-', '_')}"
            Full.__module__ = __name__
            Full = dataset(id=f"co.huggingface.nano-beir.{n}")(Full)
            return datatasks("information retrieval")(Full)

        Full = make_full_class()
        globals()[Full.__name__] = Full


register_nano_beir_subsets()
