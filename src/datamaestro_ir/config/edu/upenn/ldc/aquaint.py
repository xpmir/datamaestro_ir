"""AQUAINT newswire corpus.

LDC catalog number LDC2002T31 (ISBN 1-58563-240-6). Newswire text in English
drawn from three sources: the Xinhua News Service (PRC), the New York Times
News Service, and the Associated Press Worldstream News Service. Prepared by
the LDC for the AQUAINT Project and used in official NIST benchmark
evaluations.
"""

from datamaestro.context import DatafolderPath
from datamaestro.definitions import Dataset, dataset
from datamaestro.download.links import links, linkfolder, GlobChecker
from datamaestro_ir.data.trec import TipsterCollection


URL = "https://catalog.ldc.upenn.edu/LDC2002T31"


@dataset(url=URL, id=".apw")
class Apw(Dataset):
    """Associated Press (1998-2000)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("edu.upenn.ldc.aquaint", "APW")],
        checker=GlobChecker("*/*_APW_ENG*", "42c4746a12b2436476f62b081887b15d"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path, patterns=["*/*_APW_ENG*"])


@dataset(url=URL, id=".nyt")
class Nyt(Dataset):
    """New York Times (1998-2000)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("edu.upenn.ldc.aquaint", "NYT")],
        checker=GlobChecker("*/*_NYT*", "1edc2eb9c63a431976b453e406cbea71"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path, patterns=["*/*_NYT*"])


@dataset(url=URL, id=".xie")
class Xie(Dataset):
    """Xinhua News Agency newswires (1996-2000)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("edu.upenn.ldc.aquaint", "XIE")],
        checker=GlobChecker("*/*_XIN_ENG*", "2eaf5b3391ed943fe79daa0fead7005a"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path, patterns=["*/*_XIN_ENG*"])


@dataset(url=URL, id="")
class Aquaint(Dataset):
    """Aquaint documents"""

    DOCUMENTS = links("documents", apw=Apw, nyt=Nyt, xie=Xie)

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(
            path=self.DOCUMENTS.path,
            patterns=[
                "*/documents/*/*_APW_ENG*",
                "*/documents/*/*_NYT*",
                "*/documents/*/*_XIN_ENG*",
            ],
        )
