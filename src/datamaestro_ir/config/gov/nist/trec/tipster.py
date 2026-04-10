"""

TIPSTER is sometimes also called the Text Research Collection Volume or TREC.

The TIPSTER project was sponsored by the Software and Intelligent Systems Technology
Office of the Advanced Research Projects Agency (ARPA/SISTO) in an effort to significantly
advance the state of the art in effective document detection (information retrieval) and
data extraction from large, real-world data collections.

The detection data is comprised of a test collection built at NIST for the TIPSTER project
and the related TREC project. The TREC project has many other participating information
retrieval research groups, working on the same task as the TIPSTER groups, but meeting
once a year in a workshop to compare results (similar to MUC). The test collection consists
of three CD-ROMs of SGML encoded documents distributed by LDC plus queries and answers
(relevant documents) distributed by NIST.

See also https://trec.nist.gov/data/docs_eng.html and https://trec.nist.gov/data/intro_eng.html
"""

from datamaestro_ir.data.trec import TipsterCollection
from datamaestro.download.links import linkfolder, GlobChecker
from datamaestro.definitions import (
    Dataset,
    dataset,
)
from datamaestro.context import DatafolderPath

# Store meta-information
TIPSTER = dataset(url="https://catalog.ldc.upenn.edu/LDC93T3A")


@TIPSTER
class Ap88(Dataset):
    """Associated Press document collection (1988)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/AP")],
        checker=GlobChecker("AP*", "1daba5cd4e40e757c9c7ad0ed2d694b1"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Ap89(Dataset):
    """Associated Press document collection (1989)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/AP")],
        checker=GlobChecker("AP*", "cef8998b79f4a5f3f1726cd67979cb10"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Ap90(Dataset):
    """Associated Press document collection (1990)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk3/AP")],
        checker=GlobChecker("AP*", "68230c411fa1a16c5b965246645ede78"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Doe1(Dataset):
    """Department of Energy documents"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/DOE")],
        checker=GlobChecker("DOE*", "d6532a8206f5b73bcc69d152a78d78fd"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


# --- Wall Street Journal (1987-92)


@TIPSTER
class Wsj87(Dataset):
    """Wall Street Journal (1987)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/WSJ/1987")],
        checker=GlobChecker("WSJ*", "7ff9870168ef62358bdedca92082f3fa"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Wsj88(Dataset):
    """Wall Street Journal (1988)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/WSJ/1988")],
        checker=GlobChecker("WSJ*", "0d7620d10bce20c4a775211468a1c2d9"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Wsj89(Dataset):
    """Wall Street Journal (1989)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/WSJ/1989")],
        checker=GlobChecker("WSJ*", "dbe2f4a089465cdd1bca58f75202c75e"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Wsj90(Dataset):
    """Wall Street Journal (1990)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/WSJ/1990")],
        checker=GlobChecker("WSJ*", "8fc57e79432ea30149220b437d568d4f"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Wsj91(Dataset):
    """Wall Street Journal (1991)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/WSJ/1991")],
        checker=GlobChecker("WSJ*", "1423d60114b900006aef3f291dbca3e0"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Wsj92(Dataset):
    """Wall Street Journal (1992)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/WSJ/1992")],
        checker=GlobChecker("WSJ*", "46f73cc9289b9285a2278d0f582363c3"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


# --- Federal Register (1988-89)


@TIPSTER
class Fr88(Dataset):
    """Federal Register (1988)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/FR")],
        checker=GlobChecker("FR*", "36e1fab20bcc1471d2d7b9f7c40b4e9f"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Fr89(Dataset):
    """Federal Register (1989)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/FR")],
        checker=GlobChecker("FR*", "d2d898419ef516cc8b63ac6f88da8e77"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Fr94(Dataset):
    """Federal Register (1994)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk4/FR94")],
        checker=GlobChecker("**/*", "014308e9aadd0033baf46c56c91d9505"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


# # ZIFF (1988-92)


@TIPSTER
class Ziff1(Dataset):
    """Information from the Computer Select disks (1989-90)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk1/ZIFF")],
        checker=GlobChecker("ZF*", "7e98e187233487f34e79ec9e6a7b8538"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Ziff2(Dataset):
    """Information from the Computer Select disks (1989-90)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk2/ZIFF")],
        checker=GlobChecker("ZF*", "1f8d026eb2257bcb9b57626fe02e982f"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Ziff3(Dataset):
    """Information from the Computer Select disks (1990-91)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk3/ZIFF")],
        checker=GlobChecker("ZF*", "33824bd7583c325ea0d8e7781b367a1a"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Sjm1(Dataset):
    """San Jose Mercury News (1991)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk3/SJM")],
        checker=GlobChecker("SJM*", "11c0e621d64dc0e8a1e36b71057739d1"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Cr1(Dataset):
    """TODO"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk4/CR")],
        checker=GlobChecker("**/*", "85fb871f13de38bf6b0d36cbfec1d808"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Ft1(Dataset):
    """Financial Times"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk4/FT")],
        checker=GlobChecker("**/*", "807af9f4aa813a9fdf3390870fb37c9a"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class Fbis1(Dataset):
    """Foreign Broadcast Information Service (1996)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk5/FBIS")],
        checker=GlobChecker("FB*", "3079f7065972bd50c06c5b32dbc2028c"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)


@TIPSTER
class La8990(Dataset):
    """Los Angeles Times (1989-90)"""

    DOCUMENTS = linkfolder(
        "documents",
        [DatafolderPath("gov.nist.trec.tipster", "Disk5/LATIMES")],
        checker=GlobChecker("LA*", "e33997ae2353e47a4ce184030add8919"),
    )

    def config(self) -> TipsterCollection:
        return TipsterCollection.C(path=self.DOCUMENTS.path)
