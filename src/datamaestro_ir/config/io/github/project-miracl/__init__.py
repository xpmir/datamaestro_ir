"""MIRACL (Multilingual Information Retrieval Across a Continuum of Languages).

MIRACL is a multilingual ad hoc retrieval benchmark over Wikipedia passages
in 18 languages (Zhang et al., 2022). See https://project-miracl.github.io/

For each language ``<lang>``, registers:

- ``io.github.project-miracl.<lang>.documents``: the passage corpus, as a
  document store built from the ``docs-*.jsonl.gz`` shards
  (`miracl/miracl-corpus <https://huggingface.co/datasets/miracl/miracl-corpus>`_)
- ``io.github.project-miracl.<lang>.<split>.topics``: the queries of a split
  (``train``, ``dev``, ``test-a``, ``test-b``, depending on the language)
- ``io.github.project-miracl.<lang>.<split>.qrels``: relevance judgments
  (``train`` and ``dev`` only, test qrels are not public)
- ``io.github.project-miracl.<lang>.<split>``: the ad hoc task
  (:class:`~datamaestro_ir.data.Adhoc`); ``test-a`` and ``test-b`` are
  :class:`~datamaestro_ir.data.AdhocCollection` (no assessments)

Topics and qrels are fetched from the ungated mirror used by ir_datasets
(`macavaney/miracl-noauth <https://huggingface.co/datasets/macavaney/miracl-noauth>`_).
"""

import gzip
import json
from hashlib import md5
from pathlib import Path
from typing import Iterator, NamedTuple

from datamaestro.download.single import FileDownloader
from datamaestro.stream import Identity
from datamaestro.utils import HashCheck

from datamaestro_ir.data.csv import Topics as TsvTopics
from datamaestro_ir.data.trec import TrecAdhocAssessments
from datamaestro_ir.definitions import DatasetFactory

URL = "https://project-miracl.github.io/"
CORPUS_URL = "https://huggingface.co/datasets/miracl/miracl-corpus"
TOPICS_URL = "https://huggingface.co/datasets/macavaney/miracl-noauth/resolve/main"


class MiraclLanguage(NamedTuple):
    doc_count: int
    """Number of passages in the corpus"""

    shards: list[tuple[str, int]]
    """(MD5, size) of each ``docs-<i>.jsonl.gz`` corpus shard"""

    topics: dict[str, str]
    """split -> MD5 of the topics file"""

    qrels: dict[str, str]
    """split -> MD5 of the qrels file"""


def _read_corpus(paths: list[Path]) -> Iterator[tuple[dict[str, str], bytes]]:
    """Reads MIRACL JSONL shards (docid, title, text)."""
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as fp:
            for line in fp:
                data = json.loads(line)
                content = data["title"] + "\0" + data["text"]
                yield {"id": data["docid"]}, content.encode("utf-8")


def _register_all():
    datasets = DatasetFactory(
        __name__, prefix="io.github.project-miracl", class_prefix="Miracl"
    )
    for lang, info in MIRACL_DATA.items():
        shards = {
            f"CORPUS_{i}": FileDownloader(
                f"docs-{i}.jsonl.gz",
                f"{CORPUS_URL}/resolve/main/miracl-corpus-v1.0-{lang}/docs-{i}.jsonl.gz",
                size=size,
                transforms=Identity(),
                checker=HashCheck(md5sum, md5),
                transient=True,
            )
            for i, (md5sum, size) in enumerate(info.shards)
        }
        documents = datasets.docstore(
            f"{lang}.documents",
            shards,
            iter_factory=_read_corpus,
            doc_count=info.doc_count,
            url=CORPUS_URL,
            languages=[lang],
        )

        base_url = f"{TOPICS_URL}/miracl-v1.0-{lang}"
        for split, topics_md5 in info.topics.items():
            topics = datasets.file(
                f"{lang}.{split}.topics",
                TsvTopics,
                url=URL,
                TOPICS=FileDownloader(
                    "topics.tsv",
                    f"{base_url}/topics/topics.miracl-v1.0-{lang}-{split}.tsv",
                    checker=HashCheck(topics_md5, md5),
                ),
            )
            qrels = None
            if split in info.qrels:
                qrels = datasets.file(
                    f"{lang}.{split}.qrels",
                    TrecAdhocAssessments,
                    url=URL,
                    QRELS=FileDownloader(
                        "qrels.tsv",
                        f"{base_url}/qrels/qrels.miracl-v1.0-{lang}-{split}.tsv",
                        checker=HashCheck(info.qrels[split], md5),
                    ),
                )
            datasets.adhoc(f"{lang}.{split}", documents, topics, qrels, url=URL)


# Checksums, sizes and document counts (from ir_datasets)
MIRACL_DATA = {
    "ar": MiraclLanguage(
        doc_count=2061414,
        shards=[
            ("d484e02a3cd973b4c36bd7867d7f40fa", 94104175),
            ("2af6f236d7c37c2bde97be4a13a3abaa", 83793880),
            ("f5ee426bd15eb07e335c0b2f51798672", 70295610),
            ("3808e89c2468bc10f49d4d4ae5f8a66c", 64551259),
            ("73a29b3038c984a86da189aa16c3fb8f", 7227421),
        ],
        topics={
            "dev": "4ef84df620f1b5521ce8e7ebb438f344",
            "test-a": "8b2a7f0ca268ff4c15dd7efd5d8f59cd",
            "test-b": "3bbc0e8ed604b12b4591d1eb6be574bc",
            "train": "38c6afa5a27b8c3565205bbc23756d8f",
        },
        qrels={
            "dev": "ac6496c6f8efabd33936b144557548b0",
            "train": "f85eed133bec7e50336bbfe0cbbb01c0",
        },
    ),
    "bn": MiraclLanguage(
        doc_count=297265,
        shards=[
            ("1e30b159bc3634e068bcf8f1fbff68ef", 59713182),
        ],
        topics={
            "dev": "17ff030d33d678547626c66fb9f52b4c",
            "test-a": "1c94d4bf5b07fa8eda2b8b5f1b2ae42c",
            "test-b": "1cb9838ec9d90fdef68ff0fc5d489cd2",
            "train": "0a48f9dff4565980c6c70f60403c2541",
        },
        qrels={
            "dev": "42158a39e400a5a6ebef97b9e91e54b1",
            "train": "d59c2bedbae4a4c2c8290539c66a93ea",
        },
    ),
    "de": MiraclLanguage(
        doc_count=15866222,
        shards=[
            ("ca7ab6bb0328f2e17a634e2a3445f1ea", 97640576),
            ("68e5d3add0fb289bbc354d93471850be", 94992507),
            ("fc427890706a5d31ff2e8d559fba5552", 91425088),
            ("990dbdef008fe2901cea858524df8bab", 85007239),
            ("6c0ab305737fd9b3e0ab31d9b66aebbe", 85433314),
            ("ade6161113825b462355ebec32ab57bc", 85494784),
            ("4347f0cad367548e6776465c316d4571", 85048361),
            ("aad7fb0916949387bb1a158199d0c11d", 85544324),
            ("f820f563a1aa2aa0e1841342336553a9", 82876799),
            ("243c95ee4c7b22a058e08d7eb1027b47", 79775850),
            ("527fdba98da58526ca3f30b7a1633939", 81937816),
            ("50dedc25b39f4a9ebc1ad8e94a2f85e7", 82225480),
            ("ed1cbe22bd2bcb419a5999d8140c14cf", 82302655),
            ("2929a7c9b8a280958a388a990a14e581", 81286223),
            ("e8ab4bbd1b4a34ed7c90a0326a3d04f6", 81285393),
            ("19e90cfff174c6639b657e1f90a3609c", 81734841),
            ("75017466b064eaaed9ce11cd629a2766", 81453194),
            ("10f71f7df716a10d678bbd5bbd640b77", 81111867),
            ("2e4ccc640c95be49dd4c9118f5e7fe25", 79961121),
            ("8ea6c268ff9d6aca89b2373fbe3490b5", 80334928),
            ("2cb88bfd5e644b26aec39a59e7efd79e", 80594842),
            ("7d63fb93948260af3409e41f673f8c77", 81303770),
            ("c0db07f9c2e01282909030105797dddf", 82621593),
            ("2d710c2e4a9b1eff7a0012580ef25540", 81564108),
            ("397eba2f7fd3b310111a45f0e37aea42", 80246927),
            ("e62f5d0c7640c6b51974466b7a9bde42", 79283563),
            ("bdebcadf33dc1bb7c2e88237ace54c8a", 78943370),
            ("e56b7656cef7356ba696d77261464cf6", 80159901),
            ("e1ffce8721a7c336e2925d43100b6e78", 78672926),
            ("570dc8906b883c8566e116ac0d3c8ddb", 80702668),
            ("667aa167000a65bf6f68d82058dd803d", 80576937),
            ("d733bea534c2c2bd9e6c125fc14d97d1", 57737892),
        ],
        topics={
            "dev": "ad5cd6dc4e1fa51d6383fec385cf2854",
            "test-b": "3ef6e6e6d702714d40ad8b78ece98d7a",
        },
        qrels={
            "dev": "39d96bd97b691783fa07ff523955c8ad",
        },
    ),
    "en": MiraclLanguage(
        doc_count=32893221,
        shards=[
            ("ed94fdd2dad913fcffe0308b021082d6", 98856369),
            ("9adcfccb058f46050648dea35755a39f", 97338482),
            ("f961ce2675bc472b4d4126399acb7564", 64571457),
            ("853a5180f4c1012ebee1f2c5a3e26506", 79988237),
            ("80271ce096ccb01565857d4dd8f2b4a5", 91846360),
            ("2ae2b6919cbea03f03469633690acfd1", 91542927),
            ("7543b1726742ce466cd2a2a57f765e62", 88919126),
            ("c9ac416afef48a6f7d9dcc66ddeff2d5", 88149943),
            ("96f9fd6a228e9f46bfdb4eda0ffd6b90", 87557556),
            ("6d3d37aac292937e93dbe0eb0efb1c00", 86565246),
            ("20e5926a36d44c6ece5c1942bda19817", 85787457),
            ("7552668e7db1591d0fe71df3b044cffd", 85858461),
            ("383760be692a540b30e5b1b2239bab16", 85089158),
            ("7da0ffd49848f9c7e2912b1ad1152d3b", 82769410),
            ("2c4d6f9220716e624daa016756d707e8", 83032778),
            ("a90950760f60e5f0e92c4e7f3774f7e7", 82548385),
            ("20e2030038fbb450be2ac09fedaf77b3", 81947138),
            ("12136a0df0a80092eafd694820cc6f81", 81038479),
            ("a907bbf4ef1b264d0304bb99cc392b15", 80496066),
            ("f52ad0a000336d7a73a4e0d0e2bb996f", 80243337),
            ("8b9d4eb4374aeba9567e09470cf5b061", 80003557),
            ("32ab1841c2deb612eb3ec83a2f371452", 79751834),
            ("5334d398deaa053e6f380612973580ef", 76766869),
            ("505f419951b8665cd9fc69604f6b5e62", 75810312),
            ("eb579b4ad40d6735d370bf30b7877a25", 76272005),
            ("7de75404b5cf6c8b965c8257e62ba70b", 77082664),
            ("d42ed88eaa947b610846188185a25f54", 78228256),
            ("e61b3098053a98b9bdb753a34476e239", 76445022),
            ("de185edb91a43a6dbb434658fbdbb96f", 76434972),
            ("8f25bd36199c36e2a1812b2e22b56430", 74697707),
            ("9c2e84252fb7ca24b4fd593682bd2aef", 71839143),
            ("4fc0fb743bc402bb9d0469856fd3694c", 73437320),
            ("64c3d8cae61441df7201d44b08972e41", 73322325),
            ("42ebd5e87d5a417689c37cf336749e7f", 70699147),
            ("91635ee421cfcd9b8ea5834695e556b7", 74843362),
            ("fa7ada6f5f145f5a04bd4113cfe09787", 71914591),
            ("90c6abea75caa84381d238fd85e4520e", 73255731),
            ("6fdf53729c158371ea30cf56f37d7db5", 74956992),
            ("8dcb112f443c49e1d35bdac0db63a111", 70506538),
            ("071adb3f777cb077ff170c6856bbdcd7", 73824158),
            ("879d54b7d7826a2d7133955a689890f5", 72684614),
            ("f379d5441be8e19a690f116a7a9ad501", 74674474),
            ("1fb0a973751261adc3c2165fc99f0ce1", 74166535),
            ("e1621cf06f3e5d6b3eff4d8607641258", 72571698),
            ("47b0191ce682e60708a979b2232607f9", 72937865),
            ("3fa31818300aba29db933557f31b3494", 74948873),
            ("462641fe08c052ab1fbf7add67a48859", 75950579),
            ("5cb8207d31dab7cd82ba64317e65c8ac", 72978719),
            ("6a8433dcf294c669078cdf1e5a88aa38", 73930541),
            ("75975d9495b9cc4aa231f873281c5f02", 74004419),
            ("60dbe4823f30923e1ebe3ef5b8f0879d", 70736448),
            ("ec9c101022dfc897c058e96c220758f6", 73415840),
            ("925b65b5ef08abb41ce09d8645bd6dd2", 73776910),
            ("62385f5b6cf87e7aa039cdc542a9e55c", 72091328),
            ("58f025c4ef8d3868e5eabef9d39395dd", 72782654),
            ("5f3f8787af6baf5a85ea3cf06698af38", 70569978),
            ("eee9cccdcd05cc89ef8e5e2fa9fff7ac", 71740868),
            ("4213cb413ca6079920f0b0241bad7b56", 72217620),
            ("38de32e0b8a9723dbcaa83d23e0e8d6d", 71862927),
            ("0a6a80e46a434776ec2ff67bc6a28e6e", 71495670),
            ("a583b681b03b9637a2b80fb6e4e35afa", 69897224),
            ("3df26a61e7ecb4c2244376041d7b4e1c", 70170718),
            ("739d42eb5157373333a3c978eec150b1", 69279270),
            ("0b25fcfc2a77bd551cb32d763a83b49d", 68754531),
            ("2391aec604153b8e4b78cd6ae83decd4", 65487314),
            ("9122a610d9b816975e93b19ae0121f46", 52897612),
        ],
        topics={
            "dev": "cd1442ed7b711ea5ff246ea2be8ce0c7",
            "test-a": "3d834410789bbf4568e74942344d0831",
            "test-b": "5f8ef71d59a420b2e46afb70e4128464",
            "train": "a8a1ae3254c07c46e37abd6bbfab63ad",
        },
        qrels={
            "dev": "ce9a8e2345de41210895f65866c4c2e7",
            "train": "556f857f7f67c942f683c6a7b4e29b48",
        },
    ),
    "es": MiraclLanguage(
        doc_count=10373953,
        shards=[
            ("f28bff7c3058d2cb8c9e99b919de6c4f", 92874838),
            ("1c08c92860d8fa535b545eb530b27dbd", 89646018),
            ("fb75e0db47df90f0135df0acba637196", 85432948),
            ("c8ae25908e11c995e3d2a822c6dd8ffc", 83318741),
            ("671420453f20ef32a35cbf2c9f9c1310", 80843535),
            ("b907c427cf796810f28ee9375f91bcdf", 79438738),
            ("392e4b93c786551799fab1c2f5f161d8", 79152845),
            ("f8cf2f4c34b28ca166143f5b94349500", 78544890),
            ("d7f1edf8f756009db26deb49bf22f7ce", 77588034),
            ("456ec112182b917d91d9322cf00786d2", 66337959),
            ("c71af44961b7e0fd33c49fe1f61039bf", 62622594),
            ("0bd1d3ab02269c7ee2207ff39fe544e5", 65707558),
            ("e94562d6f2df0a1c84075c58cb035883", 72601971),
            ("19509077dc27fc650f134223b385a5e7", 75474457),
            ("deb94c634d4c1d85e78bb2ff35028383", 76497338),
            ("02a780bf6a3a31bfdfc759ec403403bc", 73009124),
            ("748934c50ae05d7273a2c470b40d6e89", 74206192),
            ("62ff853bfc22d96f4ecae8e8e1c86caa", 67140050),
            ("81e79b7ee32173601e71fd9173bf62b3", 77883748),
            ("fa6bb29881142b6df1a1049aafa864de", 79904116),
            ("b561242ee7bc55e24cd21cd1a48a256c", 57162245),
        ],
        topics={
            "dev": "8f057b82c8bb6ba44776cadacee63e7b",
            "test-b": "197e982a1bbac021b0f40cf7b5e958a7",
            "train": "efe6c440986a96ebd3edeecb91c80492",
        },
        qrels={
            "dev": "3a25e2467698ff529b7f97e76b8eab6b",
            "train": "5913f00b8ae436c198d563e8fbaefdb1",
        },
    ),
    "fa": MiraclLanguage(
        doc_count=2207172,
        shards=[
            ("6b1135045aaa46c539a2960cda7c3b0a", 80831146),
            ("a86e85011229e984dc44728baf46eebb", 56314013),
            ("88ac988b8b61287bd55f9b977532ac5e", 53167889),
            ("2e0030f301e5af7723dff4dce14f5343", 57027208),
            ("dc78362be19963c12fd511c07d1386b3", 22422574),
        ],
        topics={
            "dev": "c64fb62af718c135a1843722ea8fdb8d",
            "test-b": "3114048d32c21d7e86f36a81ec86d8b0",
            "train": "8843271f035a12a43983d7795b0e144c",
        },
        qrels={
            "dev": "a1384b550860cea3eadb9bd8407f36ad",
            "train": "b1cd41c3c72473f372f2d0c0c2f4395c",
        },
    ),
    "fi": MiraclLanguage(
        doc_count=1883509,
        shards=[
            ("1c7d419a877fd406db8321e56600e6c5", 79712094),
            ("f45db81107c579315d0e5a9d2f33a57f", 71207705),
            ("f6d202472a36d443f5e214a6de92b515", 68005963),
            ("1ea7920d58d75a309462585f3fa479cf", 51120372),
        ],
        topics={
            "dev": "e328e92f38c13f6e5a5b6903670e01fc",
            "test-a": "33480316a834ec45d935a65663f148ea",
            "test-b": "098a89152d59a59a98824bb4ed63c2e0",
            "train": "b1c1f3bceb7c52b2e44b587fb0690454",
        },
        qrels={
            "dev": "c0bdc01bdbd4753ee6b4f2e9dee08ceb",
            "train": "348b8e5b45cbd2ef4074ef9cb6ff9076",
        },
    ),
    "fr": MiraclLanguage(
        doc_count=14636953,
        shards=[
            ("fa62c438d74f8ac9501db7af6aaca817", 81033537),
            ("34262e282c58c31e965d89274df96e40", 78469930),
            ("a051dc99b9ffb81c1eaf394627064927", 70017319),
            ("f8a18f6ae5b2ebc58a39791469c38bdd", 66942793),
            ("5a240afc021ecb5dc27285abe4b6d923", 59460892),
            ("a03ba537c9d68ecef50aa5a4777407cc", 62629125),
            ("499367e79f29e565cf9bfcbe3ab2adea", 69205967),
            ("652114ff90f8cf484c8701e5f8ea5d02", 68194158),
            ("0d7e5b3231ab67d6026f8237bcf0be0f", 68757697),
            ("f916ce6802d20b4c3f9905363d3bb31f", 68056019),
            ("8816919a5cfb523ac75579453dde6d47", 67698213),
            ("642128fb3b08f35d7c8d34f6994d98cc", 67898616),
            ("daa9926bdf888aed05dbba8aafa419dd", 66768800),
            ("12afc85cd7ddda45c3f5d3f199b596a7", 65451686),
            ("46a9acd1b09a81701b89e8b1859bf30c", 65202296),
            ("3968784422a38c29653c1636938442de", 62889747),
            ("a4b18549cb5bed8365442e5a434c18c0", 62186443),
            ("245749a12a976b0a11bac0a8fcc61b92", 61659176),
            ("2f81457bb7058c4708e15d73858dfc86", 61297567),
            ("96309f23dc95423c831a2ccf30b97ec3", 62810856),
            ("db4a22111c1b3dbbbbac2d71531dd631", 62973327),
            ("f72f7bb7c14ef8a450715f8a952b2816", 61427017),
            ("666b5e7ddcc0918febb3331a14b3d612", 58091930),
            ("f9c3316e6407e17206a1eedacc6b776b", 61344070),
            ("d32c6a0cf9b59bba24774e4fd4f0691e", 61250359),
            ("a110c2681991390092e2e4920e28abf7", 56044622),
            ("2eb3482ab52c21ebf16c22184a4816c4", 64189087),
            ("d2c4f0443b2ccd875b45b5519fa6188c", 65662563),
            ("5f66fb25b1d295614b8a92c4f9636d05", 65310822),
            ("1f4c400a40b7020c797f105457e451d2", 18176293),
        ],
        topics={
            "dev": "d8b97c45c2256480c0b0c56634deb828",
            "test-b": "3988760b1e1355a92a87094b5ad0f51b",
            "train": "962aebe813c808a9e6b4cf7ac6764758",
        },
        qrels={
            "dev": "aa458c4c603c919856f2d38adb474c25",
            "train": "9e0796915fe5bb9ecb144439ebf2e2c0",
        },
    ),
    "hi": MiraclLanguage(
        doc_count=506264,
        shards=[
            ("ffefe282e6be09f06a69d2945f402bda", 95650487),
            ("3fe3f9f265b2f52a25b4a49b80c82cf5", 1161610),
        ],
        topics={
            "dev": "69aafc100baf5617d7d7e63b47c4d86a",
            "test-b": "1c1aef94f1206e4b98aa519b3e7c9dd8",
            "train": "fde7fc047da3a9947efb711fc45b79a4",
        },
        qrels={
            "dev": "0b97e26e09bdf24a21325b74016d3047",
            "train": "a4f6627539655cdb3f61ca575552354d",
        },
    ),
    "id": MiraclLanguage(
        doc_count=1446315,
        shards=[
            ("e2ff61190833252d531dea1f90d90426", 68482339),
            ("ffe27c80bfcb8061954d64282cb0f71c", 39714366),
            ("c85ce23396c8c0e61dc11806c32495e2", 61386913),
        ],
        topics={
            "dev": "1ecff4a071699fb1c1d16a342a7083ff",
            "test-a": "7dbd725b4e83a363f630bb8d0c3e775b",
            "test-b": "b51cb52b5312b99cccb16206a56ac29f",
            "train": "2051ab2d790d9e10b450d193dc1878e8",
        },
        qrels={
            "dev": "db025a23bda716dce6e37b42229b436f",
            "train": "7aeb868e30db507ec05c21a3365c7290",
        },
    ),
    "ja": MiraclLanguage(
        doc_count=6953614,
        shards=[
            ("f064d7aaea82b1619d1709c36483c461", 87302446),
            ("903c2bc115f8ef719b3d0e21c2b13daf", 84310580),
            ("57245e8a46d1cf289bc5377fd0d54918", 82263425),
            ("9771ca5370e49dbcae101650293e0cbc", 79819291),
            ("eb4abc18ea71876d8a974813701e52d2", 76369546),
            ("aeceab57d0e29f028cdad7724eefae2b", 73543156),
            ("f2e9f77af251bff8b8dbac3f0972abd9", 75527173),
            ("81e4e762a82d13fefd8d453846f286ac", 75265643),
            ("7f008e134c6acbf5ea2aeee2f2b16157", 74070028),
            ("c354f0d766ff28950b2aea0227141b5c", 72893709),
            ("651d84515e3f23a7dfebbb9dd1a97e05", 74375206),
            ("88f8bb6f3a702f0228523a02bc913e55", 73261365),
            ("2e5156a5f0580f3b2aae763a55d323f3", 71472211),
            ("3a080cb1f0e82843903c300ea3330eef", 64646621),
        ],
        topics={
            "dev": "93207778d1032e1d0908e4a3e0326e5f",
            "test-a": "de78e8e351f23488b01ddc52e756c35f",
            "test-b": "61a655f2d14ce4d0b661a11f09a5ec57",
            "train": "e3207521af4663e5d5947efe73cd3c2a",
        },
        qrels={
            "dev": "1ea89d53b04d60aecccf01d90d33b00d",
            "train": "05b2cca3cb634ada628a131bb036943d",
        },
    ),
    "ko": MiraclLanguage(
        doc_count=1486752,
        shards=[
            ("dcae1fe2c7c966f4bd92c09aa9fba88f", 87965596),
            ("8068bb2ca5796bbca7af267527518d86", 75422723),
            ("ea72e30dd4ed336b37422f559f60d4ed", 62582229),
        ],
        topics={
            "dev": "61061f60143677b7dd9eac8cd963ac15",
            "test-a": "b23aa4792976fedf3bd45cbb55d28f6d",
            "test-b": "35a60095be8f6d05e106eae2a13253fa",
            "train": "32a7b5a373761ec50b689a8d2fb2439a",
        },
        qrels={
            "dev": "c0935fc8df5e10d578a37b4ad4d1b8ac",
            "train": "2ef507d04d4c730ccb18908b8cc8e1be",
        },
    ),
    "ru": MiraclLanguage(
        doc_count=9543918,
        shards=[
            ("4373a43259b241792bcb6e979943db29", 100436081),
            ("5283c42d6542e5c8dc2c1be5789f0b7a", 98036311),
            ("e2a7250e04a82f5f257b48d3950410c7", 90050391),
            ("e0c6f72fbbb5ebb418387b9c9a22696f", 89443294),
            ("0b351f8491f1e8a68db00bd56de41862", 82032411),
            ("fe82dd5ec8fede641829e8acb71da04a", 88457515),
            ("081133522ff711ed40ae47c8fd3c56e2", 81102964),
            ("9c1297361dce8919649a09f7c72d49bb", 80229752),
            ("e488d2d37394edf07cee425dc758c11a", 74478227),
            ("51b4da27835ca5febda7848d6754ecac", 73515957),
            ("57b0d428b8d0004387eaa507e2336d79", 75656001),
            ("da7c457c057cb525ce57980a7dde0989", 78271034),
            ("ae23ce2efd5216008d5bd4d353b844ac", 76539873),
            ("6edf74f3a33c6de035881ad70bb98cb8", 78171375),
            ("ec732d5a4a36a0cee67f631f9b7ca976", 78271662),
            ("117e3c9b6c2538144717a22319c8c701", 79432861),
            ("cae1d225074503572b377254d74f90c3", 81415343),
            ("e7468fe6203f581175c2527bdc6eb5ed", 81955644),
            ("c6caaa99bf8c2746a88d5f2d0dae6a49", 81041807),
            ("4ddfc2b723b470597caa1a8543d5ee9f", 6749385),
        ],
        topics={
            "dev": "8a86550b1a041d7e512884c9683a7867",
            "test-a": "cc3e7ca3cb78b3e298b549adaf689317",
            "test-b": "a3c7b2659aba4777eed8f53df588f28b",
            "train": "ad6a5a0bf13295857602c110ba6b04c9",
        },
        qrels={
            "dev": "748511ee0442e09e836a49deb1e2e8b3",
            "train": "b6b30a6cacab487bf41d81d93f01faa0",
        },
    ),
    "sw": MiraclLanguage(
        doc_count=131924,
        shards=[
            ("afdb146539d0f488f14c2833e14df799", 10199394),
        ],
        topics={
            "dev": "5ffbaf70e2a5868780ee6fc09fe08944",
            "test-a": "7e21f860b85d3660d415f1dd5f29b2bf",
            "test-b": "002e79f781b4eb89ced2929a3e4693f9",
            "train": "6a0497e1568b7f8518ee4f5205167d89",
        },
        qrels={
            "dev": "2ca78e665778ed6be4f9b01e0b6e80c0",
            "train": "e582f9074db31dea26fe43dce87d0ff5",
        },
    ),
    "te": MiraclLanguage(
        doc_count=518079,
        shards=[
            ("7b1067004a12a018eb4ca9e7e7655600", 68858140),
            ("703a0ac0d66c6fa6b684df9d7d6afd20", 3454223),
        ],
        topics={
            "dev": "6382282438291981c2b65ae3c8f3dd71",
            "test-a": "413a15ae8e5dbe9bd55a1d3c722f83e6",
            "test-b": "bdc6dc6bd94405d48dc7ff96b32eec0a",
            "train": "43de28312b4b7fe39eb5c0a7e72a3883",
        },
        qrels={
            "dev": "2f7c45f44bc8842633a8eb14dbb99f19",
            "train": "b55f03829da637e3e186780292e0b305",
        },
    ),
    "th": MiraclLanguage(
        doc_count=542166,
        shards=[
            ("060bf4cb7fa0127a571023709f117554", 101610412),
            ("2e05644bee9ab5da8f7f7177f0c7bafc", 8027002),
        ],
        topics={
            "dev": "142e213631cd7ee91cb5b9e384a14a6d",
            "test-a": "f4786ff257fcffca790112a17696c261",
            "test-b": "61a692a875d8b39824a591d864cc09ed",
            "train": "2e188722d9f0014be8edcaa2f6ef4581",
        },
        qrels={
            "dev": "89c8aa2ebb027d3f3c1fd984aeb58ee9",
            "train": "0745678058c20c179c6b2dc0206d88ca",
        },
    ),
    "yo": MiraclLanguage(
        doc_count=49043,
        shards=[
            ("07bab606c31660be729353419ff6537d", 10946),
        ],
        topics={
            "dev": "7820b5fddac56306dfef4863271d3f5c",
            "test-b": "90123239f6b66592385b0e1386a03186",
        },
        qrels={
            "dev": "523aeb9fb3c631dee98e267760a46735",
        },
    ),
    "zh": MiraclLanguage(
        doc_count=4934368,
        shards=[
            ("68248e7866b0571e750140a4baf31ad8", 97061486),
            ("c3bc8bd34a53d9589c29f580b94c2d17", 85538355),
            ("eefbf18d0e8edcee9db00a748ce08f86", 80287705),
            ("cc7924a7bacc7abd0974fd07168064d1", 76098139),
            ("18879a45e3b27f0b0b588e68a697e282", 60594024),
            ("be7e9f9b61c23306e6feedb2e26aab20", 50492481),
            ("6a6359359436b4f89c5be50d856e625d", 76002940),
            ("06aa37a25a84c6830ae5fb57ff877b05", 71709349),
            ("3ec702e19ef98ed6c9f639c936e293cd", 72069621),
            ("841c4f70b603aee993451b3b75ab5236", 63401726),
        ],
        topics={
            "dev": "43d975e76eec94edac3f325139fe04d1",
            "test-b": "4cb3cda9f62d284476bd212ad6b4038f",
            "train": "421e6122338cb80d43fcaa7ecacc1611",
        },
        qrels={
            "dev": "5906a4dc2373cf524167487660d7b6f1",
            "train": "2c3b64e46276e1df97a15c65ab5b90f1",
        },
    ),
}


_register_all()
