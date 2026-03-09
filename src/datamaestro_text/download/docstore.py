import logging
from pathlib import Path
from typing import Callable, Iterator, Optional

from tqdm import tqdm

from datamaestro.download import FolderResource, Resource
from datamaestro.utils import FileChecker


class docstore_builder(FolderResource):
    """Builds an impact-index DocumentStore from a downloaded file.

    Takes a ``FileResource`` as a transient dependency for the source data.
    The framework handles downloading/caching the source file and cleaning
    up intermediate data.
    """

    def __init__(
        self,
        source: Resource,
        iter_factory: Callable[[Path], Iterator[tuple[dict[str, str], bytes]]],
        *,
        keys: list[str],
        checker: Optional[FileChecker] = None,
        block_size: int = 4096,
        zstd_level: int = 3,
        doc_count: Optional[int] = None,
    ):
        """
        :param source: A Resource providing the source data (file or folder)
        :param iter_factory: Given the source file path, yields (keys, content) pairs
        :param keys: Key field names (documents the schema)
        :param checker: Optional hash check on the source file
        :param block_size: Documents per compressed block
        :param zstd_level: Zstandard compression level
        """
        super().__init__()
        self.source = source
        self.iter_factory = iter_factory
        self.keys = keys
        self.checker = checker
        self.block_size = block_size
        self.zstd_level = zstd_level
        self.doc_count = doc_count
        self._dependencies.append(source)

    def _download(self, destination: Path) -> None:
        import impact_index

        destination.mkdir(parents=True, exist_ok=True)

        source_path = self.source.path
        if self.checker:
            self.checker.check(source_path)

        logging.info("Building the document store in %s from %s", destination, source_path)
        builder = impact_index.DocumentStoreBuilder(
            str(destination), self.block_size, self.zstd_level
        )
        for keys, content in tqdm(
            self.iter_factory(source_path), desc="Building document store", total=self.doc_count
        ):
            builder.add(keys, content)
        builder.build()
