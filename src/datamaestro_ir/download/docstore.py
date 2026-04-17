import json
import logging
import urllib.request
from pathlib import Path
from typing import Any, Callable, IO, Iterator, Optional, Tuple

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

        logging.info(
            "Building the document store in %s from %s", destination, source_path
        )
        builder = impact_index.DocumentStoreBuilder(
            str(destination), self.block_size, self.zstd_level
        )
        for keys, content in tqdm(
            self.iter_factory(source_path),
            desc="Building document store",
            total=self.doc_count,
        ):
            builder.add(keys, content)
        builder.build()


StreamFactory = Callable[
    [IO[bytes], Callable[[int], None]],
    Iterator[Tuple[dict[str, str], bytes]],
]


class streaming_docstore_builder(FolderResource):
    """Builds an impact-index DocumentStore by streaming from an HTTP URL.

    Unlike :class:`docstore_builder`, the source is never persisted to disk:
    the URL is streamed directly and passed to ``stream_factory`` which
    yields ``(keys, content)`` pairs. The underlying
    ``impact_index.DocumentStoreBuilder`` is opened with a non-zero
    ``checkpoint_frequency`` so that a crash or interruption can be
    recovered on the next run.

    **Resume strategy.** Whenever ``builder.add()`` signals that a
    checkpoint was just written (by returning a truthy value, available
    from impact-index 0.3.1 / 1.3.1+), the builder's position is paired
    with the latest HTTP byte offset reported by the factory and
    persisted to ``{destination}/.stream_progress.json``. On restart,
    the stream is re-opened with an HTTP ``Range`` request starting at
    that byte offset, avoiding a full re-download. If no progress file
    is available (older impact-index, or no safe boundary reached yet),
    the stream is re-opened from the beginning and already-ingested
    records are skipped.

    **Factory signature.** ``stream_factory(stream, mark)`` receives the
    open byte stream and a ``mark`` callback. The factory should call
    ``mark(byte_offset)`` whenever the bytes consumed so far from
    ``stream`` correspond to a position from which parsing can be safely
    resumed (e.g., just past a tar record header boundary). Marks are
    only persisted in conjunction with a successful builder checkpoint.

    Suitable for very large source archives (tens of GB) where keeping
    the raw tarball around is undesirable.
    """

    _PROGRESS_FILE = ".stream_progress.json"

    def __init__(
        self,
        url: str,
        stream_factory: StreamFactory,
        *,
        keys: list[str],
        headers: Optional[dict[str, str]] = None,
        block_size: int = 4096,
        zstd_level: int = 3,
        checkpoint_frequency: int = 100_000,
        doc_count: Optional[int] = None,
        max_retries: int = 5,
        retry_backoff: float = 5.0,
    ):
        """
        :param url: URL of the source archive to stream
        :param stream_factory: ``(stream, mark) -> Iterator[(keys, content)]``.
            The factory should call ``mark(byte_offset)`` at safely
            resumable positions in ``stream`` (byte_offset counted from
            the start of the open stream).
        :param keys: Key field names (documents the schema)
        :param headers: Optional HTTP headers for the request
        :param block_size: Documents per compressed block
        :param zstd_level: Zstandard compression level
        :param checkpoint_frequency: Flush builder state every N
            documents so the build can resume after a crash
        :param doc_count: Optional hint for the progress bar
        :param max_retries: Number of times to re-open the stream after
            a network error before giving up
        :param retry_backoff: Seconds to wait between retries
        """
        super().__init__()
        self.url = url
        self.stream_factory = stream_factory
        self.keys = keys
        self.headers = headers or {}
        self.block_size = block_size
        self.zstd_level = zstd_level
        self.checkpoint_frequency = checkpoint_frequency
        self.doc_count = doc_count
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    @property
    def can_recover(self) -> bool:
        return True

    def _open_stream(self, byte_offset: int = 0) -> IO[bytes]:
        headers = dict(self.headers)
        if byte_offset > 0:
            headers["Range"] = f"bytes={byte_offset}-"
        req = urllib.request.Request(self.url, headers=headers)
        resp = urllib.request.urlopen(req)
        if byte_offset > 0:
            status = getattr(resp, "status", None) or resp.getcode()
            if status != 206:
                resp.close()
                raise RuntimeError(
                    f"Server does not support Range requests (status {status}); "
                    f"cannot resume stream from byte {byte_offset}"
                )
        return resp

    def _progress_path(self, destination: Path) -> Path:
        return destination / self._PROGRESS_FILE

    def _read_progress(self, destination: Path) -> Tuple[Optional[int], int]:
        """Returns (byte_offset, doc_count_at_offset) or (None, 0) if
        missing, malformed, or incompatible with the current URL."""
        p = self._progress_path(destination)
        if not p.exists():
            return None, 0
        try:
            data = json.loads(p.read_text())
            url = data.get("url")
            offset = int(data["byte_offset"])
            count = int(data["doc_count"])
        except (OSError, ValueError, KeyError, TypeError) as exc:
            logging.warning("Ignoring invalid stream progress file %s: %s", p, exc)
            return None, 0
        if url != self.url:
            logging.warning(
                "Stream progress file %s was written for a different URL "
                "(%s != %s); discarding and restarting from scratch",
                p,
                url,
                self.url,
            )
            self._clear_progress(destination)
            return None, 0
        return offset, count

    def _write_progress(
        self, destination: Path, byte_offset: int, doc_count: int
    ) -> None:
        p = self._progress_path(destination)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(
            json.dumps(
                {
                    "url": self.url,
                    "byte_offset": byte_offset,
                    "doc_count": doc_count,
                }
            )
        )
        tmp.replace(p)

    def _clear_progress(self, destination: Path) -> None:
        p = self._progress_path(destination)
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    def _ingest_from_stream(self, builder: Any, pbar: Any, destination: Path) -> None:
        """Stream records and add them to the builder.

        Uses the persisted progress file (if any) to issue an HTTP Range
        request and avoid re-streaming bytes that were consumed before
        the last checkpoint. Falls back to a full re-stream + skip when
        no progress is available.
        """
        builder_docs = builder.num_documents()
        saved_offset, saved_count = self._read_progress(destination)

        if saved_offset is not None and saved_count > builder_docs:
            logging.warning(
                "Stream progress file is ahead of the builder "
                "(saved doc_count=%d, builder=%d); discarding and "
                "restarting the stream from the beginning",
                saved_count,
                builder_docs,
            )
            self._clear_progress(destination)
            saved_offset, saved_count = None, 0

        if saved_offset is not None and saved_offset > 0:
            start_offset = saved_offset
            skip = builder_docs - saved_count
            logging.info(
                "Range-resuming stream at byte %d (doc %d); skipping %d docs",
                start_offset,
                saved_count,
                skip,
            )
        else:
            start_offset = 0
            skip = builder_docs
            if skip:
                logging.info("Full re-stream; skipping %d already-ingested docs", skip)

        last_mark_offset: Optional[int] = None
        last_mark_count: int = 0

        def mark(byte_offset: int) -> None:
            nonlocal last_mark_offset, last_mark_count
            last_mark_offset = byte_offset
            last_mark_count = builder.num_documents()

        seen = 0
        with self._open_stream(start_offset) as stream:
            for keys, content in self.stream_factory(stream, mark):
                seen += 1
                if seen <= skip:
                    continue
                checkpointed = builder.add(keys, content)
                pbar.update(1)
                if checkpointed and last_mark_offset is not None:
                    # Translate the in-stream offset to an absolute HTTP
                    # offset (factory marks are relative to the start of
                    # the current open stream).
                    absolute = start_offset + last_mark_offset
                    self._write_progress(destination, absolute, last_mark_count)

    def _download(self, destination: Path) -> None:
        import impact_index
        import time

        destination.mkdir(parents=True, exist_ok=True)

        builder = impact_index.DocumentStoreBuilder(
            str(destination),
            self.block_size,
            self.zstd_level,
            checkpoint_frequency=self.checkpoint_frequency,
        )
        initial = builder.num_documents()
        if initial:
            logging.info(
                "Resuming docstore build at %s from document %d", destination, initial
            )
        else:
            logging.info(
                "Building streaming document store in %s from %s",
                destination,
                self.url,
            )

        with tqdm(
            desc="Building document store",
            total=self.doc_count,
            initial=initial,
        ) as pbar:
            attempt = 0
            while True:
                try:
                    self._ingest_from_stream(builder, pbar, destination)
                    break
                except KeyboardInterrupt:
                    builder.checkpoint()
                    raise
                except Exception as exc:
                    attempt += 1
                    builder.checkpoint()
                    if attempt > self.max_retries:
                        logging.error(
                            "Giving up after %d retries: %s", self.max_retries, exc
                        )
                        raise
                    logging.warning(
                        "Stream error after %d documents (attempt %d/%d): %s. "
                        "Retrying in %.1fs",
                        builder.num_documents(),
                        attempt,
                        self.max_retries,
                        exc,
                        self.retry_backoff,
                    )
                    time.sleep(self.retry_backoff)

        builder.build()
        self._clear_progress(destination)


# Re-exported for backwards compatibility; the canonical home is
# :mod:`datamaestro_ir.utils.streaming`.
from datamaestro_ir.utils.streaming import ByteCountingStream  # noqa: E402, F401
