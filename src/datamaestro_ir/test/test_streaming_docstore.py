"""Tests for the streaming document store builder."""

import gzip
import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path
from typing import Callable, IO, Iterator

import impact_index

from datamaestro_ir.download.docstore import streaming_docstore_builder
from datamaestro_ir.utils.streaming import iter_tar_gz_jsonl


def _make_tar_bytes(num_docs: int, shards: int = 3) -> bytes:
    """Build a tar of gzipped JSON-Lines in-memory (matches MS MARCO v2 layout)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        per_shard = (num_docs + shards - 1) // shards
        doc_id = 0
        for shard in range(shards):
            inner = io.BytesIO()
            with gzip.GzipFile(fileobj=inner, mode="wb") as gz:
                for _ in range(per_shard):
                    if doc_id >= num_docs:
                        break
                    line = f'{{"id":"doc{doc_id:04d}","text":"body {doc_id}"}}\n'
                    gz.write(line.encode("utf-8"))
                    doc_id += 1
            data = inner.getvalue()
            info = tarfile.TarInfo(name=f"shard_{shard:02d}.gz")
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _stream_factory(
    stream: IO[bytes], mark: Callable[[int], None]
) -> Iterator[tuple[dict[str, str], bytes]]:
    for data in iter_tar_gz_jsonl(stream, mark):
        yield {"id": data["id"]}, data["text"].encode("utf-8")


def _first_shard_aligned_end(tar_bytes: bytes) -> int:
    """Byte offset of the start of the 2nd tar record header."""
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
        members = tf.getmembers()
    # With two members, the second one's offset is where the next
    # record header begins.
    assert len(members) >= 2
    return members[1].offset


class _TruncatingStream(io.RawIOBase):
    """Bytes-backed stream that raises after ``fail_after`` bytes are read."""

    def __init__(self, data: bytes, fail_after: int):
        self._buf = io.BytesIO(data)
        self._fail_after = fail_after
        self._consumed = 0

    def readable(self) -> bool:
        return True

    def read(self, size: int = -1) -> bytes:
        remaining = self._fail_after - self._consumed
        if remaining <= 0:
            raise ConnectionResetError("simulated mid-stream failure")
        if size is None or size < 0:
            size = remaining
        size = min(size, remaining)
        chunk = self._buf.read(size)
        self._consumed += len(chunk)
        return chunk

    def readinto(self, b):
        data = self.read(len(b))
        n = len(data)
        b[:n] = data
        return n

    def close(self):
        self._buf.close()
        super().close()


class _RangeableTar:
    """Serves ``tar_bytes`` as an HTTP-like stream with Range support.

    ``open(byte_offset)`` returns a fresh BytesIO seeked to that offset,
    mimicking a 206 response from a Range request.
    """

    def __init__(self, tar_bytes: bytes):
        self.tar_bytes = tar_bytes
        self.range_requests: list[int] = []

    def open(self, byte_offset: int = 0) -> IO[bytes]:
        self.range_requests.append(byte_offset)
        return io.BytesIO(self.tar_bytes[byte_offset:])


def _make_builder(**kwargs) -> streaming_docstore_builder:
    kwargs.setdefault("url", "http://example.invalid/archive.tar")
    kwargs.setdefault("stream_factory", _stream_factory)
    kwargs.setdefault("keys", ["id"])
    return streaming_docstore_builder(**kwargs)


class StreamingDocStoreBuilderTest(unittest.TestCase):
    def test_full_build(self):
        tar_bytes = _make_tar_bytes(num_docs=20, shards=3)
        source = _RangeableTar(tar_bytes)
        resource = _make_builder(doc_count=20, checkpoint_frequency=5)
        resource._open_stream = source.open  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"
            resource._download(dest)

            store = impact_index.DocumentStore.load(str(dest))
            self.assertEqual(store.num_documents(), 20)
            docs = store.get_by_number(list(range(20)))
            for i, doc in enumerate(docs):
                self.assertEqual(doc.keys["id"], f"doc{i:04d}")
                self.assertEqual(doc.content, f"body {i}".encode("utf-8"))
            # Progress file should be removed after successful build.
            self.assertFalse((dest / ".stream_progress.json").exists())

    def test_resume_after_crash_full_restream(self):
        """Crash before any mark is persisted ⇒ fall back to full restream."""
        tar_bytes = _make_tar_bytes(num_docs=20, shards=3)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"

            def open_truncated(byte_offset=0):
                self.assertEqual(byte_offset, 0)
                return _TruncatingStream(tar_bytes, 1024)

            resource = _make_builder(
                doc_count=20, checkpoint_frequency=3, max_retries=0
            )
            resource._open_stream = open_truncated  # type: ignore[assignment]
            with self.assertRaises(Exception):
                resource._download(dest)

            # Some docs should have been checkpointed by the builder.
            probe = impact_index.DocumentStoreBuilder(str(dest), checkpoint_frequency=3)
            self.assertGreater(probe.num_documents(), 0)
            self.assertLess(probe.num_documents(), 20)
            del probe

            source = _RangeableTar(tar_bytes)
            resource2 = _make_builder(
                doc_count=20, checkpoint_frequency=3, max_retries=0
            )
            resource2._open_stream = source.open  # type: ignore[assignment]
            resource2._download(dest)

            # If the truncation happened mid first shard, no mark was
            # persisted and the retry re-opens from byte 0.
            self.assertEqual(source.range_requests[0], 0)
            store = impact_index.DocumentStore.load(str(dest))
            self.assertEqual(store.num_documents(), 20)

    def test_retry_recovers_transient_failures(self):
        tar_bytes = _make_tar_bytes(num_docs=20, shards=3)
        source = _RangeableTar(tar_bytes)

        calls = {"n": 0}

        def open_stream(byte_offset=0):
            calls["n"] += 1
            if calls["n"] == 1:
                return _TruncatingStream(tar_bytes, 1024)
            return source.open(byte_offset)

        resource = _make_builder(
            doc_count=20,
            checkpoint_frequency=3,
            max_retries=3,
            retry_backoff=0.0,
        )
        resource._open_stream = open_stream  # type: ignore[assignment]

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"
            resource._download(dest)
            store = impact_index.DocumentStore.load(str(dest))
            self.assertEqual(store.num_documents(), 20)

    def test_range_resume_after_checkpoint(self):
        """With persisted progress, a restart Range-resumes the stream."""
        tar_bytes = _make_tar_bytes(num_docs=30, shards=3)
        source = _RangeableTar(tar_bytes)
        first_shard_end = _first_shard_aligned_end(tar_bytes)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"
            dest.mkdir()

            # Pre-populate: ingest the 10 docs of the first shard so
            # the builder's state matches the progress file we write.
            builder = impact_index.DocumentStoreBuilder(
                str(dest), checkpoint_frequency=5
            )
            with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
                first = tf.getmembers()[0]
                inner = tf.extractfile(first)
                with gzip.open(inner, "rb") as fp:
                    for line in fp:
                        data = json.loads(line)
                        builder.add(
                            {"id": data["id"]},
                            data["text"].encode("utf-8"),
                        )
            builder.checkpoint()
            del builder

            (dest / ".stream_progress.json").write_text(
                json.dumps(
                    {
                        "url": "http://example.invalid/archive.tar",
                        "byte_offset": first_shard_end,
                        "doc_count": 10,
                    }
                )
            )

            resource = _make_builder(doc_count=30, checkpoint_frequency=5)
            resource._open_stream = source.open  # type: ignore[assignment]
            resource._download(dest)

            # The first open call should have issued a Range request
            # exactly at the persisted offset.
            self.assertEqual(source.range_requests, [first_shard_end])

            store = impact_index.DocumentStore.load(str(dest))
            self.assertEqual(store.num_documents(), 30)
            docs = store.get_by_number(list(range(30)))
            for i, doc in enumerate(docs):
                self.assertEqual(doc.keys["id"], f"doc{i:04d}")

    def test_inconsistent_progress_is_discarded(self):
        """Progress file ahead of the builder ⇒ discard + restart from 0."""
        tar_bytes = _make_tar_bytes(num_docs=15, shards=3)
        source = _RangeableTar(tar_bytes)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"
            dest.mkdir()

            # Builder empty, but progress claims 10 docs ingested.
            (dest / ".stream_progress.json").write_text(
                json.dumps(
                    {
                        "url": "http://example.invalid/archive.tar",
                        "byte_offset": 999_999,
                        "doc_count": 10,
                    }
                )
            )

            resource = _make_builder(doc_count=15, checkpoint_frequency=5)
            resource._open_stream = source.open  # type: ignore[assignment]
            resource._download(dest)

            # Inconsistency detected ⇒ opened at byte 0, not 999_999.
            self.assertEqual(source.range_requests, [0])
            self.assertFalse((dest / ".stream_progress.json").exists())

            store = impact_index.DocumentStore.load(str(dest))
            self.assertEqual(store.num_documents(), 15)

    def test_mismatched_url_is_discarded(self):
        tar_bytes = _make_tar_bytes(num_docs=10, shards=2)
        source = _RangeableTar(tar_bytes)

        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "store"
            dest.mkdir()
            (dest / ".stream_progress.json").write_text(
                json.dumps(
                    {
                        "url": "http://example.invalid/OTHER.tar",
                        "byte_offset": 5000,
                        "doc_count": 5,
                    }
                )
            )

            resource = _make_builder(doc_count=10, checkpoint_frequency=3)
            resource._open_stream = source.open  # type: ignore[assignment]
            resource._download(dest)

            self.assertEqual(source.range_requests, [0])


if __name__ == "__main__":
    unittest.main()
