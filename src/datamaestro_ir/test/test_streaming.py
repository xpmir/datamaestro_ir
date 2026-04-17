"""Tests for :mod:`datamaestro_ir.utils.streaming`."""

import gzip
import io
import tarfile
import unittest

from datamaestro_ir.utils.streaming import (
    ByteCountingStream,
    iter_tar_gz_jsonl,
)


def _make_tar(members: dict[str, bytes]) -> bytes:
    """Build a tar in-memory from ``name -> raw bytes``."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            tf.addfile(info, io.BytesIO(payload))
    return buf.getvalue()


def _gzip_lines(lines: list[dict]) -> bytes:
    import json

    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        for line in lines:
            gz.write((json.dumps(line) + "\n").encode("utf-8"))
    return out.getvalue()


class ByteCountingStreamTest(unittest.TestCase):
    def test_tracks_bytes_read_via_read(self):
        stream = ByteCountingStream(io.BytesIO(b"hello world"))
        self.assertEqual(stream.read(5), b"hello")
        self.assertEqual(stream.bytes_read, 5)
        self.assertEqual(stream.read(), b" world")
        self.assertEqual(stream.bytes_read, 11)

    def test_tracks_bytes_read_via_readinto(self):
        stream = ByteCountingStream(io.BytesIO(b"abcdef"))
        buf = bytearray(4)
        n = stream.readinto(buf)
        self.assertEqual(n, 4)
        self.assertEqual(bytes(buf), b"abcd")
        self.assertEqual(stream.bytes_read, 4)


class IterTarGzJsonlTest(unittest.TestCase):
    def test_yields_all_records_in_order(self):
        shard_a = _gzip_lines([{"id": 0}, {"id": 1}, {"id": 2}])
        shard_b = _gzip_lines([{"id": 3}, {"id": 4}])
        tar_bytes = _make_tar({"a.gz": shard_a, "b.gz": shard_b})

        marks: list[int] = []
        records = list(iter_tar_gz_jsonl(io.BytesIO(tar_bytes), marks.append))
        self.assertEqual(records, [{"id": i} for i in range(5)])

    def test_marks_at_shard_boundaries(self):
        shard_a = _gzip_lines([{"id": 0}, {"id": 1}])
        shard_b = _gzip_lines([{"id": 2}])
        tar_bytes = _make_tar({"a.gz": shard_a, "b.gz": shard_b})

        marks: list[int] = []
        list(iter_tar_gz_jsonl(io.BytesIO(tar_bytes), marks.append))

        # Two shards ⇒ two marks, one per boundary.
        self.assertEqual(len(marks), 2)
        for offset in marks:
            self.assertEqual(offset % 512, 0)

        # The first mark must land exactly on the 2nd tar member header.
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            second_offset = tf.getmembers()[1].offset
        self.assertEqual(marks[0], second_offset)

    def test_range_resume_parses_from_second_member(self):
        shard_a = _gzip_lines([{"id": 0}])
        shard_b = _gzip_lines([{"id": 1}, {"id": 2}])
        tar_bytes = _make_tar({"a.gz": shard_a, "b.gz": shard_b})

        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r") as tf:
            second_offset = tf.getmembers()[1].offset

        sliced = io.BytesIO(tar_bytes[second_offset:])
        records = list(iter_tar_gz_jsonl(sliced, lambda _off: None))
        self.assertEqual(records, [{"id": 1}, {"id": 2}])

    def test_skips_non_matching_members(self):
        shard = _gzip_lines([{"id": 0}])
        tar_bytes = _make_tar({"README.txt": b"ignore me", "data.gz": shard})
        records = list(iter_tar_gz_jsonl(io.BytesIO(tar_bytes), lambda _off: None))
        self.assertEqual(records, [{"id": 0}])

    def test_custom_suffix(self):
        shard = _gzip_lines([{"k": "v"}])
        tar_bytes = _make_tar({"data.jsonl.gz": shard})
        records = list(
            iter_tar_gz_jsonl(
                io.BytesIO(tar_bytes), lambda _off: None, suffix=".jsonl.gz"
            )
        )
        self.assertEqual(records, [{"k": "v"}])

    def test_ignores_blank_lines(self):
        import json

        payload = io.BytesIO()
        with gzip.GzipFile(fileobj=payload, mode="wb") as gz:
            gz.write(b"\n")
            gz.write((json.dumps({"id": 0}) + "\n").encode("utf-8"))
            gz.write(b"   \n")
            gz.write((json.dumps({"id": 1}) + "\n").encode("utf-8"))
        tar_bytes = _make_tar({"data.gz": payload.getvalue()})

        records = list(iter_tar_gz_jsonl(io.BytesIO(tar_bytes), lambda _off: None))
        self.assertEqual(records, [{"id": 0}, {"id": 1}])


if __name__ == "__main__":
    unittest.main()
