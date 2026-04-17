"""Streaming helpers for building document stores without staging to disk.

These utilities are meant to be composed with
:class:`datamaestro_ir.download.docstore.streaming_docstore_builder`:
the builder opens an HTTP stream and hands it to a factory whose job is
to yield ``(keys, content)`` pairs. Factories built on top of
:func:`iter_tar_gz_jsonl` get resumable checkpoints for free — the
helper calls ``mark(byte_offset)`` at each tar-member boundary so the
builder can persist a resumable HTTP ``Range`` position.
"""

from __future__ import annotations

import gzip
import json
import tarfile
from typing import Any, Callable, IO, Iterator


_TAR_BLOCK = 512


class ByteCountingStream:
    """File-like wrapper that tracks how many bytes have been read.

    Wrap an HTTP response (or any byte stream) with this before handing
    it to ``tarfile.open(mode='r|')`` so factories can report the raw
    byte offset consumed so far.
    """

    def __init__(self, stream: IO[bytes]):
        self._stream = stream
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._stream.read(size)
        self.bytes_read += len(data)
        return data

    def readinto(self, b) -> int:
        data = self._stream.read(len(b))
        n = len(data)
        b[:n] = data
        self.bytes_read += n
        return n

    def close(self) -> None:
        self._stream.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _aligned_up(value: int, block: int = _TAR_BLOCK) -> int:
    """Round ``value`` up to the next multiple of ``block``."""
    return ((value + block - 1) // block) * block


def iter_tar_gz_jsonl(
    stream: IO[bytes],
    mark: Callable[[int], None],
    *,
    suffix: str = ".gz",
) -> Iterator[dict]:
    """Iterate JSON records from a tar of gzipped JSON-Lines members.

    Reads ``stream`` in streaming tar mode (``r|``); each tar member
    whose name ends in ``suffix`` is decompressed and parsed as JSONL,
    and every parsed record is yielded as a dict. After each member is
    fully consumed, ``mark(byte_offset)`` is called with the offset
    (relative to the start of ``stream``) of the next tar record
    header — a valid ``Range`` resume point (tar has no central index,
    just sequential 512-byte blocks).

    :param stream: Byte stream positioned at the start of a tar file
    :param mark: Callback invoked at each member boundary
    :param suffix: Only members with this name suffix are decompressed
    """
    with tarfile.open(fileobj=stream, mode="r|") as tarf:
        for record in tarf:
            if not record.name.endswith(suffix):
                # Account for skipped members too so the next mark
                # still points at the right header.
                continue
            inner = tarf.extractfile(record)
            if inner is None:
                continue
            with gzip.open(inner, "rb") as fp:
                for line in fp:
                    if not line.strip():
                        continue
                    yield json.loads(line)
            # tarfile records carry their byte offset directly; the next
            # header starts immediately after this member's padded data.
            next_header = record.offset + _TAR_BLOCK + _aligned_up(record.size)
            mark(next_header)
