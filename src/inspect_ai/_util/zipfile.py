"""ZIP file helpers and monkey-patches for zstd support.

On Python < 3.14 we import ``zipfile_zstd`` to monkey-patch zstandard
compression into the stdlib ``zipfile`` module. Python 3.14+ handles zstd
natively via stdlib.

Additionally, we install a second monkey-patch on ``zipfile._get_compressor``
that caps each emitted zstd frame at ``_MAX_INPUT_PER_FRAME`` bytes of input.
Large single zstd frames (>= 256 MiB compressed) trigger an overflow bug in
the pure-JS ``fzstd`` decoder used by our viewers; multi-framing keeps every
frame well under that threshold.
"""

from __future__ import annotations

import logging
import sys
import zipfile
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import zstandard

logger = logging.getLogger(__name__)

# On Python < 3.14, monkey-patch zipfile to support zstandard compression.
if sys.version_info < (3, 14):
    import zipfile_zstd  # type: ignore[import-not-found, import-untyped]  # noqa: F401

# Resolve once so the rest of the module can use a plain int; also fails loudly
# at import time rather than at first call if the attribute is ever missing.
_ZIP_ZSTANDARD: int = zipfile.ZIP_ZSTANDARD  # type: ignore[attr-defined]

zipfile_compress_kwargs: dict[str, Any] = {
    "compression": _ZIP_ZSTANDARD,
    "compresslevel": None,
}


# 200 MiB. Well under fzstd's 256 MiB (2^28) compressed-frame overflow
# threshold, applied to *input* bytes which compress smaller.
_MAX_INPUT_PER_FRAME = 200 * 1024 * 1024

# Matches zipfile_zstd's hardcoded default so multi-frame and single-frame zip
# writes use the same thread count. If zipfile_zstd ever changes its default,
# update here too.
_ZSTD_THREADS = 12

# A zstd block expands to at most 128 KiB. Feeding four compressed bytes at a
# time lets an RLE block complete without allowing a single call to materialize
# more than one block's output.
_ZSTD_DECOMPRESS_INPUT_SIZE = 4

# Release consumed compressed input without repeatedly moving a large buffer.
_ZSTD_PENDING_COMPACTION_SIZE = 64 * 1024


class _MultiFrameZstdCompressObj:
    """A zstd compressobj that chunks its output into multiple frames.

    Wraps a ``zstandard`` compressobj and flushes it (finalizing the current
    frame) and replaces it (starting a new frame) every
    ``_MAX_INPUT_PER_FRAME`` bytes of input. Multi-frame zstd streams are
    valid per spec -- any compliant decoder reads them transparently.
    """

    def __init__(self, factory: Callable[[], zstandard.ZstdCompressionObj]) -> None:
        self._factory = factory
        self._obj = factory()
        self._input_bytes = 0

    def compress(self, data: bytes) -> bytes:
        view = memoryview(data)
        pieces: list[bytes] = []
        offset = 0
        n = len(view)
        while offset < n:
            remaining_cap = _MAX_INPUT_PER_FRAME - self._input_bytes
            end = min(offset + remaining_cap, n)
            chunk = view[offset:end]
            pieces.append(self._obj.compress(chunk))
            self._input_bytes += end - offset
            offset = end
            if self._input_bytes >= _MAX_INPUT_PER_FRAME:
                pieces.append(self._obj.flush())
                self._obj = self._factory()
                self._input_bytes = 0
        return b"".join(pieces)

    def flush(self) -> bytes:
        # If the last ``compress()`` call landed exactly on the frame boundary,
        # ``self._obj`` was replaced with a fresh compressobj that has received
        # no bytes. Flushing it would append an empty 9-byte trailing frame.
        if self._input_bytes == 0:
            return b""
        return self._obj.flush()


class _MultiFrameZstdDecompressObj:
    """A zstd decompressobj that transparently spans multiple frames.

    ``zstandard.ZstdDecompressor().decompressobj()`` stops at the first frame
    boundary and marks ``eof=True``.  When the compressor splits large entries
    into multiple frames (see ``_MultiFrameZstdCompressObj``), the reader must
    recognise the frame boundary, start a fresh inner decompressobj, and
    continue until all compressed bytes have been consumed.

    The stdlib ``zipfile._read1`` path for non-deflate compression drives EOF
    from the compressed input count. We therefore report ``eof=False`` always
    and retain input after a completed frame for a fresh inner decompressobj.

    Since CPython gh-156002, ``_read1`` calls ``decompress(data, max_length)``
    and reads more compressed bytes only while ``needs_input`` is True. The
    zstandard decompressobj materializes all output for every input it receives,
    so bounded calls feed it only a few compressed bytes at a time and retain
    any output beyond ``max_length`` for subsequent empty-input drains.
    """

    def __init__(self) -> None:
        import zstandard as zstd  # local import -- already a hard dep

        self._dctx: zstandard.ZstdDecompressor = zstd.ZstdDecompressor()
        self._obj: zstandard.ZstdDecompressionObj = self._dctx.decompressobj()
        self._pending = bytearray()
        self._pending_offset = 0
        self._output = bytearray()
        self._output_offset = 0

    def decompress(self, data: bytes, max_length: int = -1) -> bytes:
        self._pending.extend(data)
        if max_length < 0:
            return self._decompress_unbounded()
        if max_length == 0:
            return b""
        if self._output:
            return self._take_output(max_length)

        while self._pending_offset < len(self._pending):
            chunk = memoryview(self._pending)[
                self._pending_offset : self._pending_offset
                + _ZSTD_DECOMPRESS_INPUT_SIZE
            ]
            result = self._decompress_chunk(chunk)
            del chunk
            self._discard_consumed_input()
            self._output.extend(result)
            if len(self._output) - self._output_offset >= max_length:
                break
        return self._take_output(max_length) if self._output else b""

    def _decompress_unbounded(self) -> bytes:
        pieces: list[bytes] = []
        if self._output:
            pieces.append(self._take_output(-1))

        while self._pending_offset < len(self._pending):
            chunk = memoryview(self._pending)[self._pending_offset :]
            pieces.append(self._decompress_chunk(chunk))
            del chunk
            self._discard_consumed_input()

        return b"".join(pieces)

    def _decompress_chunk(self, chunk: memoryview) -> bytes:
        result = self._obj.decompress(chunk)
        if self._obj.eof:
            consumed = len(chunk) - len(self._obj.unused_data)
            self._obj = self._dctx.decompressobj()
        else:
            consumed = len(chunk) - len(self._obj.unconsumed_tail)
        if consumed <= 0:
            raise RuntimeError("zstd decompressor made no progress")
        self._pending_offset += consumed
        return result

    def _discard_consumed_input(self) -> None:
        if self._pending_offset == len(self._pending):
            self._pending.clear()
            self._pending_offset = 0
        elif self._pending_offset >= _ZSTD_PENDING_COMPACTION_SIZE:
            del self._pending[: self._pending_offset]
            self._pending_offset = 0

    def _take_output(self, max_length: int) -> bytes:
        end = len(self._output)
        if max_length >= 0:
            end = min(end, self._output_offset + max_length)
        result = bytes(self._output[self._output_offset : end])
        self._output_offset = end
        if self._output_offset == len(self._output):
            self._output.clear()
            self._output_offset = 0
        return result

    def flush(self) -> bytes:
        return b""

    @property
    def eof(self) -> bool:
        # Always False: let compress_left drive the outer EOF check.
        return False

    @property
    def needs_input(self) -> bool:
        return not self._output and self._pending_offset == len(self._pending)


def _install_multiframe_patches() -> None:
    """Install multi-frame zstd compressor and decompressor patches.

    Idempotent. Wraps whatever ``_get_compressor`` / ``_get_decompressor`` were
    installed before us (stdlib on Py >= 3.14; ``zipfile_zstd``'s versions on
    Py < 3.14), so compression level and thread count are preserved.
    """
    if getattr(zipfile, "_inspect_ai_multiframe_installed", False):
        return

    original_compressor = zipfile._get_compressor  # type: ignore[attr-defined]
    original_decompressor = zipfile._get_decompressor  # type: ignore[attr-defined]

    def patched_compressor(compress_type: int, compresslevel: int | None = None) -> Any:
        if compress_type == _ZIP_ZSTANDARD:
            # Share one ``ZstdCompressor`` across all frames of the entry.
            # Delegating to ``original_compressor`` would instead create a
            # fresh ``ZstdCompressor(threads=N)`` per frame, re-initialising
            # its thread pool every 200 MiB.
            import zstandard

            level = 3 if compresslevel is None else compresslevel
            compressor = zstandard.ZstdCompressor(level=level, threads=_ZSTD_THREADS)
            return _MultiFrameZstdCompressObj(compressor.compressobj)
        return original_compressor(compress_type, compresslevel)

    def patched_decompressor(compress_type: int) -> Any:
        if compress_type == _ZIP_ZSTANDARD:
            return _MultiFrameZstdDecompressObj()
        return original_decompressor(compress_type)

    zipfile._get_compressor = patched_compressor  # type: ignore[attr-defined]
    zipfile._get_decompressor = patched_decompressor  # type: ignore[attr-defined]
    zipfile._inspect_ai_multiframe_installed = True  # type: ignore[attr-defined]


_install_multiframe_patches()


__all__ = ["zipfile_compress_kwargs"]
