"""Minimal GGUF header reader (stdlib only).

Reads the metadata key/values and the tensor table of a GGUF file without
loading any weights, so we can reason about memory before starting a model.
Format reference: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

# value type id -> struct format
_SCALARS = {
    0: "<B",  # uint8
    1: "<b",  # int8
    2: "<H",  # uint16
    3: "<h",  # int16
    4: "<I",  # uint32
    5: "<i",  # int32
    6: "<f",  # float32
    7: "<?",  # bool
    10: "<Q",  # uint64
    11: "<q",  # int64
    12: "<d",  # float64
}
_T_STRING = 8
_T_ARRAY = 9

_BLOCK_RE = re.compile(r"^blk\.(\d+)\.")


class GGUFError(ValueError):
    pass


@dataclass(frozen=True)
class TensorInfo:
    name: str
    dims: tuple[int, ...]
    dtype: int
    offset: int  # relative to start of the tensor data section
    nbytes: int

    @property
    def block(self) -> int | None:
        """Transformer block index, or None for non-block tensors."""
        m = _BLOCK_RE.match(self.name)
        return int(m.group(1)) if m else None


@dataclass
class GGUFHeader:
    version: int
    metadata: dict[str, object]
    tensors: list[TensorInfo]
    file_size: int
    data_offset: int

    @property
    def architecture(self) -> str:
        return str(self.metadata.get("general.architecture", ""))

    def arch_key(self, suffix: str, default: object = None) -> object:
        return self.metadata.get(f"{self.architecture}.{suffix}", default)

    @property
    def n_layer(self) -> int:
        n = self.arch_key("block_count")
        if n is None:
            blocks = [t.block for t in self.tensors if t.block is not None]
            if not blocks:
                raise GGUFError("cannot determine block count")
            return max(blocks) + 1
        return int(n)  # type: ignore[arg-type]

    @property
    def n_embd(self) -> int:
        return int(self.arch_key("embedding_length", 0))  # type: ignore[arg-type]

    @property
    def n_head(self) -> int:
        return int(self.arch_key("attention.head_count", 0))  # type: ignore[arg-type]

    @property
    def n_head_kv(self) -> int:
        return int(self.arch_key("attention.head_count_kv", self.n_head))  # type: ignore[arg-type]

    @property
    def n_ctx_train(self) -> int:
        return int(self.arch_key("context_length", 0))  # type: ignore[arg-type]

    @property
    def head_dim(self) -> int:
        key_len = self.arch_key("attention.key_length")
        if key_len:
            return int(key_len)  # type: ignore[arg-type]
        if not self.n_head:
            raise GGUFError("cannot determine attention head size")
        return self.n_embd // self.n_head

    def block_bytes(self) -> list[int]:
        """Weight bytes per transformer block, index = block number."""
        out = [0] * self.n_layer
        for t in self.tensors:
            b = t.block
            if b is not None and b < len(out):
                out[b] += t.nbytes
        return out

    def non_block_bytes(self) -> dict[str, int]:
        """Weight bytes outside the blocks, split into input / output side."""
        inp = outp = 0
        for t in self.tensors:
            if t.block is not None:
                continue
            if t.name.startswith(("token_embd", "position_embd", "token_types")):
                inp += t.nbytes
            else:
                outp += t.nbytes
        return {"input": inp, "output": outp}


def _read(f: BinaryIO, fmt: str):
    size = struct.calcsize(fmt)
    raw = f.read(size)
    if len(raw) != size:
        raise GGUFError("unexpected end of file")
    return struct.unpack(fmt, raw)[0]


def _read_string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    if n > 1 << 30:
        raise GGUFError("implausible string length")
    raw = f.read(n)
    if len(raw) != n:
        raise GGUFError("unexpected end of file")
    return raw.decode("utf-8", errors="replace")


def _read_value(f: BinaryIO, vtype: int, *, keep_arrays: bool):
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _T_STRING:
        return _read_string(f)
    if vtype == _T_ARRAY:
        etype = _read(f, "<I")
        count = _read(f, "<Q")
        if etype in _SCALARS and not keep_arrays:
            # skip big numeric arrays (token scores etc.) cheaply
            f.seek(struct.calcsize(_SCALARS[etype]) * count, 1)
            return None
        items = [_read_value(f, etype, keep_arrays=keep_arrays) for _ in range(count)]
        return items if keep_arrays else None
    raise GGUFError(f"unknown GGUF value type {vtype}")


def read_header(path: str | Path, *, keep_arrays: bool = False) -> GGUFHeader:
    """Parse the header of a GGUF file. Arrays (vocab etc.) are dropped by default."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as f:
        if f.read(4) != MAGIC:
            raise GGUFError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        if version < 2:
            raise GGUFError(f"unsupported GGUF version {version}")
        n_tensors = _read(f, "<Q")
        n_kv = _read(f, "<Q")

        metadata: dict[str, object] = {}
        for _ in range(n_kv):
            key = _read_string(f)
            vtype = _read(f, "<I")
            value = _read_value(f, vtype, keep_arrays=keep_arrays)
            if value is not None:
                metadata[key] = value

        raw_infos = []
        for _ in range(n_tensors):
            name = _read_string(f)
            n_dims = _read(f, "<I")
            dims = tuple(_read(f, "<Q") for _ in range(n_dims))
            dtype = _read(f, "<I")
            offset = _read(f, "<Q")
            raw_infos.append((name, dims, dtype, offset))

        alignment = int(metadata.get("general.alignment", DEFAULT_ALIGNMENT))  # type: ignore[arg-type]
        pos = f.tell()
        data_offset = pos + (-pos % alignment)

    # Tensor sizes from consecutive offsets: exact, and needs no dtype size table.
    data_size = max(file_size - data_offset, 0)
    order = sorted(range(len(raw_infos)), key=lambda i: raw_infos[i][3])
    sizes = [0] * len(raw_infos)
    for pos_in_order, idx in enumerate(order):
        start = raw_infos[idx][3]
        end = raw_infos[order[pos_in_order + 1]][3] if pos_in_order + 1 < len(order) else data_size
        sizes[idx] = max(end - start, 0)

    tensors = [
        TensorInfo(name=n, dims=d, dtype=t, offset=o, nbytes=sizes[i])
        for i, (n, d, t, o) in enumerate(raw_infos)
    ]
    return GGUFHeader(
        version=version,
        metadata=metadata,
        tensors=tensors,
        file_size=file_size,
        data_offset=data_offset,
    )
