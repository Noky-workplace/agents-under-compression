"""Builds tiny synthetic GGUF files for tests."""

from __future__ import annotations

import struct
from pathlib import Path

ALIGN = 32


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _s(key) + struct.pack("<I", vtype) + payload


def u32(key: str, v: int) -> bytes:
    return _kv(key, 4, struct.pack("<I", v))


def string(key: str, v: str) -> bytes:
    return _kv(key, 8, _s(v))


def f32_array(key: str, values: list[float]) -> bytes:
    body = struct.pack("<IQ", 6, len(values)) + b"".join(struct.pack("<f", x) for x in values)
    return _kv(key, 9, body)


def str_array(key: str, values: list[str]) -> bytes:
    body = struct.pack("<IQ", 8, len(values)) + b"".join(_s(x) for x in values)
    return _kv(key, 9, body)


def write_gguf(
    path: Path,
    *,
    n_layer: int = 4,
    n_embd: int = 64,
    n_head: int = 8,
    n_head_kv: int = 2,
    block_bytes: int = 4096,
    embd_bytes: int = 1024,
    output_bytes: int = 512,
    arch: str = "qwen3",
) -> Path:
    kvs = [
        string("general.architecture", arch),
        u32(f"{arch}.block_count", n_layer),
        u32(f"{arch}.embedding_length", n_embd),
        u32(f"{arch}.attention.head_count", n_head),
        u32(f"{arch}.attention.head_count_kv", n_head_kv),
        u32(f"{arch}.context_length", 32768),
        f32_array("tokenizer.ggml.scores", [0.0] * 10),
        str_array("tokenizer.ggml.tokens", ["a", "bb", "ccc"]),
    ]
    # two tensors per block (half the bytes each), plus embeddings and output head
    tensors: list[tuple[str, int]] = [("token_embd.weight", embd_bytes)]
    for b in range(n_layer):
        tensors.append((f"blk.{b}.attn_q.weight", block_bytes // 2))
        tensors.append((f"blk.{b}.ffn_up.weight", block_bytes // 2))
    tensors.append(("output_norm.weight", output_bytes // 2))
    tensors.append(("output.weight", output_bytes // 2))

    infos = b""
    offset = 0
    for name, nbytes in tensors:
        infos += _s(name) + struct.pack("<I", 1) + struct.pack("<Q", nbytes)
        infos += struct.pack("<I", 0) + struct.pack("<Q", offset)
        offset += nbytes

    head = b"GGUF" + struct.pack("<I", 3) + struct.pack("<QQ", len(tensors), len(kvs))
    blob = head + b"".join(kvs) + infos
    blob += b"\x00" * (-len(blob) % ALIGN)
    blob += b"\x01" * offset
    path.write_bytes(blob)
    return path
