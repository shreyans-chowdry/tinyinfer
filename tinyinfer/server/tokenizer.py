"""Tokenizer wrapper and incremental detokenisation for streaming."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol


class Tokenizer(Protocol):
    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: list[int]) -> str: ...


class HFTokenizer:
    """Wraps a Hugging Face tokenizer.json via the fast `tokenizers` library."""

    def __init__(self, model_dir: str | Path) -> None:
        from tokenizers import Tokenizer as _Tok

        self._tok = _Tok.from_file(str(Path(model_dir) / "tokenizer.json"))

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids  # adds BOS through the post-processor

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids, skip_special_tokens=True)


class IncrementalDetokenizer:
    """Turns a growing list of token ids into text deltas.

    Decoding token by token breaks characters whose UTF-8 bytes span several
    tokens and drops the spaces SentencePiece attaches to word starts. So we
    decode the whole output each time and emit only the new suffix, holding it
    back while it ends in an incomplete character (U+FFFD).

    The last few prompt tokens are decoded in front of the output as context:
    decoded alone, a first token like "_Paris" loses its leading space.
    """

    def __init__(self, tokenizer: Tokenizer, prompt_ids: list[int] | None = None) -> None:
        self.tok = tokenizer
        self.prefix = list(prompt_ids or [])[-4:]
        self.prefix_text = tokenizer.decode(self.prefix) if self.prefix else ""
        self.ids: list[int] = []
        self.text = ""

    def push(self, token: int) -> str:
        self.ids.append(token)
        full = self.tok.decode(self.prefix + self.ids)
        if full.endswith("\ufffd"):
            return ""
        if full.startswith(self.prefix_text):
            full = full[len(self.prefix_text) :]
        delta = full[len(self.text) :] if full.startswith(self.text) else full
        self.text = full
        return delta
