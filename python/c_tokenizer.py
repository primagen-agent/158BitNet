"""Exact tokenizer bridge backed by the C runtime."""
from __future__ import annotations

import subprocess


class CTokenizer:
    """Keep one tokenizer probe process alive for repeated encodes."""

    def __init__(self, probe_bin: str, model_path: str):
        self.probe = probe_bin
        self.model = model_path
        self._proc = subprocess.Popen(
            [probe_bin, model_path, "--serve-esc"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )

    def encode(self, text: str, add_bos: bool) -> list[int]:
        escaped = text.replace("\\", "\\\\").replace("\n", "\\n")
        self._proc.stdin.write(escaped.encode("utf-8") + b"\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline().decode()
        token_ids = [int(value) for value in line.split() if value.strip()]
        if add_bos:
            token_ids = [self.bos()] + token_ids
        return token_ids

    def bos(self) -> int:
        if not hasattr(self, "_bos"):
            self._proc.stdin.write(b"BOS?\n")
            self._proc.stdin.flush()
            self._bos = int(
                self._proc.stdout.readline().decode().split()[0])
        return self._bos

    def eos(self) -> int:
        self._proc.stdin.write(b"EOS?\n")
        self._proc.stdin.flush()
        return int(
            self._proc.stdout.readline().decode().split()[0])

    def decode_pieces(self, token_ids: list[int]) -> list[bytes]:
        """Exact runtime token bytes, including context-dependent boundaries."""
        if not hasattr(self, "_pieces"):
            output = subprocess.check_output(
                [self.probe, self.model, "--dump-vocab"],
                text=True, stderr=subprocess.DEVNULL,
            )
            self._pieces = {}
            for line in output.splitlines():
                token, _, encoded = line.partition(" ")
                self._pieces[int(token)] = bytes.fromhex(encoded)
        return [self._pieces[int(token)] for token in token_ids]
